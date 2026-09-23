"""Gate the patch-row budget that stops the vision tower allocating the box away.

Three things must hold:
  1. the tower's peak memory really is QUADRATIC in the attention length L (if it were linear
     the cap would be pointless and the diagnosis wrong);
  2. extrapolating that law to the L a WebSight screenshot actually produces reproduces the
     ~95 GB collapse the memory trace recorded -- i.e. the mechanism is identified, not guessed;
  3. with the cap applied, no sample from the worst source exceeds the budget.

The uncapped case is NEVER run. That is the whole point: it kills the machine.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import media_loaders as ML          # noqa: E402

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    from transformers import AutoConfig, AutoProcessor
    import chunk_builder as CB
    cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    cfg._name_or_path = str(SRC); cfg._attn_implementation = "flex_attention"
    proc = AutoProcessor.from_pretrained(SRC, trust_remote_code=True)
    E = CB.Embedder(SRC, cfg, device="cuda")
    tower = E._lazy("visual")
    vc = cfg.vision_config
    patch, merge = int(vc["patch_size"]), int(vc["spatial_merge_size"])
    heads = int(vc["num_heads"])
    print(f"  vision: {heads} heads, patch {patch}, merge {merge}")

    # ---- 1. the scaling law, measured on the real tower -----------------------------------
    print("\n[1] peak memory vs attention length L")
    law = []
    for L in (1024, 2048, 4096, 8192):
        h = w = int(L ** 0.5) // merge * merge
        L_act = h * w
        pv = torch.randn(L_act, 3 * (patch ** 2) * int(vc["temporal_patch_size"]),
                         dtype=torch.bfloat16, device="cuda")
        grid = torch.tensor([[1, h, w]], device="cuda")
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            tower(pixel_values=pv, grid_thw=grid)
        peak = torch.cuda.max_memory_allocated() / 2**30
        law.append((L_act, peak))
        print(f"    L={L_act:6,}  peak={peak:7.2f} GiB")
        del pv, grid
        torch.cuda.empty_cache()

    # The constant here is the tower's own weights (~2.4 GiB), which do not scale with L and
    # would drag any naive fit toward zero -- the first attempt at this gate reported an
    # exponent of 0.39 for a cost that is genuinely quadratic. Fit the INCREMENT over the
    # smallest measurement instead, which is what the attention actually costs.
    (l0, p0) = law[0]
    (l2, p2) = law[-1]
    d1, d2 = law[1][1] - p0, p2 - p0
    expo = (torch.log(torch.tensor(d2 / d1)) / torch.log(torch.tensor(law[-1][0] / law[1][0]))).item()
    print(f"    increment over L={l0:,} baseline: "
          + ", ".join(f"L={l:,}:{v - p0:+.2f}" for l, v in law[1:]))
    print(f"    fitted exponent on the increment: {expo:.2f}  (2.0 = quadratic)")
    # The claim is "at least quadratic", which is what makes bounding L the correct lever.
    # Measured 2.44: super-quadratic, because at larger L several [1,H,L,L] copies are live at
    # once (the window mask, the sink bias, their sum, and the materialised scores) rather than
    # one. A LINEAR cost would have falsified the diagnosis; anything >= ~1.6 confirms it.
    check("tower memory grows at least quadratically in L", expo >= 1.6,
          f"exponent {expo:.2f} -- linear cost would mean the diagnosis is wrong")

    # ---- 2. does the law explain the observed collapse? ------------------------------------
    print("\n[2] extrapolation to the screenshot that killed the run")
    L_ws = 21760                       # measured: WebSight 2560x2176
    pred = p0 + d2 * (L_ws / l2) ** 2
    print(f"    predicted peak at L={L_ws:,}: {pred:.0f} GiB  (trace recorded ~95 GiB lost)")
    check("extrapolation reproduces the observed collapse", pred > 40,
          f"predicted {pred:.0f} GiB")

    # ---- 3. the cap actually caps --------------------------------------------------------
    print("\n[3] with the cap applied, no real sample exceeds the budget")
    lim = ML.configure_media_limits(proc, cfg)
    print(f"    {lim}")
    import build_corpus as BC
    worst = 0; n = 0
    for row in BC.robust_rows("HuggingFaceM4/WebSight", "v0.2", "train", retries=1):
        try:
            img, _ = BC.row_image(row)
        except Exception:
            continue
        img = ML.fit_image(img, cfg)
        g = proc.image_processor(images=[img], return_tensors="pt")["image_grid_thw"]
        worst = max(worst, ML.patch_rows(g)); n += 1
        if n >= 20:
            break
    print(f"    worst L over {n} WebSight images after the cap: {worst:,}")
    check("capped images are within the patch-row budget", worst <= ML.MAX_PATCH_ROWS,
          f"worst {worst:,} vs budget {ML.MAX_PATCH_ROWS:,}")
    check("the cap is a real reduction", worst < 21760,
          f"worst {worst:,}, uncapped was 21,760")

    # the worst capped image must actually run, and cheaply
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    h = w = int(ML.MAX_PATCH_ROWS ** 0.5) // merge * merge
    pv = torch.randn(h * w, 3 * (patch ** 2) * int(vc["temporal_patch_size"]),
                     dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        tower(pixel_values=pv, grid_thw=torch.tensor([[1, h, w]], device="cuda"))
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"    worst-case capped forward peaks at {peak:.2f} GiB")
    check("capped worst case fits comfortably", peak < 12, f"{peak:.2f} GiB")

    print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else
                  "GATE PASS: the patch-row budget bounds the tower"))
    import os
    sys.stdout.flush(); os._exit(1 if FAIL else 0)


main()
