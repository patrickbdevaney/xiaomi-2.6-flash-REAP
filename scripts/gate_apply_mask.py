"""Gate checkpoint materialisation.

The failure this is built to catch: the router is sliced in a different order from the one the
experts are renumbered in. That checkpoint loads cleanly, has the right shapes, the right file
count and the right size -- and every token is routed to the wrong expert. Nothing downstream
would notice, so the ordering is checked here against tensors whose identity is encoded in
their own values.
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(Path(__file__).parent))
import apply_mask as AM     # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


N_EXP, HID, L = 8, 4, 2


def build_src(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    t = {}
    for li in (1, 2):
        for e in range(N_EXP):
            # Every expert's weights carry its own index, so a renumbering bug is visible.
            t[f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight"] = \
                torch.full((2, HID), float(e), dtype=torch.float32)
            t[f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight_scale"] = \
                torch.full((2,), float(e), dtype=torch.uint8)
        # Router row e is all-e, so slicing order is checkable by value.
        t[f"model.layers.{li}.mlp.gate.weight"] = \
            torch.arange(N_EXP, dtype=torch.float32)[:, None].repeat(1, HID)
        t[f"model.layers.{li}.mlp.gate.e_score_correction_bias"] = \
            torch.arange(N_EXP, dtype=torch.float32)
        t[f"model.layers.{li}.input_layernorm.weight"] = torch.ones(HID)
    t["model.embed_tokens.weight"] = torch.ones(3, HID)
    save_file(t, str(src / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (src / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}))
    (src / "config.json").write_text(json.dumps({"n_routed_experts": N_EXP,
                                                 "num_hidden_layers": 3}))


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src, dst = td / "src", td / "dst"
    build_src(src)

    # Prune a NON-CONTIGUOUS set so a renumbering bug cannot pass by accident.
    pruned = {"model.layers.1.mlp": [1, 3, 5, 7], "model.layers.2.mlp": [0, 2, 4, 6]}
    mask = td / "mask.json"
    mask.write_text(json.dumps({"mask": pruned}))

    print("[1] dry run accounts for every tensor")
    s = AM.run(src, dst, mask, dry_run=True)
    check("4 experts dropped per layer x 2 tensors x 2 layers = 16 dropped",
          s["dropped"] == 16, str(s["dropped"]))
    check("4 experts kept per layer x 2 tensors x 2 layers = 16 renumbered",
          s["renumbered"] == 16, str(s["renumbered"]))
    check("2 router tensors per layer sliced", s["sliced"] == 4, str(s["sliced"]))

    print("\n[2] materialise")
    s = AM.run(src, dst, mask)
    out = load_file(str(dst / "model-00001-of-00001.safetensors"))
    cfg = json.loads((dst / "config.json").read_text())
    check("n_routed_experts updated to 4", cfg["n_routed_experts"] == 4,
          str(cfg["n_routed_experts"]))
    check("expert indices are contiguous 0..3",
          all(f"model.layers.1.mlp.experts.{i}.gate_proj.weight" in out for i in range(4))
          and "model.layers.1.mlp.experts.4.gate_proj.weight" not in out)

    print("\n[3] THE ORDERING CHECK: renumbered experts carry the right original weights")
    kept_l1 = [0, 2, 4, 6]        # complement of the pruned set
    got = [float(out[f"model.layers.1.mlp.experts.{n}.gate_proj.weight"][0, 0])
           for n in range(4)]
    check("expert n holds the weights of the n-th KEPT original", got == [float(e) for e in kept_l1],
          f"{got} vs {[float(e) for e in kept_l1]}")

    print("\n[4] THE ROUTER MUST BE SLICED IN THAT SAME ORDER")
    rw = out["model.layers.1.mlp.gate.weight"]
    rb = out["model.layers.1.mlp.gate.e_score_correction_bias"]
    check("router rows follow the kept order", [float(v) for v in rw[:, 0]] ==
          [float(e) for e in kept_l1], f"{[float(v) for v in rw[:,0]]}")
    check("router bias follows the kept order", [float(v) for v in rb] ==
          [float(e) for e in kept_l1], f"{[float(v) for v in rb]}")
    kept_l2 = [1, 3, 5, 7]
    rw2 = out["model.layers.2.mlp.gate.weight"]
    check("a DIFFERENT layer's mask is applied to that layer's router",
          [float(v) for v in rw2[:, 0]] == [float(e) for e in kept_l2],
          f"{[float(v) for v in rw2[:,0]]}")

    print("\n[5] quantised bytes are copied verbatim, never dequantised")
    sc = out["model.layers.1.mlp.experts.1.gate_proj.weight_scale"]
    check("weight_scale keeps its uint8 dtype", sc.dtype == torch.uint8, str(sc.dtype))
    check("weight_scale keeps the ORIGINAL expert's value (2 == kept[1])",
          int(sc[0]) == 2, str(int(sc[0])))

    print("\n[6] untouched tensors survive")
    check("layernorms copied", "model.layers.1.input_layernorm.weight" in out)
    check("embeddings copied", "model.embed_tokens.weight" in out)
    check("index lists every written tensor",
          set(json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"])
          == set(out))

    print("\n[7] a ragged budget is REFUSED unless explicitly allowed")
    ragged = td / "ragged.json"
    ragged.write_text(json.dumps({"mask": {"model.layers.1.mlp": [1, 3, 5, 7],
                                           "model.layers.2.mlp": [0, 2]}}))
    try:
        AM.run(src, td / "d2", ragged)
        check("ragged refused by default", False, "it was accepted")
    except SystemExit as e:
        check("ragged refused by default", "would not load" in str(e))
    s2 = AM.run(src, td / "d3", ragged, allow_ragged=True)
    c2 = json.loads((td / "d3" / "config.json").read_text())
    check("--allow-ragged marks the checkpoint non-portable",
          c2.get("_ragged_experts") is True and "n_routed_experts_per_layer" in c2)

print("\n" + ("GATE FAIL: " + ", ".join(FAIL) if FAIL else
              "GATE PASS: mask application preserves router-to-expert correspondence"))
sys.exit(1 if FAIL else 0)
