"""Does calibration sequence length change WHICH experts look salient?

The question matters because the pass is about to run at a fixed S, and if saliency is
S-sensitive then a short S silently calibrates for short context and the REAP degrades exactly
the 1M-context capability the model was chosen for.

MiMo makes the question sharper than usual. 39 of its 48 layers are SWA with a window of 128, so
their attention context is capped at 128 tokens NO MATTER WHAT S IS -- structurally, S cannot
change what those layers see beyond position 128. Only the 9 full-attention layers accumulate
long range. So the prediction is: SWA layers should be nearly S-invariant, GA layers should not.

That is a prediction with a sign, which is what makes this a real test (CLAUDE.md §6) rather
than a fishing trip. It runs the SAME TOKENS at several S and compares the resulting per-layer
expert rankings.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import calib_pass as CP
import mimo_saliency as MS
from mimo_shards import ShardReader

SRC = Path.home() / "models" / "MiMo-V2.6-Flash-RL"
DEV = "cuda"
SEQS = [1024, 4096, 16384]
TOTAL = 16384                      # same token budget at every S
LAYERS = list(range(0, 7))         # through layer 5, the first GA layer that is also MoE


def main():
    from transformers import AutoConfig, AutoTokenizer
    from datasets import load_dataset
    cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    cfg._name_or_path = str(SRC); cfg._attn_implementation = "flex_attention"
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

    # Real long text, from the ballast bucket's own source -- a synthetic repeat would have no
    # long-range structure and would answer the wrong question.
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                      streaming=True)
    buf = []
    for row in ds:
        buf.extend(tok(row["text"], add_special_tokens=False)["input_ids"])
        if len(buf) >= TOTAL:
            break
    ids_all = torch.tensor(buf[:TOTAL], dtype=torch.long)
    del ds
    print(f"{TOTAL} real tokens; testing S in {SEQS}", flush=True)

    from chunk_builder import Embedder
    E = Embedder(SRC, cfg, device=DEV)
    base = E.embed(ids_all.to(DEV)).to(torch.bfloat16)      # [TOTAL, H], embedded once
    del E
    torch.cuda.empty_cache()

    reader = ShardReader(SRC)
    scores = {}                                             # (S, layer) -> REAP score vector
    for S in SEQS:
        MS.ACC.clear()
        MS.configure(["x"], n_layers=cfg.num_hidden_layers, n_experts=cfg.n_routed_experts,
                     device=DEV)
        MS.LAYER_INDEX.update({f"model.layers.{i}.mlp": i for i in LAYERS})
        MS.patch(CP._modeling(cfg))
        hs = base.view(TOTAL // S, S, -1).clone()
        masks = CP.masks_for(S, cfg.sliding_window, DEV, torch.bfloat16)
        pos = torch.arange(S, device=DEV)[None]
        for li in LAYERS:
            layer = CP.build_layer(cfg, li, reader, torch.bfloat16).to(DEV)
            pe = CP._rope(cfg, layer.attention_type, hs[:1], pos, DEV, torch.bfloat16)
            with torch.no_grad():
                for b in range(hs.shape[0]):
                    MS.CTX.update({"layer": f"model.layers.{li}.mlp", "bucket": 0, "valid": None})
                    hs[b:b+1] = layer(hs[b:b+1], attention_mask=masks[layer.attention_type],
                                      position_ids=pos, position_embeddings=pe)
            del layer; reader.release(); torch.cuda.empty_cache()
        for li in LAYERS:
            k = f"model.layers.{li}.mlp"
            if k in MS.ACC:
                a = MS.ACC[k]
                scores[(S, li)] = (a["sum"].sum(0) / a["cnt"].sum(0).clamp(min=1)).cpu().numpy()
        print(f"  S={S:6d} done", flush=True)

    ref = SEQS[-1]
    print(f"\nagreement with S={ref} (the longest), per layer:")
    print(f"  {'layer':>5} {'type':>4} {'spearman':>9} {'top-128 overlap':>16}")
    from scipy.stats import spearmanr
    rows = []
    for li in LAYERS:
        if (ref, li) not in scores:
            continue
        hp = cfg.hybrid_layer_pattern[li]
        typ = "SWA" if hp == 1 else "GA"
        r = scores[(ref, li)]
        for S in SEQS[:-1]:
            v = scores[(S, li)]
            live = (r > 0) | (v > 0)
            rho = spearmanr(r[live], v[live]).statistic if live.sum() > 2 else float("nan")
            ov = len(set(np.argsort(-r)[:128]) & set(np.argsort(-v)[:128])) / 128
            rows.append((li, typ, S, rho, ov))
            print(f"  {li:5d} {typ:>4}  S={S:<6d} rho {rho:+.4f}   overlap {ov:.3f}")
    swa = [r for r in rows if r[1] == "SWA"]
    ga = [r for r in rows if r[1] == "GA"]
    if swa and ga:
        print(f"\n  SWA mean rho {np.mean([r[3] for r in swa]):+.4f}  "
              f"overlap {np.mean([r[4] for r in swa]):.3f}")
        print(f"  GA  mean rho {np.mean([r[3] for r in ga]):+.4f}  "
              f"overlap {np.mean([r[4] for r in ga]):.3f}")
    sys.stdout.flush()
    os._exit(0)


main()
