"""Materialise the pruned checkpoint from a mask.

THE EXPERTS ARE NEVER DEQUANTISED. MiMo stores each expert as its own tensor --
`model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.weight` plus a `.weight_scale` -- so
pruning is dropping whole tensors and renumbering the survivors. The MXFP4 bytes are copied
VERBATIM. That is not merely an optimisation: dequantising 256 experts across 47 layers would
materialise a second full-precision copy of a 173 GB model on a box with 43 GB free, and the
routed experts are already FP4, so a round trip could only lose precision, never add it.

WHAT MUST BE SLICED, NOT COPIED
-------------------------------
Two per-layer tensors are indexed BY EXPERT and must be cut to the kept set, in the same order
the experts are renumbered:
    mlp.gate.weight                  [n_experts, hidden]   the router
    mlp.gate.e_score_correction_bias [n_experts]           the noaux_tc bias
Get the order wrong and the router points at the wrong experts -- a failure that loads cleanly,
runs at full speed, and produces confident nonsense. The renumbering map is therefore built once
and used for both the tensor names and the two slices.

UNIFORM BUDGETS ONLY, BY DEFAULT
--------------------------------
`n_routed_experts` is a scalar read globally by the modeling code, and llama.cpp reads a single
`n_expert`, so a per-layer count needs patched modeling and cannot be a GGUF. `--allow-ragged`
writes it anyway and records the per-layer counts in the config, clearly marked as non-portable.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

EXPERT_RE = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts\.)(\d+)(\..+)$")
GATE_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$")


def load_mask(path: Path) -> dict[int, list[int]]:
    d = json.loads(path.read_text())
    mask = d["mask"] if "mask" in d else d
    return {int(k.split(".")[2]): sorted(int(i) for i in v) for k, v in mask.items()}


def keep_maps(pruned: dict[int, list[int]], n_exp: int) -> dict[int, dict[int, int]]:
    """layer -> {old_expert_index: new_expert_index} for the KEPT experts, order preserved."""
    out = {}
    for li, pr in pruned.items():
        drop = set(pr)
        kept = [e for e in range(n_exp) if e not in drop]
        out[li] = {old: new for new, old in enumerate(kept)}
    return out


def load_router_kd(path: Path | None):
    """{layer_index: (weight, bias, keep)} from stage 6, or None.

    WITHOUT THIS THE WHOLE OF STAGE 6 IS A NO-OP. apply_mask originally sliced the teacher's
    router and never looked at router_kd.pt, so the pipeline would train routers for hours, save
    them, and then write a checkpoint that did not contain them -- succeeding end to end with
    the repair silently absent.
    """
    if path is None:
        return None
    import torch
    d = torch.load(Path(path), map_location="cpu")
    out = {}
    for name, v in d.items():
        out[int(name.split(".")[2])] = (v["weight"], v["bias"], v["keep"])
    return out


def run(src: Path, dst: Path, mask_path: Path, allow_ragged: bool = False,
        dry_run: bool = False, router_kd: Path | None = None) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.loads((src / "config.json").read_text())
    n_exp = int(cfg["n_routed_experts"])
    pruned = load_mask(mask_path)
    kmap = keep_maps(pruned, n_exp)
    KD = load_router_kd(router_kd)
    if KD is not None:
        missing = sorted(set(kmap) - set(KD))
        if missing:
            raise SystemExit(
                f"router KD covers {len(KD)} layers but the mask prunes {len(kmap)}; "
                f"{len(missing)} would keep the UNTRAINED sliced router while the rest are "
                f"repaired, which is a checkpoint nobody measured: layers {missing[:8]}. "
                f"Finish stage 6 (it is resumable per layer) or drop --router-kd.")
        # VALIDATE THE KEEP ORDER BEFORE ANY SHARD IS WRITTEN. This check used to live inside the
        # shard loop, so a mismatch surfaced partway through -- after tens of GB had already been
        # written -- leaving a half-materialised checkpoint that looks like a real one. It costs
        # nothing here and it is also the only place a --dry-run can catch it.
        for li, km in sorted(kmap.items()):
            order = [old for old, _ in sorted(km.items(), key=lambda kv: kv[1])]
            trained = [int(x) for x in KD[li][2]]
            if trained != order:
                bad = next(i for i, (a, b) in enumerate(zip(trained, order)) if a != b)
                raise SystemExit(
                    f"router KD for layer {li} was trained against a different kept set than "
                    f"this mask selects (first mismatch at position {bad}: trained kept expert "
                    f"{trained[bad]}, mask keeps {order[bad]}). The router would point at the "
                    f"wrong experts -- the checkpoint would load, run at full speed and emit "
                    f"confident nonsense. Re-run stage 6 against this mask, or drop --router-kd. "
                    f"Nothing has been written.")

    counts = {li: n_exp - len(p) for li, p in pruned.items()}
    uniform = len(set(counts.values())) == 1
    if not uniform and not allow_ragged:
        raise SystemExit(
            f"per-layer expert counts differ {sorted(set(counts.values()))} but "
            f"`n_routed_experts` is a single scalar read globally by the modeling code (and by "
            f"llama.cpp). This checkpoint would not load in transformers, vLLM or as a GGUF. "
            f"Re-run with a uniform budget, or pass --allow-ragged to write a "
            f"custom-server-only checkpoint.")

    index = json.loads((src / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in wmap.items():
        by_shard.setdefault(shard, []).append(name)

    stats = {"copied": 0, "dropped": 0, "renumbered": 0, "sliced": 0,
             "experts_before": n_exp, "experts_after": counts, "uniform": uniform}
    if dry_run:
        for shard, names in by_shard.items():
            for name in names:
                m = EXPERT_RE.match(name)
                if m:
                    li, e = int(m.group(2)), int(m.group(3))
                    if li in kmap:
                        if e in kmap[li]:
                            stats["renumbered"] += 1
                        else:
                            stats["dropped"] += 1
                        continue
                if GATE_RE.match(name) and int(GATE_RE.match(name).group(1)) in kmap:
                    stats["sliced"] += 1
                    continue
                stats["copied"] += 1
        return stats

    dst.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    for shard, names in sorted(by_shard.items()):
        out_tensors = {}
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for name in names:
                m = EXPERT_RE.match(name)
                if m:
                    li, e = int(m.group(2)), int(m.group(3))
                    if li in kmap:
                        if e not in kmap[li]:
                            stats["dropped"] += 1
                            continue
                        new_name = f"{m.group(1)}{kmap[li][e]}{m.group(4)}"
                        out_tensors[new_name] = f.get_tensor(name)   # MXFP4 bytes, verbatim
                        stats["renumbered"] += 1
                        continue
                g = GATE_RE.match(name)
                if g and int(g.group(1)) in kmap:
                    li = int(g.group(1))
                    order = [old for old, _ in sorted(kmap[li].items(), key=lambda kv: kv[1])]
                    t = f.get_tensor(name)
                    if KD is not None and li in KD:
                        w, b, keep = KD[li]
                        # The trained router is indexed in the STUDENT's order. If that order is
                        # not the order the experts are being renumbered in, the router points at
                        # the wrong experts -- a checkpoint that loads, runs at full speed and
                        # emits confident nonsense. Refuse rather than reconcile.
                        rep = w if g.group(2) == "weight" else b
                        if tuple(rep.shape) != tuple(t[order].shape):
                            raise SystemExit(
                                f"router KD tensor for layer {li}.{g.group(2)} is "
                                f"{tuple(rep.shape)}, expected {tuple(t[order].shape)}")
                        out_tensors[name] = rep.to(t.dtype).contiguous()
                        stats["kd_applied"] = stats.get("kd_applied", 0) + 1
                    else:
                        out_tensors[name] = t[order].contiguous()
                    stats["sliced"] += 1
                    continue
                out_tensors[name] = f.get_tensor(name)
                stats["copied"] += 1
        if out_tensors:
            save_file(out_tensors, str(dst / shard), metadata={"format": "pt"})
            for k in out_tensors:
                new_map[k] = shard
        del out_tensors

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": index.get("metadata", {}), "weight_map": new_map}, indent=1))

    cfg["n_routed_experts"] = int(next(iter(set(counts.values())))) if uniform else n_exp
    if not uniform:
        cfg["n_routed_experts_per_layer"] = {str(k): v for k, v in sorted(counts.items())}
        cfg["_ragged_experts"] = True
    cfg["_reap"] = {"source": str(src), "mask": str(mask_path),
                    "experts_before": n_exp, "uniform": uniform}
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))

    for extra in src.glob("*"):
        if extra.suffix in (".py", ".json", ".txt", ".model") and \
                extra.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(extra, dst / extra.name)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "models" / "MiMo-V2.6-Flash-RL"))
    ap.add_argument("--dst", default=str(Path.home() / "models" / "MiMo-V2.6-Flash-REAP50"))
    ap.add_argument("--mask", default="artifacts/masks/mask.json")
    ap.add_argument("--allow-ragged", action="store_true")
    ap.add_argument("--router-kd", default=None,
                    help="artifacts/masks/router_kd.pt from stage 6; without it "
                         "the checkpoint keeps the untrained sliced router")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s = run(Path(a.src), Path(a.dst), Path(a.mask), a.allow_ragged, a.dry_run,
            router_kd=Path(a.router_kd) if a.router_kd else None)
    print(json.dumps(s, indent=1))
