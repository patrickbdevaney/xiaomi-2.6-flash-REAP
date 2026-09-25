"""Publish the pruned checkpoint to the Hugging Face Hub, unattended.

This runs with nobody watching, at the end of a 30-hour pass, and it uploads ~100 GB to a
service that caches and indexes what it receives. A half-written shard, a checkpoint that is
secretly a copy of the source, or a model that cannot chat because one small file was not
carried over are all things that LOOK fine in a directory listing. So every check below is a
refusal, not a warning, and the upload is the last thing that happens.

Visibility is a deliberate choice, not a default: publishing weights cannot be undone. This
pipeline runs with --public (set in reap-publish.service) because a ~100 GB checkpoint does not
fit the free tier's PRIVATE storage quota -- private would mean the upload fails, not that it
stays safe. Drop HF_PUBLIC from the unit to go back to a private repo.
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import sys
from pathlib import Path

REQUIRED_FILES = (
    "config.json", "model.safetensors.index.json", "generation_config.json",
    "modeling_mimo_v2.py", "configuration_mimo_v2.py",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "preprocessor_config.json",
)
REQUIRED_DIRS = ("audio_tokenizer",)
SKIP = {"__pycache__", ".git", ".cache"}


class Refuse(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSING TO PUBLISH: {msg}")


def safetensors_span(path: Path) -> int:
    """Bytes the header says the file must be. Catches a truncated shard, which is what an
    interrupted 100 GB write leaves behind and what a size-only check waves through."""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise Refuse(f"{path.name} is shorter than a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 200_000_000:
            raise Refuse(f"{path.name} declares an implausible header length {n}")
        head = json.loads(fh.read(n))
    end = 0
    for k, v in head.items():
        if k == "__metadata__":
            continue
        end = max(end, int(v["data_offsets"][1]))
    return 8 + n + end


def preflight(dst: Path, repo_root: Path, expect_experts: int | None) -> dict:
    stage = (repo_root / "logs" / ".stage")
    if not stage.exists() or stage.read_text().strip() != "done":
        raise Refuse(f"the pipeline has not finished (logs/.stage = "
                     f"{stage.read_text().strip() if stage.exists() else 'absent'}). "
                     f"Only a run that reached stage 7 may be published.")
    if not dst.is_dir():
        raise Refuse(f"{dst} does not exist")

    cfg = json.loads((dst / "config.json").read_text())
    if "_reap" not in cfg:
        raise Refuse("config.json has no _reap block -- this is not a pruned checkpoint, it is "
                     "a copy of the source. Publishing it would ship the wrong model.")
    if cfg.get("_ragged_experts"):
        raise Refuse("this checkpoint has a per-layer expert count. transformers, vLLM and GGUF "
                     "all read a single n_expert, so nobody could load what would be uploaded.")
    n_now, n_before = int(cfg["n_routed_experts"]), int(cfg["_reap"]["experts_before"])
    if n_now >= n_before:
        raise Refuse(f"n_routed_experts {n_now} is not below the source's {n_before}")
    if expect_experts is not None and n_now != expect_experts:
        raise Refuse(f"n_routed_experts is {n_now}, expected {expect_experts}")

    for f in REQUIRED_FILES:
        if not (dst / f).is_file():
            raise Refuse(f"{f} is missing -- a checkpoint without it does not load or does not "
                         f"chat. Check the copy list at the end of apply_mask.py.")
    for d in REQUIRED_DIRS:
        if not (dst / d).is_dir():
            raise Refuse(f"{d}/ is missing -- it is loaded by name from modeling_mimo_v2.py, so "
                         f"uploading without it publishes a model with a dead modality.")

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    referenced = sorted(set(wmap.values()))
    on_disk = sorted(p.name for p in dst.glob("*.safetensors"))
    missing = [s for s in referenced if not (dst / s).is_file()]
    if missing:
        raise Refuse(f"{len(missing)} shard(s) in the index are not on disk: {missing[:3]}")
    stray = [s for s in on_disk if s not in set(referenced)]
    if stray:
        raise Refuse(f"{len(stray)} shard(s) on disk are not in the index: {stray[:3]}. An "
                     f"unreferenced shard means the write was interrupted or restarted.")
    partial = [p.name for p in dst.rglob("*") if p.suffix in (".tmp", ".incomplete", ".part")]
    if partial:
        raise Refuse(f"partial files present: {partial[:5]}")

    total = 0
    for s in referenced:
        p = dst / s
        want, have = safetensors_span(p), p.stat().st_size
        if have < want:
            raise Refuse(f"{s} is TRUNCATED: header describes {want} bytes, file is {have}")
        total += have

    # Every expert index in the index must fit the new count. This is the check that would have
    # caught a mask applied to the wrong layer, which nothing downstream looks at.
    bad = []
    for k in wmap:
        if ".experts." in k:
            try:
                ei = int(k.split(".experts.")[1].split(".")[0])
            except ValueError:
                continue
            if ei >= n_now:
                bad.append(k)
    if bad:
        raise Refuse(f"{len(bad)} tensor(s) name an expert index >= {n_now}: {bad[:3]}")

    return {"experts": n_now, "experts_before": n_before, "shards": len(referenced),
            "bytes": total, "tensors": len(wmap), "config": cfg}


def model_card(dst: Path, repo_root: Path, pre: dict, repo_id: str) -> str:
    masks = repo_root / "artifacts" / "masks"
    m = json.loads((masks / "mask.json").read_text()) if (masks / "mask.json").exists() else {}
    kd_path = masks / "router_kd_state.json"
    kd = json.loads(kd_path.read_text()) if kd_path.exists() else {}
    src = Path(pre["config"]["_reap"]["source"]).name
    gib = pre["bytes"] / 2**30
    ratio = 1 - pre["experts"] / pre["experts_before"]

    rows = ""
    for b, v in sorted(m.get("retention_by_domain", {}).items(), key=lambda kv: kv[1]):
        rows += f"| {b} | {v:.4f} |\n"
    kd_layers = kd.get("layers", {})
    kept_teacher = sum(1 for v in kd_layers.values() if v.get("kept_teacher"))
    imp = [v["improvement"] for v in kd_layers.values() if not v.get("kept_teacher")]
    kd_line = (f"{len(kd_layers)} routers refitted, {kept_teacher} kept at the teacher weights "
               f"(the fit did not beat the baseline), median improvement "
               f"{sorted(imp)[len(imp)//2]:+.1%} on the kept experts."
               if kd_layers else "not applied.")

    return f"""---
license: mit
base_model: XiaomiMiMo/{src}
library_name: transformers
tags:
- moe
- pruned
- reap
- hope
- mimo_v2
- multimodal
---

# {repo_id.split('/')[-1]}

`XiaomiMiMo/{src}` with **{ratio:.0%} of its routed experts removed** — {pre['experts_before']}
experts per layer down to **{pre['experts']}** — so that it fits and serves on a single
NVIDIA Jetson AGX Thor (117 GiB unified memory). {gib:.1f} GiB across {pre['shards']} shards.

Vision, audio and video input are preserved; `audio_tokenizer/` ships with the checkpoint.

## How the experts were chosen

Not by activation frequency. Expert saliency was accumulated over a calibration corpus and the
prune set was solved with **HOPE**, which minimises the output error a prune set actually causes
including the *interaction* terms between experts — REAP is the same objective with the
off-diagonal zeroed, and that off-diagonal cannot be recovered after the pass.

| setting | value |
|---|---|
| objective | `{m.get('mode', 'hope')}` |
| saliency criterion | `{m.get('criterion', '-')}` |
| prune ratio | {m.get('ratio', ratio):.2f}, uniform across layers |
| per-domain protection | top {m.get('protect_frac', 0):.0%} of every domain held out of the prune set |
| worst domain retained | {m.get('worst_retention', float('nan')):.4f} ({m.get('worst_domain', '-')}) |
| mean retained | {m.get('mean_retention', float('nan')):.4f} |
| HOPE objective pᵀFp | {m.get('interaction_cost', float('nan')):.5f} |

Selection is scored per domain and ranked by the **worst** one, never the mean: an average is
how a criterion that destroys one capability outscores one that preserves all of them.

### Retained gated output mass, by calibration domain

| domain | retention |
|---|---|
{rows}
## Routers

Pruning an expert leaves its router column behind. The routers were refitted by output matching
against the unpruned teacher, routers only, every expert frozen: {kd_line}

A refit that failed to beat the untouched baseline was discarded in favour of the baseline, so
no router here is worse than simply slicing the teacher's.

## Limitations

- Calibration was English/Chinese text, code, math, science, finance, agentic traces, and
  image/audio/video captions. Domains outside that mix were not measured.
- The `dflash/` speculative-decoding draft head from the source repo is **not** included: it was
  trained against the unpruned expert set and is not valid for this checkpoint.
- Pruned MoE experts do not come back. This is a lossy, irreversible transform of the base model.

## Provenance

Produced by [{repo_id.split('/')[0]}/xiaomi-2.6-flash-REAP](https://github.com/patrickbdevaney/xiaomi-2.6-flash-REAP)
on a single Jetson AGX Thor. MIT, inherited from the base model — attribution to Xiaomi MiMo.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=str(Path.home() / "models" / "MiMo-V2.6-Flash-REAP50"))
    ap.add_argument("--repo", default="patrickbdevaney/MiMo-V2.6-Flash-REAP50")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--expect-experts", type=int, default=128)
    ap.add_argument("--public", action="store_true",
                    help="create the repo public. Off by default: publishing weights is the one "
                         "step in this script that cannot be undone.")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="preflight and card only, no network")
    a = ap.parse_args()

    dst, root = Path(a.dst), Path(a.repo_root)
    pre = preflight(dst, root, a.expect_experts)
    print(f"preflight OK: {pre['experts']}/{pre['experts_before']} experts, {pre['shards']} "
          f"shards, {pre['tensors']} tensors, {pre['bytes']/2**30:.1f} GiB", flush=True)

    card = model_card(dst, root, pre, a.repo)
    upstream = dst / "README.md"
    if upstream.exists() and not (dst / "README_upstream.md").exists():
        upstream.rename(dst / "README_upstream.md")   # kept, never deleted
    (dst / "README.md").write_text(card)
    receipt = {"repo": a.repo, "private": not a.public, **{k: v for k, v in pre.items()
                                                           if k != "config"}}
    (root / "artifacts" / "publish_receipt.json").write_text(json.dumps(receipt, indent=1))

    if a.dry_run:
        print("DRY RUN -- nothing uploaded. Card written to", dst / "README.md")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()["name"]
    print(f"authenticated as {who}", flush=True)
    api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
    if a.public:
        # exist_ok=True does NOT change the visibility of a repo that already exists, so a repo
        # created private by an earlier attempt would stay private and the upload would fail
        # against the free tier's private quota with no obvious cause. Say it outright.
        api.update_repo_settings(a.repo, repo_type="model", private=False)
    print(f"uploading {pre['bytes']/2**30:.1f} GiB to {a.repo} "
          f"({'PUBLIC' if a.public else 'private'})", flush=True)
    api.upload_large_folder(
        repo_id=a.repo, repo_type="model", folder_path=str(dst),
        ignore_patterns=["__pycache__/*", "*.tmp", "*.part", "*.pt"],
        num_workers=a.workers, print_report=True)
    print(f"DONE: https://huggingface.co/{a.repo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
