"""Gate the Hub publisher's refusals.

The publisher runs unattended and uploads ~100 GB to a service that caches and indexes what it
receives, so its value is entirely in what it REFUSES. Every case below is a checkpoint that
would pass a directory listing and be wrong on the Hub forever. Nothing here touches the network.
"""
from __future__ import annotations
import json, struct, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import publish_hf as P          # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def refuses(name, dst, root, expect=128, needle=None):
    try:
        P.preflight(Path(dst), Path(root), expect)
    except SystemExit as e:
        ok = needle is None or needle in str(e)
        check(name, ok, str(e)[:120])
        return
    check(name, False, "preflight ACCEPTED it")


def write_shard(path: Path, nbytes=64, truncate=0):
    head = json.dumps({"t": {"dtype": "F32", "shape": [nbytes // 4],
                             "data_offsets": [0, nbytes]}}).encode()
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(head)))
        fh.write(head)
        fh.write(b"\0" * (nbytes - truncate))


_N = [0]


def build(td: Path, *, experts=128, ragged=False, reap=True, shards=2,
          drop=(), stray=False, truncate=0, bad_expert=False) -> Path:
    # A FRESH directory per case. Reusing one leaks the previous case's damage into the next,
    # which is how a gate ends up reporting the same refusal for every check and passing anyway.
    _N[0] += 1
    dst = td / f"ckpt{_N[0]:02d}"
    (dst / "audio_tokenizer").mkdir(parents=True, exist_ok=True)
    (dst / "audio_tokenizer" / "config.json").write_text("{}")
    cfg = {"n_routed_experts": experts, "num_hidden_layers": 4}
    if reap:
        cfg["_reap"] = {"source": "/models/MiMo-V2.6-Flash-RL", "experts_before": 256,
                        "mask": "m.json", "uniform": True}
    if ragged:
        cfg["_ragged_experts"] = True
    for f in P.REQUIRED_FILES:
        if f in drop:
            continue
        (dst / f).write_text("{}" if f.endswith(".json") else "x")
    (dst / "config.json").write_text(json.dumps(cfg))
    wmap = {}
    for i in range(shards):
        s = f"model-{i:05d}.safetensors"
        write_shard(dst / s, truncate=truncate if i == 0 else 0)
        wmap[f"model.layers.{i}.mlp.experts.{experts - 1}.gate_proj.weight"] = s
        wmap[f"model.layers.{i}.self_attn.q_proj.weight"] = s
    if bad_expert:
        wmap[f"model.layers.0.mlp.experts.{experts}.gate_proj.weight"] = "model-00000.safetensors"
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wmap}))
    if stray:
        write_shard(dst / "model-09999.safetensors")
    for f in drop:
        (dst / f).unlink(missing_ok=True)
    return dst


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    root = td / "repo"
    (root / "logs").mkdir(parents=True)
    (root / "artifacts" / "masks").mkdir(parents=True)
    (root / "logs" / ".stage").write_text("stage7-apply\n")

    print("[1] a run that has not finished is never published")
    dst = build(td)
    refuses("stage marker is not 'done'", dst, root, needle="has not finished")
    (root / "logs" / ".stage").write_text("done\n")

    print("\n[2] a checkpoint that is secretly the SOURCE is refused")
    refuses("no _reap block in config.json", build(td, reap=False), root,
            needle="not a pruned checkpoint")

    print("\n[3] a checkpoint nobody could load is refused")
    refuses("ragged per-layer expert count", build(td, ragged=True), root,
            needle="single n_expert")
    refuses("expert count did not go down", build(td, experts=256), root, 256,
            needle="not below")
    refuses("expert count is not the one asked for", build(td, experts=64), root, 128,
            needle="expected 128")

    print("\n[4] a checkpoint that loses a capability is refused")
    refuses("chat_template.jinja missing", build(td, drop=("chat_template.jinja",)), root,
            needle="chat_template.jinja")
    refuses("preprocessor_config.json missing", build(td, drop=("preprocessor_config.json",)),
            root, needle="preprocessor_config.json")
    d = build(td)
    for p in sorted((d / "audio_tokenizer").rglob("*"), reverse=True):
        p.unlink()
    (d / "audio_tokenizer").rmdir()
    refuses("audio_tokenizer/ missing", d, root, needle="dead modality")

    print("\n[5] a half-written upload candidate is refused")
    refuses("a shard is truncated", build(td, truncate=16), root, needle="TRUNCATED")
    refuses("a shard is not in the index", build(td, stray=True), root, needle="not in the index")
    d = build(td)
    (d / "model-00001.safetensors").unlink()
    refuses("an indexed shard is absent", d, root, needle="not on disk")
    d = build(td)
    (d / "model-00000.safetensors.tmp").write_text("x")
    refuses("a .tmp file is present", d, root, needle="partial files")

    print("\n[6] a mask applied to the wrong expert set is refused")
    refuses("a tensor names an expert index past the new count", build(td, bad_expert=True),
            root, needle="expert index >=")

    print("\n[7] a well-formed checkpoint passes, and the card carries the real numbers")
    dst = build(td)
    pre = P.preflight(dst, root, 128)
    check("preflight returns the expert counts", (pre["experts"], pre["experts_before"]) == (128, 256),
          f"{pre['experts']}/{pre['experts_before']}")
    check("it counts both shards", pre["shards"] == 2, str(pre["shards"]))
    check("bytes equal the files on disk",
          pre["bytes"] == sum((dst / f"model-{i:05d}.safetensors").stat().st_size for i in range(2)),
          str(pre["bytes"]))
    (root / "artifacts" / "masks" / "mask.json").write_text(json.dumps({
        "mode": "hope", "criterion": "reap_1_1_1", "ratio": 0.5, "protect_frac": 0.08,
        "worst_domain": "audio", "worst_retention": 0.99392, "mean_retention": 0.99653,
        "interaction_cost": 0.015497,
        "retention_by_domain": {"audio": 0.9939, "video": 0.994, "code": 0.9984}}))
    (root / "artifacts" / "masks" / "router_kd_state.json").write_text(json.dumps({
        "layers": {"a": {"improvement": 0.1, "kept_teacher": False},
                   "b": {"improvement": 0.0, "kept_teacher": True}}}))
    card = P.model_card(dst, root, pre, "u/MiMo-V2.6-Flash-REAP50")
    for needle in ("license: mit", "0.9939", "top 8% of every domain", "hope", "audio_tokenizer",
                   "1 kept at the teacher weights", "dflash"):
        check(f"card states {needle!r}", needle in card)
    check("card does not invent a public URL claim", "PUBLIC" not in card)

print("\n[8] the shipped configuration says what it is going to do")
unit = Path.home() / ".config" / "systemd" / "user" / "reap-publish.service"
sh = (Path(__file__).parent / "await_and_publish.sh").read_text()
check("the watcher maps HF_PUBLIC=1 to --public",
      'HF_PUBLIC:-0}" = "1" ] && PUBLIC_FLAG="--public"' in sh)
check("visibility is never assumed -- the log states which one happened",
      'PUBLISHED ($vis)' in sh)
if unit.exists():
    u = unit.read_text()
    check("the installed unit carries a visibility decision",
          "HF_PUBLIC=1" in u or "HF_PUBLIC" not in u,
          "HF_PUBLIC=1" if "HF_PUBLIC=1" in u else "absent -> private")
else:
    check("reap-publish.service is installed", False, str(unit))

print("\nGATE " + ("FAIL: " + ", ".join(FAIL) if FAIL else "PASS"))
sys.exit(1 if FAIL else 0)
