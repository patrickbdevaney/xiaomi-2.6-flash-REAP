# xiaomi-2.6-flash-REAP

A 50%-pruned, **omnimodal**, CUDA-servable `XiaomiMiMo/MiMo-V2.6-Flash-RL` that fits a 128 GB
unified-memory box (Jetson AGX Thor) at full 1M context.

Nothing published does this. ggml-org's own Q2_K GGUF is **126 GB** — over the envelope before a
single KV byte — and the only existing REAP50 of this model is Apple MLX and text-only.

## Why this model, and why 50%

`quantization_config.store_dtype = "mxfp4"`: the routed experts already ship as packed FP4
(158 GB of `U8`, **92.9%** of the 172.9 GB index). **The quant lever is spent**, so REAP is the
only compression lever available — which makes the quality of the REAP the entire product.

Against 125.6 GB usable:

| REAP | weights | +KV@1M | +DFlash | +workspace | total |
|---|---|---|---|---|---|
| **50%** | 94.7 GB | 11.5 | 2.94 | ~5 | **114.1** ✅ |
| 40% | 109.7 | 11.5 | 2.94 | ~5 | 129.7 ❌ |

**50% is forced, not chosen.** Full reasoning: [`research/MIMO_V26_FLASH_SURVEY_2026-09-23.md`](research/MIMO_V26_FLASH_SURVEY_2026-09-23.md).

## Methodology

Not stock REAP. See [`research/REAP_METHODOLOGY_SOTA_2026-09-23.md`](research/REAP_METHODOLOGY_SOTA_2026-09-23.md).

1. **HOPE, not REAP** (arXiv 2609.18916) — REAP is provably HOPE with the interaction terms
   zeroed. On a model with this exact shape (256 experts, top-8, 48 MoE layers) HOPE beat REAP in
   *every* agentic experiment at 40–50%: mean **+2.8%**, up to **+6.1%**.
2. **Every criterion in the unified family** (arXiv 2606.15716) is recorded, including the two
   that win at 50% when calibration is capability-aligned — `(0,1,1)` and `(0,2,2)`.
3. **Decomposed evaluation only.** At high sparsity, calibration strategies span 2.85 points of
   *averaged* accuracy but **51.9 points of code retention** (arXiv 2606.03328). An averaged
   number here is not a weak signal, it is a misleading one.
4. **Searched per-layer budgets**, not uniform (EvoESAP): +15.2% Math Avg at 50%.
5. **Router KD is required**, not optional (arXiv 2603.02217) — post-compression damage is
   largely router-expert mismatch, worst on fine-grained MoEs, and at 256 experts this is one.

## The omni hole this exists to avoid

Audio and video tokens route through the same 256 experts as text. An expert specialised in
those distributions shows **exactly zero** gated mass against a text+image calibration corpus —
so it is not under-weighted, it is **guaranteed pruned, and pruned first**, silently, while
every text benchmark still passes.

`scripts/mimo_corpus_spec.py` splits the multimodal share into image / audio / video and feeds
each with **real media** through the real processor. Per the corpus's own rule: a transcript
protects the audio path no better than a caption protects the vision path.

## Layout

```
scripts/hope_fmatrix.py        HOPE F-matrix accumulator + QP solver
scripts/mimo_saliency.py       patches MiMoV2MoE.moe; all criteria + F in ONE pass
scripts/mimo_corpus_spec.py    omni calibration corpus (licence-verified, permissive only)
scripts/corpus_spec.py         VENDORED text buckets; the GLM repo's copy is authoritative
scripts/pull_mimo.sh           detached 177.8 GB checkpoint pull, with verification
scripts/calib_pass.py          layer-sequential sweep; one layer resident, resumable
scripts/chunk_builder.py       towers + embedding table -> bucketed chunks of inputs_embeds
scripts/media_loaders.py       audio (log-mel -> RVQ -> patch encoder) and video loaders
scripts/mimo_shards.py         streaming reader; MXFP4 + FP8-block dequant
scripts/gate_*.py              gates — the first four need no GPU; run them first
```

## Gates

Every gate is CPU-only, runs in seconds and needs no checkpoint, because the pass they guard is
expensive and **HOPE's off-diagonal cannot be recovered from a finished pass**.

```bash
PY=~/glm-5.3-reap/.venv/bin/python     # any env with torch + numpy
$PY scripts/gate_hope_fmatrix.py
$PY scripts/gate_mimo_saliency.py
```

They have already earned their keep twice: the projected-gradient solver reached the
brute-force optimum on only 27/40 random problems before swap refinement was added, and the
F-matrix silently deflated every entry under a valid-mask until `F[k,k] == sq/cnt` caught it.

## Status

- [x] Checkpoint pulled and verified (177.8 GB, all 65 shards)
- [x] HOPE F-matrix + solver, gated
- [x] Saliency patcher, gated bit-identical against the upstream forward
- [x] Omni corpus split specified
- [x] Calibration-pass driver, GPU-smoked at S=4096 (~6% instrumentation overhead)
- [x] Chunk builder: embedding table + towers, with the text-only-media-bucket guard
- [x] Audio and video loaders, gated end-to-end on the real towers and the shipped RVQ tokenizer
- [x] Ballast share decided: **0.15 → 0.20**, out of science (0.10→0.07) and finance (0.06→0.04)
- [ ] Calibration pass → HOPE QP → per-layer budget search → Router KD
- [ ] CUDA server (48 layers, 9 full + 39 SWA, GQA with per-type KV head counts, no shared expert)

Base model: `XiaomiMiMo/MiMo-V2.6-Flash-RL` (MIT). Method: REAP (arXiv 2510.13999), HOPE
(arXiv 2609.18916).
