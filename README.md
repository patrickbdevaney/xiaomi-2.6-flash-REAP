# xiaomi-2.6-flash-REAP

Pipeline that prunes `XiaomiMiMo/MiMo-V2.6-Flash-RL` (309B parameters, 15B active, omnimodal: text, image, audio
and video input) from 256 to 128 routed experts per layer. The target is one 128 GB NVIDIA Jetson AGX Thor. Expert
selection uses HOPE, an interaction-aware objective, over a calibration corpus that carries real image, audio and
video. Without real media in calibration, the experts that serve those modalities are pruned first and nothing
reports it.

## Status (2026-10-26)

Published but not evaluated. The pruned checkpoint and its GGUF builds are public. None has been served, benchmarked
or compared end to end against the unpruned model. Treat them as untested research artifacts.

| Artifact | Format | State |
|---|---|---|
| [`MiMo-V2.6-Flash-REAP50`](https://huggingface.co/patrickbdevaney/MiMo-V2.6-Flash-REAP50) | Safetensors, MXFP4 experts (bit-exact subset), BF16 attention, ~86 GiB | Public; never run for generation |
| [`MiMo-V2.6-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF) | MXFP4_MOE 86.06 GiB, Q2_K 61.64 GiB, vision/audio `mmproj`, MTP draft | Public; never loaded by us |
| End-to-end quality evaluation | — | Not run |
| Serving measurement (tok/s, long-context retrieval) | — | Not run. The 1M-context fit below is sizing arithmetic |
| CUDA server | — | Not started |

The selection-time retention figures below are a proxy, not quality. On the companion GLM-5.3 project, the same
proxy and its end-to-end measurement diverged: predicted per-domain retention of 0.49–0.75 against measured top-1 of
0.57–0.92 ([glm-5.3-reap](https://github.com/patrickbdevaney/glm-5.3-reap)).

## Why this model, and why 50%

`quantization_config.store_dtype = "mxfp4"`: the routed experts already ship as packed FP4. That is 158 GB of `U8`,
92.9% of the 172.9 GB index. Re-quantizing below 4 bits would be a second lossy quantization, so expert pruning is
the only remaining compression lever, and its quality is the whole product.

Sizing arithmetic against 125.6 GB usable. These are estimates, not measured footprints:

| Experts kept | Weights | KV at 1M tokens | DFlash drafter | Workspace | Total | Fits |
|---|---|---|---|---|---|---|
| **128 of 256 (50%)** | 94.7 GB | 11.5 GB | 2.94 GB | ~5 GB | **114.1 GB** | yes |
| 154 of 256 (40%) | 109.7 GB | 11.5 GB | 2.94 GB | ~5 GB | 129.7 GB | no |

- **KV cache.** It is 11,520 bytes per token, because only the 9 full-attention layers keep a long cache; the 39
  sliding-window layers are capped at 128 tokens. Pruning does not change it.
- **The 50% ratio is forced by memory, not chosen for quality.** See
  [`research/MIMO_V26_FLASH_SURVEY_2026-09-23.md`](research/MIMO_V26_FLASH_SURVEY_2026-09-23.md).
- **The DFlash drafter is not shipped.** The sizing reserves room for it, but the source's drafter was trained
  against the unpruned expert set and is not valid for this checkpoint.

As of the 2026-09-23 survey, no CUDA-servable omnimodal 50% build of this model existed:
- ggml-org's Q2_K GGUF is 126 GB, over the envelope before any KV cache.
- The only other REAP-50 was Apple MLX and text-only.

## What was measured

**Selection** (published checkpoint):

| Setting | Value |
|---|---|
| Objective | HOPE, minimising pᵀFp (0.01624 at the solution) |
| Saliency criterion | `reap_1_1_1` |
| Prune ratio | 0.50, uniform across layers |
| Per-domain protection | Top 8% of every domain excluded from the prune set |
| Routers | Refit by output matching against the teacher did not beat the teacher; all 47 kept at teacher weights |

**Retained gated output mass by calibration domain** (selection-time proxy):

| Domain | Audio | Image | Video | Science | Math | Finance | General ("ballast") | Code | Agentic |
|---|---|---|---|---|---|---|---|---|---|
| Retention | 0.9935 | 0.9937 | 0.9938 | 0.9973 | 0.9974 | 0.9974 | 0.9976 | 0.9983 | 0.9983 |

Selection is ranked by the worst domain, never the mean. Even so, these numbers say how much routed mass the kept
experts carry on calibration data. They do not say how far the model's outputs moved.

## Method

1. **HOPE instead of scalar REAP** (arXiv 2609.18916). Scalar REAP is HOPE with the expert-interaction terms zeroed.
   The literature reports HOPE winning every agentic experiment at 40–50% on a model of this shape (256 experts,
   top-8): mean +2.8%, up to +6.1%. That gain has not been measured on this checkpoint.
2. **Every criterion in the unified family** (arXiv 2606.15716) is recorded in the same pass. That includes `(0,1,1)`
   and `(0,2,2)`, reported to win at 50% under capability-aligned calibration.
3. **Decomposed evaluation by design.** At high sparsity, calibration strategies span 2.85 points of averaged
   accuracy but 51.9 points of code retention (arXiv 2606.03328), so selection scores each domain separately.
4. **Omnimodal calibration with real media.**
   [`scripts/mimo_corpus_spec.py`](scripts/mimo_corpus_spec.py) splits the multimodal share into image, audio and
   video, and feeds each through the model's real processors. A transcript protects the audio path no better than a
   caption protects the vision path. An expert that serves only audio or video shows exactly zero gated mass on a
   text-and-image corpus, so it would be pruned first.
5. **Router refit, a negative result.** Post-compression damage is largely router-expert mismatch (arXiv
   2603.02217), so a router-only refit was required. It did not beat the untouched routers, and the teacher routers
   shipped.

Planned, not applied: per-layer budget search (EvoESAP reports +15.2% math average at 50%). The published checkpoint
uses a uniform ratio.

## Engineering

- **Layer-sequential calibration.** One layer is resident at a time; the pass is resumable and instruments all
  criteria and the HOPE F-matrix together. Measured overhead is about 6% at 4,096-token sequences
  ([`scripts/calib_pass.py`](scripts/calib_pass.py)).
- **Gates before GPU time.** The off-diagonal of HOPE's F-matrix cannot be recovered from a finished pass, so every
  accumulator is gated first, on CPU, in seconds, without the checkpoint. The gates caught two defects:
  - the QP solver reached the brute-force optimum on only 27 of 40 random problems before swap refinement was added;
  - the F-matrix silently deflated every entry under a valid-mask, caught by asserting `F[k,k] == sq/cnt`.
- **The saliency patcher is bit-identical to the upstream forward** (`gate_mimo_saliency.py`).
- **No corpus in git.** `.gitignore` excludes artifacts by directory and by extension. The original `artifacts/*.pt`
  rule did not recurse into `artifacts/chunks/` and committed corpus chunks before anyone noticed.

```bash
PY=python3                                  # any environment with torch and numpy
$PY scripts/gate_hope_fmatrix.py
$PY scripts/gate_mimo_saliency.py
```

## Getting started

There is no tested way to run the published weights yet. The
[GGUF model card](https://huggingface.co/patrickbdevaney/MiMo-V2.6-Flash-REAP50-GGUF) lists llama.cpp commands; it
states they are illustrative and untested. The safetensors checkpoint ships the source's `modeling_mimo_v2.py` for
`transformers` with `trust_remote_code`.

To rerun the pipeline: [`scripts/pull_mimo.sh`](scripts/pull_mimo.sh) pulls and verifies the 177.8 GB source. Then
build the corpus ([`scripts/build_corpus.py`](scripts/build_corpus.py),
[`scripts/chunk_builder.py`](scripts/chunk_builder.py)) and run [`scripts/calib_pass.py`](scripts/calib_pass.py).
Long stages run under `systemd --user` ([`systemd/`](systemd/)).

## Limitations

- **No end-to-end evidence.** Quality, speed and long-context behaviour are all unmeasured.
- **Calibration length.** Calibration sequences were at most 4,095 tokens. Behaviour toward the 1M-token context is
  untested.
- **Video in GGUF.** llama.cpp's multimodal path (`mtmd`) handles image and audio; video is on its roadmap, so the
  GGUF builds cannot use video input.
- **Domain coverage.** Calibration covered English and Chinese text, code, math, science, finance, agentic traces,
  and image, audio and video. Other domains and languages were not measured.
- **Irreversible.** Pruned experts do not come back; this is a lossy transform of the base model.

## Roadmap

Ordered by value:
1. **Serve the MXFP4_MOE GGUF on Thor.** Record decode tok/s and needle-in-a-haystack at long context with an FP8 KV
   cache.
2. **Paired teacher-forced evaluation** against the unpruned model, per domain.
3. **Per-layer budget search.**
4. **CUDA server.** 48 layers, 9 full-attention and 39 sliding-window, per-type KV head counts, no shared expert.

## Repository map

| Path | Contents |
|---|---|
| [`scripts/`](scripts/) | Corpus spec and builders, media loaders, layer-sequential pass, HOPE F-matrix and QP solver, saliency patcher, gates |
| [`research/`](research/) | Model survey, REAP methodology review, fine-tune viability, OOM containment on unified memory |
| [`wiki/`](wiki/) | Running log and long-context, capability-coverage and router-KD notes |
| [`conf/`](conf/), [`systemd/`](systemd/) | Pipeline configuration; user services |

Not included: weights, calibration corpora, pass outputs. These are published on Hugging Face or kept local.

## License

- **Pipeline code:** MIT ([`LICENSE`](LICENSE)).
- **Weights:** MIT, inherited from
  [`XiaomiMiMo/MiMo-V2.6-Flash-RL`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL). This work is not
  affiliated with or endorsed by Xiaomi.
- **Calibration sources:** licence-verified, permissive only.

## Citation

```bibtex
@misc{devaney2026mimoreap,
  title  = {xiaomi-2.6-flash-REAP: HOPE expert pruning of MiMo-V2.6-Flash for a single Jetson AGX Thor},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/xiaomi-2.6-flash-REAP}
}
```

Builds on REAP (Lasby et al., 2025, arXiv:2510.13999), HOPE (arXiv:2609.18916) and Xiaomi's MiMo-V2.6-Flash.
