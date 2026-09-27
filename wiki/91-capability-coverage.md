# 91 — Capability coverage: what the calibration set actually protects

*Opened 2026-09-26. Question asked directly: "with this third reap and mimo not only are we
preserving the multimodalities but also long context and all of the other needed model
capabilities that i may have not thought of correct".*

Short answer for MiMo: **the named buckets are covered, long context is in better shape here
than on GLM but still unverified, and non-English is untested.**

---

## 1. What the nine buckets cover

`agentic, code, math, ballast(general), science, finance, image, audio, video`. MEASURED from
`artifacts/chunks/manifest.json` (27 chunks, 48.99 M tokens):

| bucket | tokens | | bucket | tokens |
|---|---|---|---|---|
| agentic | 11,501,033 | | image | 3,001,238 |
| ballast | 10,003,073 | | science | 2,980,200 |
| code | 9,503,748 | | audio | 1,502,892 |
| math | 7,502,972 | | video | 1,502,610 |
| finance | 1,503,455 | | | |

All three media modalities carry real routed tokens — 6.0 M between them, 12.3 % of the corpus.
Multimodal preservation is **structural and measured**, not incidental, and `protect_frac`
reserves the top-k of every domain separately so no modality can be zeroed by a louder bucket.

---

## 2. Long context: better than GLM, still unverified  `[EST]`

MiMo calibrated at **`SEQ_LEN` default 4096** (`scripts/run_reap.sh:13`), twice GLM's 2,048.
The corpus spec allows up to `MAX_TOKENS = 16_384` with `BAND_HARD = (4_000, 16_384)`, so the
hard band is *partially* represented rather than erased — a 4,096-token window admits the bottom
of the hard band intact.

The structural argument also runs in MiMo's favour, and it is the reason
`scripts/exp_seqlen_saliency.py` was written:

> 39 of 48 layers are SWA with a window of 128, so their attention context is capped at 128
> tokens NO MATTER WHAT S IS — structurally, S cannot change what those layers see beyond
> position 128. Only the 9 full-attention layers accumulate long range.

If that holds, 39/48 layers are S-invariant by construction and only the 9 GA layers are exposed.
**But `exp_seqlen_saliency.py` never ran** — the 166 GB source was deleted before it executed, so
there is no result. The prediction is sound and untested. `[OPEN]`

Re-running it now costs a re-download; see [98-router-kd-global.md](98-router-kd-global.md) §4 for
the same blocker. **Cheaper path: run the equivalent test on GLM**, where the source is staged,
and treat MiMo's SWA majority as the reason to expect a *smaller* effect there rather than a
reason to assume none.

> Cross-reference: GLM is the worse case and the argument **inverts** on it. GLM is ~3/4 KDA
> linear-attention layers carrying recurrent state over the full context, with no window capping
> it, and it truncates at 2,048 — discarding 73.5 % of its collected calibration tokens. See the
> GLM repo's `wiki/91-capability-coverage.md`. `[EST]`

---

## 3. Non-English is untested  `[OPEN]`

The GLM corpus, built by the sibling of this pipeline, MEASURED at **0.3 % CJK-bearing** across
all text buckets. The source lists and builder logic are closely related, so MiMo's corpus is
very unlikely to be materially different — but it has **not** been measured here, and the model
source is deleted, so decoding its packed chunks needs the tokenizer restored first.

Why it matters: MiMo-V2.6-Flash is a Xiaomi model with substantial Chinese capability, MoE
experts are known to specialise by language, and REAP prunes on routing mass. An
English-dominated calibration set gives a language-specialised expert almost no mass. There is
also **no eval bucket for it**, so the paired teacher/student dNLL/flip gate would score a
Chinese-degraded model as clean.

Recorded as a known unknown on the shipped artifact. The cheap honest fix is a Chinese held-out
slice in the *eval*, which costs no calibration budget.

---

## 4. Axes checked and judged covered  `[EXT]`

- **Instruction following / chat format** — carried by `agentic` (11.5 M tokens, the largest
  bucket) and `ballast`; samples are rendered through the chat template.
- **Tool calling / structured output** — inside `agentic` by construction.
- **Reasoning depth / long CoT** — partly covered; it is the long-context question wearing a
  different hat, and a 4,096 window truncates the deepest traces. `[OPEN]`
- **Safety / refusal behaviour** — no bucket, no plan to add one. Unmeasured, recorded so it is
  not a surprise.

---

## 5. Standing answer

> Multimodal (image, audio, video): **yes — 6.0 M routed tokens, protected and measured.**
> Long context: **calibrated at 4,096 with 39/48 layers structurally S-invariant, so exposure is
> low — but the test that would confirm it was written and never run.**
> Non-English: **untested here; 0.3 % on the sibling corpus.**
> Everything else named: covered.

Nothing here says the shipped mask is bad. It says two claims we have not earned are easy to
mistake for claims we have. `[EST]`
