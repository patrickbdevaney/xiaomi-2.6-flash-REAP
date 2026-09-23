# MiMo-V2.6-Flash-RL on Thor — feasibility survey, 2026-09-23

Model released **2026-09-22**, one day old at time of writing. Everything below marked MEASURED
comes from `config.json` and range-read safetensors headers, not the card
([[thor-base-model-shortlist]]: never trust a model card for parameter counts).

---

## 1. Verdict

**Viable, and the REAP is worth more here than it was for GLM — but for the opposite reason,
and the decode ceiling is roughly a third of what the pasted estimate claims.**

- REAP-50 is the **only** level that serves the full 1M context on Thor. Not a preference — an
  arithmetic constraint. REAP-40 misses by ~0.6 GB at 1M and fits at ~256K.
- The **quant lever is already spent.** Experts ship as packed MXFP4 (`U8`), 92.9% of the
  checkpoint. This is the exact criterion that got DeepSeek-V4.1-Flash rejected on 2026-09-10.
  It does **not** reject MiMo, because unlike V4.1-Flash there is no 203 GB Engram table and the
  required prune lands at 50%, not 62.9%. But it means REAP is the only lever we have, so the
  quality of the REAP is the whole ballgame.
- **Nothing published fits this box.** ggml-org's own Q2_K GGUF is 126 GB — over the envelope
  before a single KV byte. The only REAP50 in existence is MLX (Apple), text-only.
- The realistic decode number is **~5 tok/s base AR, ~15 tok/s with the shipped DFlash drafter**,
  not the 34-45 tok/s in the pasted analysis. See §4.

---

## 2. MEASURED architecture

| | value | source |
|---|---|---|
| Total / active | 309B / 15B | card; active confirmed by arithmetic below |
| Layers | 48 (47 MoE + layer 0 dense) | `moe_layer_freq` |
| Attention | **9 full + 39 SWA**, window 128 | `hybrid_layer_pattern` zeros at 0,5,11,17,23,29,35,41,47 |
| GA heads | 64 Q, **4 KV**, head_dim 192, v_head_dim 128 | config |
| SWA heads | 64 Q, **8 KV**, head_dim 192, v_head_dim 128 | config (differs from GA!) |
| Experts | **256 routed, top-8, NO shared expert**, inter 2048 | config |
| Router | `sigmoid` + `noaux_tc`, n_group 1 | config |
| MTP | `num_nextn_predict_layers: 3` (in `model_mtp.safetensors`) | config + index |
| DFlash | separate 5-layer SWA drafter, `block_size 8`, `is_causal: false`, taps layers [0,11,23,35,47] | `dflash/config.json` |
| Context | 1,048,576 | config |
| Vocab | 152,576, untied | config |
| Licence | **MIT** | HF API |
| Repo size | **177.8 GB** (index total_size 172.9 GB) | HF API `usedStorage` |

### Quantization — the decisive field
```
"quantization_config": { "quant_method": "fp8", "store_dtype": "mxfp4",
                         "mxfp4_block_size": 32, "weight_block_size": [128,128],
                         "ignored_layers": [ ...49 entries, ALL o_proj... ] }
```

Byte split from range-reading 64 of 65 shard headers (170.4 of 172.9 GB covered):

| component | GB | share | dtype |
|---|---|---|---|
| routed experts | **158.4** (≈160 full) | **92.9%** | `U8` = packed MXFP4 |
| attention | 6.09 | 3.6% | BF16 + F8_E4M3 |
| embed + lm_head | 2.50 | 1.5% | BF16 |
| dense FFN (layer 0) + gates | 1.39 | 0.8% | BF16 + F8_E4M3 |
| MTP heads | 1.19 | 0.7% | BF16 + F8_E4M3 |
| audio tower | 0.47 | 0.3% | BF16 |
| vision tower (681M) | ~1.4 | — | in the one unfetched shard |

Per expert = 13.37 MB, which back-solves to 3 x 4096 x 2048 x 0.5 B + block scales. **Confirms FP4
on disk.** 72,192 expert tensors = 47 x 256 x 6 exactly.

---

## 3. Fit math

Thor usable: **117 GiB = 125.6 GB** (measured, [[thor-unified-memory-oom]]).

KV per token, FP8 — REAP never touches this, it prunes FFN experts only:
- 9 GA layers x 4 KV heads x (192 K + 128 V) x 1 B = **11,520 B/token**
- 39 SWA layers, window-capped: 39 x 8 x 320 x 128 = **12.8 MB total, constant**

At 1M tokens: **11.5 GB**. *(The pasted analysis got this right, including the 39/9 split.)*

Non-expert weights ≈ **14.3 GB** (reconciled against the artifact below).

| REAP | experts kept | weights | +KV@1M | +workspace | total | vs 125.6 GB |
|---|---|---|---|---|---|---|
| **REAP-50** | 128/256 | **94.7 GB** | 11.5 | ~5 | **111.2** | ✅ 14.4 GB spare |
| REAP-40 | 154/256 | 109.7 GB | 11.5 | ~5 | 126.2 | ❌ over by 0.6 |
| REAP-37.5 | 160/256 | 113.4 GB | 11.5 | ~5 | 129.9 | ❌ over by 4.3 |

REAP-40 fits at **256K** (KV 3.0 GB → 117.7 GB total, 8 GB spare). REAP-37.5 fits nowhere useful.

**The 94.7 GB is not modelled — it is measured.** `tacodevs/MiMo-V2.6-Flash-RL-MLX-REAP50-mxfp4-MTP`
exists and its `usedStorage` is 94.7 GB. My header-derived prediction was 92.5-94.7 GB. A
comparison whose sign I already knew (CLAUDE.md §6), and it landed.

### Correction to the pasted estimate
Its expert-footprint column (80 / 95 / 99 GB) is **numerically right but derived wrongly** — it
assumes an FP8 → NVFP4 conversion that halves the experts. There is no such conversion available;
they are already FP4. Same answer, non-existent lever. Two consequences it then gets wrong:
1. It omits the ~14 GB of non-expert weights from the totals ("REAP-50: ~97 GB, ~23 GB to spare"
   → actually 111.2 GB, 14.4 GB to spare).
2. It implies REAP-40/37.5 are merely "tight". At 1M they do not fit at all.

---

## 4. Decode — the estimate is ~3x optimistic

Bytes read per token (B_tok), the binding term on unified memory:

| term | GB/token | note |
|---|---|---|
| routed experts (top-8 of 256) | 4.73 | **REAP does not change this** — top-8 is unchanged |
| attention | 6.09 | **45% of B_tok**; o_proj is BF16 (in `ignored_layers`) |
| lm_head | 1.25 | |
| dense FFN + gates | 1.40 | |
| **total** | **13.5** | at short context |
| KV read @1M | +11.5 | doubles B_tok at full context |

The pasted analysis divides by Thor's **rated 273 GB/s**. Our own servers measure **57-66 GB/s
effective** on exactly this access pattern (GLM: 11.86 tok/s at B_tok 4.8 GB = 57 GB/s;
DSV4-0731: 14.61 tok/s ≈ 66 GB/s). Rated bandwidth has never been achievable here — that is the
whole subject of [[glm53-decode-gap-profile]].

| context | B_tok | base AR @ 60-80 GB/s | + DFlash 3x (vendor claim) |
|---|---|---|---|
| short (32K) | 13.9 | **4.4 - 5.9 tok/s** | ~13 - 18 tok/s |
| 256K | 16.5 | 3.6 - 4.8 | ~11 - 14 |
| 1M | 25.0 | 2.4 - 3.2 | ~7 - 10 |

**Do not stack the DFlash 3x with a DSpark-style +25%** the way the pasted text does — the +25%
we measured was a tuned head *replacing* a stock head, not composing with it.

### The one big fixable term
`o_proj` is excluded from quantization on all 49 layers, so 3.2 GB of every token's read is BF16
attention output projection. Our NVFP4 dense overlay ([[glm53-nvfp4-dense-overlay]]) applies
directly: 3.2 → 0.8 GB, **B_tok -18%, ~+21% decode**, for a 2.4 GB overlay file. This is the
single highest-value port of existing work and it is independent of the REAP.

---

## 5. What already exists (checked 2026-09-23, model is 1 day old)

| artifact | size | verdict |
|---|---|---|
| `ggml-org/MiMo-V2.6-Flash-RL-GGUF` Q2_K | **126 GB** | official, **over the envelope**; mmproj for vision+audio; `--mtp` works |
| same, MXFP4 | 167 GB | no |
| `tacodevs/...-MLX-REAP50-mxfp4-MTP` | 94.7 GB | **Apple MLX, text-only** — proves the fit, takes the Mac niche |
| `ProCreations/...-NVFP4` | 395 GB | bf16 upcast then transcode; bloated, useless |
| `dymoo/mimo-halo-lab` | — | AMD Strix Halo; REAP "inactive P2", **nothing public, no numbers** |

**The gap is exact: no CUDA-servable, omni-modal, REAP'd MiMo exists.** Also note ggml-org shipped
the GGUF themselves within a day — the GGUF niche you had with GLM-5.3-Flash is *already filled
here by upstream*, and by five others. A GGUF from us is only differentiated if it is **of our
REAP**, i.e. a size nobody else can produce.

### Modality reality check for GGUF
llama.cpp's `mtmd` stack has **image and audio**; **video input is still on the roadmap** (FOSDEM
2026 talk lists it as future work). So a GGUF of this model ships text+image+audio and drops
video. A CUDA server is the only path that carries all four, which is the actual argument for
building one — not speed.

---

## 6. Capability vs the field

Vendor self-reported unless noted. **Never compare a vendor card against a third-party harness**
— the two blocks below are not commensurable.

MiMo-V2.6-Flash card: DeepSWE v1.1 **67.9**, Terminal-Bench 2.1 **87.6**, Terminal-Bench 4.0
**28.8**, CyberGym **95.1**, Toolathlon-Verified **73.6**, AutomationBench v1.0.6 **52.3**,
MiMo Code Bench 61.2, ProgramBench 26.0.

Third-party (Artificial Analysis v4.3.2): MiMo-V2.6-**Pro** 46.32 (top open-weights, ties
Grok 4.7) > GLM-5.3 45 > Kimi K3 44. **MiMo-V2.6-Flash has no independent AA score yet** — every
"46" in the press is the 1T Pro, not this 309B model. Do not attribute it to Flash.

One directly comparable pair, both on Terminal-Bench 4.0: **MiMo-Flash 28.8 vs GLM-5.3-Flash 33**
(the latter from our 2026-09-10 survey). On the newest terminal-agentic benchmark the model we
already run scores *higher*. Flash's strong numbers are on TB 2.1 (87.6), a much older and more
saturated harness. **This is the single fact most in tension with the "strictly better base"
premise**, and it deserves a real measurement before we spend weeks on a REAP.

---

## 7. Disk — PROPOSAL ONLY, nothing deleted

Current: **42 GB free of 936 GB (96% used)**. The download alone is 178 GB, and a REAP needs
room for the source plus the output.

| candidate | size | case for | case against |
|---|---|---|---|
| `glm-5.3-reap/` GLM source (311 GB) | 311 G | REAP50 is published; pass-2 done | **user paused this deletion pending the head-to-head** |
| `models/DeepSeek-V4-Flash-Vision-Exp-REAP-145B` | 78 G | never served; its REAP plan is ludo-tech's, not ours | it is step 3-4 of the approved criterion experiment |
| `s5-capture/` | 37 G | trace capture, likely consumed | unverified |
| `Qwen3.6-35B-A3B-NVFP4` + `-DFlash` | 25 G | superseded | may be a DFlash reference |
| `models/dspark-head-s3recap-p25-b0.1` | 24 G | a fine-tune checkpoint | is it the promoted one? |
| `models/gemma-4-26B-A4B-*` | 17 G | gemma-era, superseded | gate scripts still reference it |

**Nothing here is deletable without your say-so, and the two biggest items are both already under
an explicit hold.** Cheapest real option: the 78 GB Vision-Exp + 37 GB s5-capture + 25 GB Qwen3.6
= 140 GB, which is enough, and none of it is the paused GLM source.

---

## 8. Recommended sequence

Gate 0 first, because it can kill the whole plan for the price of an afternoon.

0. **Measure before building.** Serve the existing `ggml-org` Q2_K GGUF... except it is 126 GB and
   does not fit. So instead: run our HumanEval/BFCL/GSM8K harness against MiMo-V2.6-Flash through
   a hosted API ($0.14/M in — a few dollars, and *not* cloud training spend) and compare to our
   GLM-5.3-Flash REAP50 numbers on the identical harness. If it does not beat 91.5% HumanEval /
   73% BFCL by a real margin, the Terminal-Bench 4.0 signal in §6 was right and we stop.
1. **Port the o_proj NVFP4 overlay** (§4). Independent of the REAP, ~21% decode, already-proven code.
2. **Free disk** per §7, on your explicit approval only.
3. **REAP-50 with our criterion and our corpus**, with the 2026-03-11 upstream fix: REAP now
   renormalizes top-k router logits to sum to 1 (mean accuracy drop 1.9% vs 2.6% without).
   **Check whether our pipeline predates this** — it is a free quality win if so.
   Our own audit says the gap was corpus imbalance, not criterion ([[reap-criterion-and-corpus-audit]]),
   so weight the calibration corpus toward the agentic/coding/long-context mix we actually serve.
4. **CUDA server**: 48 layers, hybrid SWA/GA, GQA with different KV head counts per layer type,
   sigmoid+noaux_tc routing, no shared expert. The DeepSeek engine is the closer starting point
   than the GLM one (no KDA), but neither has a 9/39 hybrid attention schedule.
5. **DFlash drafter last.** It is block-parallel (`is_causal: false`, block 8) — a different animal
   from both our MTP and DSpark work, and our spec-decode history on Thor is a graveyard
   ([[glm53-spec-decode-moe-wall]], [[glm53-mtp-serving-path]]). Treat the vendor 3x as unverified.

**Do not start at step 3.** Every previous cycle that skipped the capability gate paid for it.
