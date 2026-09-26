# Is our REAP the best available, and what can be built on top of it?

**2026-09-26.** Written against the shipped MiMo-V2.6-Flash-REAP50 and GLM-5.3-Flash-REAP50
artifacts and their own accumulators, plus a literature sweep. Supersedes nothing in
`REAP_METHODOLOGY_SOTA_2026-09-23.md` — it answers a different question: not "what should we
build" but "what did we build, how good is it really, and what can be built on it".

---

## The short answer

**Our selection method is at the published frontier. Our ability to tell whether it worked is
not, and that gap is the real finding.**

We run HOPE, which is state of the art as of September 2026. But both numbers we have used to
judge a mask are rulers that one of the candidate selectors maximises *by construction*, so the
comparison cannot settle anything. On MiMo we shipped on those rulers alone. On GLM pass 3 we
will, for the first time, have a neutral one.

The calibration split is good and the literature independently vindicates it. The one thing it
measurably got wrong — world knowledge — is already fixed in pass 3.

---

## 1. Where we sit against the published frontier

| Lever | Frontier | Us | Gap |
|---|---|---|---|
| Within-layer criterion | HOPE, 2nd-order (arXiv 2609.18916, Amazon, 2026-09-16) | HOPE, both models | none — this *is* the frontier |
| Calibration design | balanced multi-source capability buckets (2606.03328) | 7–9 domain buckets, token-space targeted | none; ours is more granular than the paper's |
| Multimodal coverage | not addressed in any REAP paper | real images/audio/video, licence-verified | ahead of the literature |
| Per-layer budget | evolutionary search (EvoESAP, 2603.06003): +15.2% Math at 50% | uniform | **measured small for us — see §4** |
| Router calibration | "necessary, not optional" (2603.02217) | MiMo: ran, rejected 47/47 layers. GLM pass 2: **never ran** | **real gap** |
| Post-prune recovery | LoRA / self-distillation, recovers ~half the loss | per-layer scalar rescale only | **largest untaken lever** |
| Neutral evaluation | paired teacher-student, per domain | GLM: yes. MiMo: **none** | **the important gap** |

Nothing has superseded HOPE. It was submitted 2026-09-16 and the searches turn up no successor.

---

## 2. The finding that matters: both our rulers are rigged

From `artifacts/masks/comparison.txt` on MiMo's complete accumulators, at `protect_frac=0`:

| selector | worst-domain retention | pᵀFp (HOPE's objective) |
|---|---|---|
| HOPE, any of the four criteria | 0.986687 | **1.606e-02** |
| REAP + (0,1,1) `total` | **0.999869** | 3.903e-02 |
| REAP + (0,2,2) `total` | 0.999861 | 3.161e-02 |
| REAP + (1,2,2) `mean` | 0.745937 | 9.949e-02 |
| REAP + (1,1,1) `mean` — REAP as published | 0.729564 | 1.046e-01 |

Read the first two rows carefully. **HOPE wins by 2.4x on pᵀFp. REAP-(0,1,1) wins on retention.
Each wins on the quantity it minimises or maximises by construction:**

- *Retention* is the fraction of gated output mass the keep-set carries. Ranking experts by total
  gated mass — which is exactly what (0,1,1) does — and keeping the top half is close to a direct
  greedy maximisation of it.
- *pᵀFp* is the quantity HOPE's quadratic program minimises. Of course HOPE wins it.

So the headline we published for MiMo — 99.3–99.8% retention across nine domains — is measured on
a ruler where a selector we *rejected* scores 0.9999. That does not make the shipped mask wrong.
It makes it **unverified**. We chose HOPE because the HOPE paper reports real downstream
benchmarks (rank 1.58 vs REAP's 2.42 at 50%; +2.8% mean, +6.1% max on agentic coding), not
because our own numbers showed it. Our own numbers *cannot* show it.

A third-party summary reading those retention figures back — as happened this week — is not
independent confirmation. It is the same self-confirming number a second time.

Note also row 5: **REAP exactly as published scores 0.7296.** The mean-over-routed-tokens
criterion drops busy generalists in favour of rare high-magnitude specialists and sheds a quarter
of the output mass doing it. Which brings us to the independent corroboration.

---

## 3. The literature independently reproduces our failure mode — and our criterion flip

**arXiv 2607.16721, "Half the Experts, All the Code"** (Qwen3.6-35B-A3B, Gemma-4-26B-A4B, 50%):

> Coding perplexity +2.1% from base. General chat perplexity **+131%**.
> Degradation "lands almost entirely on abilities outside coding."

That is our GLM pass-2 result in another lab's numbers. Ours: ballast dNLL 0.989 / top-1 0.580,
against code 0.057 / 0.916, with the mask retaining 48.7% of general saliency mass versus 71–75%
elsewhere. Two independent teams, two model families, same shape of damage. **Domain-opinionated
pruning does not gently de-emphasise the domains you left out; it removes them.**

The same paper, on criterion choice:

> Gemma-4: routing mass matched base (0.890 HumanEval+). **REAP collapsed to 0.762, −16.5 points.**
> "The criterion that dominates Qwen loses by a significant 12.8 points on Gemma."

Routing mass ≈ our (0,1,1) `total`. REAP ≈ our (1,1,1) `mean`. Their 16.5-point collapse and our
0.9999-vs-0.7296 split are the same phenomenon, found independently. Two consequences:

1. **No REAP recipe transfers across model families without per-model validation.** This is why
   MiMo's `protect_frac=0.08` is not being copied to GLM, and why it should not be.
2. Our 2026-09-23 survey predicted (0,1,1) would be the task-specific winner and flagged that we
   had closed the question on mask-overlap evidence alone. That prediction is now confirmed twice
   over — once by our own table, once by an external benchmark.

Also, on evaluation method:

> Rank inversion: a **randomly** pruned model showed better code perplexity (4.82) than base while
> losing 57.9 points on pass@1. A model with perfect 8/8 smoke tests scored 0.091 pass@1.

Perplexity is not a valid readout for a pruned MoE. Our paired teacher-student dNLL / top-1 / flip
rate is the right instrument, and it is the one MiMo does not have.

---

## 4. What we measured ourselves, that closes two open questions

**Non-uniform per-layer budgets are worth ~1 point here, not 15.** `artifacts/masks/layer_budget.json`,
MiMo, 47 layers × 256 experts, searched against uniform:

```
uniform_worst   0.73315
searched_worst  0.74460
gain            +0.01145
```

EvoESAP reports +15.2% Math at 50% on ERNIE-4.5-21B (64 experts). We get +1.1 points on a
256-expert model. High granularity appears to eat most of the benefit — at 256 experts every layer
already has enough resolution that equalising across layers buys little. **On GLM it is worth zero
regardless**, because `Glm5NextTextExperts.__init__` reads a single scalar `num_local_experts` and
applies it to every layer; a ragged checkpoint is unloadable by vLLM and the GGUF converters. The
lever is closed on both models for different reasons. Good: that is one fewer thing to build.

**Our router KD is a no-op, and we should stop reporting it as a stage that did something.**
`artifacts/masks/router_kd_state.json`, MiMo:

```
layers 47; kept_teacher (KD update REJECTED) 47; layers improved 0
median improvement 0.000000; max 0.000000
```

The guard rejected its own update on every single layer and kept the teacher router. The likely
cause is visible in the same file: **2,048 tokens per layer.** arXiv 2603.02217 trains the router
for ~2 GPU-hours on Qwen3-30B. We gave it a rounding error's worth of data, the update failed to
beat the teacher, and the guard correctly refused it. So the stage is honest — it did not ship a
worse router — but it bought nothing, and the literature says it should be worth real points on
fine-grained MoE. **GLM pass 2 never ran it at all** (`grep -c router_kd logs/pipeline.log` → 0).

This is now the cheapest unclaimed gain we have: more tokens, same code.

---

## 5. The uncomfortable one: pruning versus just quantizing harder

Same paper, at **equal memory budgets**:

> Below 3 bits per weight: the pruned specialist wins.
> At 3 bits and above: full-model quantization is superior.
> "Quantize down to 3 bits before pruning anything."

Run the arithmetic for GLM-5.3-Flash, 321B parameters, Thor's 117 GiB ≈ 125.6 GB usable:

| option | bits/weight | weights | headroom for KV, activations, runtime |
|---|---|---|---|
| full model at 3 bits | 3.00 | 120.4 GB | 5 GB — not viable |
| full model, actually fits | ~2.6 | 104 GB | ~21 GB |
| **our REAP50 + NVFP4** | ~4 on half the experts | **103 GB** | ~22 GB |

To genuinely fit with working room for long context, a full GLM-5.3-Flash must go to about
**2.6 bits/weight — below the crossover.** So the pruned artifact is on the right side of the
line. But the margin is roughly one bit, and **we have never measured this head to head.** It is
the strongest available challenge to the entire strategy and it deserves an honest flag rather
than a footnote.

For MiMo the question does not arise: its experts ship as MXFP4 already, so there is no
quantization headroom left to trade against pruning. REAP is the only lever, which is why that
model was chosen for it.

---

## 6. Is the calibration split good?

**Yes, and the reasoning behind it is the literature's central recommendation.**

arXiv 2606.03328 measured, on dense pruning: balanced multi-source mixing beats the strongest
single source by **8.8 points** and C4 by **18.8**, and — the part that matters at our ratio —
**the margin grows with sparsity: +4.6 at 30% → +18.8 at 60%.** Single sources trade off with
opposite sign: C4 retains 60.7% General but 0.4% Code; MetaMath 52.2% Math but 46.2% General.

The stated rationale — finance is practical and load-bearing, math underpins it, science spans
physics/chemistry/biology/genetics/medicine, code and agentic tool use are the primitives, ballast
holds the world model, vision/audio/video protect the modalities — is a capability-bucketed
multi-source mixture. That is the thing the paper says to do.

Two qualifications, both real:

**The "you can RAG or procedurally instruct your way to the rest" argument holds for skills, not
for priors.** Retrieval supplies text; the model still needs the latent world model to know which
passage is relevant, which is wrong, and what it implies. Ballast is not a domain you can retrieve
around — it is what makes retrieval usable. GLM pass 2 is the worked example: 42% of top-1 tokens
changed on general knowledge, and no amount of context fixes a model that has lost the priors for
reading it.

**A domain absent from calibration is not de-weighted, it is deleted.** An expert serving a
distribution with zero tokens in the corpus shows exactly zero gated mass, so it is not merely
ranked low — it is pruned first, and every benchmark for every other domain still looks fine. This
is why the MiMo corpus carries real audio and real video rather than text descriptions of them,
and it is the single most important property of the split.

So: keep it. The change worth considering is not *which* domains but their weights, and that
should come off measured accumulators, as pass 3's ballast 4.8% → 15% did — never off assertion.

---

## 7. Are these good foundations for downstream fine-tuning?

This is the part the earlier survey did not cover. The honest answer has a clear shape.

**What the literature establishes:**

| finding | source |
|---|---|
| LoRA on non-expert params, experts frozen, 95.2M trainable: recovered **52%** of the HumanEval+ gap, 23% of MBPP+ | 2607.16721 |
| "capacity–knowledge asymmetry" — the recovery module itself bottlenecks; overcomplete-then-merge buys **+8.4 points at 50% pruning** | 2609.06974 (OverRep) |
| General capability recoverable with **4–20%** of the original post-training data under selective sampling | 2502.12594 (PASER) |
| Self-data distillation beats continued pretraining for recovery at equal token budget | recovery literature, 2026 |
| Expert-wise KD retains ~99% of base after pruning, "small data, low GPU hours" | 2606.27866 (FlexMoE) |

**What that means concretely for us:**

*Good news.* A REAP50 model is a genuinely attractive fine-tuning substrate. Half the expert
parameters are gone, so LoRA and even fuller tunes fit the envelope where the unpruned model does
not. MiMo's router was at least nominally re-calibrated and its modalities are intact. And the
strongest recovery result in the literature — 52% of the gap — comes from adapting **non-expert**
parameters with experts frozen, which is exactly the cheap regime.

*The catch, and it is structural.* Fine-tuning teaches a model to use capacity it has. It does not
restore capacity that was deleted. The recovery numbers above are all ~50%, not ~100%, and OverRep
names the reason: the adapter cannot reconstruct knowledge the pruned weights no longer carry. So:

- **Fine-tuning a domain that was in the calibration corpus works normally.** The expert subspace
  is intact; you are adapting, not rebuilding.
- **Fine-tuning a domain that was not** runs into a ceiling that presents as "needs more data" but
  is actually structural. You are teaching into a subspace that was removed.

Mapped onto the specific plans:

| downstream target | verdict |
|---|---|
| game dev, browser dev, agentic traces from your own projects | **Strong fit.** These are code + agentic + tool use — the best-covered buckets in both corpora. This is adaptation inside retained capacity. |
| medicine, genetics, chemistry, physics | **Workable on MiMo, poor on GLM pass 2.** MiMo's science bucket retains 0.9973; GLM pass 2's general/world knowledge sits at 0.487 and medicine leans hard on it. Pass 3 is the prerequisite. |
| anything leaning on broad world knowledge — law, history, general assistant | **Not on GLM pass 2.** 42% top-1 flip on ballast is the ceiling you would be fighting. |

**The recipe that follows** (none of it built yet, all of it cheap): LoRA on attention and shared
pathways with routed experts frozen; distil from the *unpruned teacher* rather than training on
raw domain text, since self-data distillation beats continued pretraining at equal budget; select
the recovery set rather than using all of it, since 4–20% suffices; and keep a 5–20% general
replay mix so domain SFT does not re-open the ballast hole the pruning already cost.

---

## 8. Should we use a weaker base that does not need pruning instead?

Considered, and the answer is model-specific rather than general.

The case for it is real: a model that fits unpruned has no deleted subspace, so it has no
structural fine-tuning ceiling, and §7 says that ceiling is the dominant downstream risk.

The case against, for our envelope:

- Every frontier open-weights model is 3–12x over 117 GiB. "Fits unpruned" means dropping a
  capability tier, not avoiding a tradeoff. The comparison is *pruned frontier model* versus
  *intact smaller model*, and REAP's own results say a high-granularity 50% prune costs ~6.9% mean
  on coding — less than a tier drop typically costs.
- Granularity governs the damage, and both our models are at or above the best granularity in the
  literature (MiMo 256 experts/top-8; GLM 288/top-8, the highest published). We are in the
  favourable regime, which is precisely where 50% is defensible.
- §5's crossover only favours pruning below 3 bits/weight. That is where our envelope puts us for
  GLM, and MiMo has no quantization headroom at all.

**But the honest version is that this is a measurable question we have not measured.** The
decisive experiment is cheap and we already own both halves of it: score an intact candidate base
and our REAP50 on the same held-out set with the same paired harness, then fine-tune both on one
real target domain and compare the ceiling, not the starting point. Until that is run, the choice
rests on the literature's priors, not on our evidence.

---

## 9. What to do, ranked by value per hour

1. **Finish pass 3 and read the neutral ruler.** GLM has a cached teacher (`teacher.pt`, 338,113
   tokens) and a paired harness. This is the first end-to-end number we will have for any mask we
   have ever shipped. Everything below is speculation until it lands.
2. **Give router KD real data.** 2,048 tokens per layer produced 47/47 rejections. The code works
   and the guard is honest; it has simply never been fed. Cheapest unclaimed gain we have.
3. **Build MiMo a neutral evaluation.** It shipped with no teacher-student measurement at all.
   Without one we cannot say whether HOPE or REAP-(0,1,1) was the right call on that model, and
   §2 says the retention number cannot answer it.
4. **Run a recovery LoRA on one model and measure it.** Literature says ~50% of the gap for
   ~95M trainable parameters and experts frozen. If it holds, it changes what a REAP artifact is
   worth as a fine-tuning base.
5. **Settle §5 head to head** — 2.6-bit full model versus REAP50+NVFP4 at matched memory. It is
   the strongest challenge to the strategy and one experiment answers it.
6. Leave per-layer budget search alone. Measured at +1.1 points on MiMo and architecturally
   impossible on GLM.

---

## What we can claim honestly today

- We use the best published selection method, on the best-suited model granularity in the
  literature, with a calibration design the calibration literature explicitly recommends, and with
  multimodal coverage no REAP paper attempts.
- Our masks retain 99.3–99.8% of gated output mass across nine domains on MiMo — **on an in-sample
  ruler that a rejected alternative scores higher on.**
- We have independent, external confirmation that our one measured failure — world knowledge —
  is the characteristic failure of the whole method class, and we have already fixed its cause.
- **We cannot yet claim a state-of-the-art 50% REAP,** because we have never measured one
  end to end. Pass 3 is the first time we will be able to.

That last line is the difference between a good artifact and a demonstrated one, and it is worth
saying out loud before anyone downloads 100 GB on the strength of a retention table.

## Sources

Internal: `artifacts/masks/comparison.txt`, `mask.json`, `layer_budget.json`,
`router_kd_state.json` (MiMo); `artifacts/eval/*.json`, `artifacts/domain_holes.json`,
`scripts/corpus_spec.py` (GLM).

External: [HOPE, arXiv 2609.18916](https://arxiv.org/abs/2609.18916) ·
[REAP, arXiv 2510.13999](https://arxiv.org/abs/2510.13999) ·
[Half the Experts, All the Code, arXiv 2607.16721](https://arxiv.org/abs/2607.16721) ·
[Expert scoring family, arXiv 2606.15716](https://arxiv.org/abs/2606.15716) ·
[Train Overcomplete Deploy Compact, arXiv 2609.06974](https://arxiv.org/abs/2609.06974) ·
[PASER, arXiv 2502.12594](https://arxiv.org/abs/2502.12594) ·
[FlexMoE, arXiv 2606.27866](https://arxiv.org/abs/2606.27866) ·
[LoRA-MoE expert pruning, arXiv 2604.26340](https://arxiv.org/abs/2604.26340)
