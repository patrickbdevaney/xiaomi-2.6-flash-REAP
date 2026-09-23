# Optimal REAP methodology for MiMo-V2.6-Flash — literature sweep, 2026-09-23

Goal: one pass, done right. Target is a long-horizon agentic coding model that keeps math /
science / finance and does not lose general world knowledge, at the 50% prune that Thor forces.

Five changes below. Ranked by expected value. Two of them are things our current pipeline gets
*wrong*, not merely things it lacks.

---

## 1. HOPE instead of REAP — the single biggest upgrade

**arXiv 2609.18916, "Higher-order pruning of experts in mixture-of-experts language models."**

> "We show that REAP (a state-of-the-art first-order pruning method) is a special case of HOPE
> where interaction terms are ignored."

REAP ranks each expert independently. HOPE minimises a provable upper bound on substitution
error that includes *pairwise* terms, so it keeps experts that **cooperate**:

    F_{i,j} = (1/|X_{i,j}|) · Σ_{x : i,j ∈ T(x)}  g_i(x) g_j(x) ‖f_i(x)‖₂ ‖f_j(x)‖₂
    P* = argmin_p  pᵀ F p     s.t.  p ∈ {0,1}^E,  Σ p_k = |P|

solved by continuous relaxation and rounding to the top |P|.

**Why it matters here specifically**
- Evaluated on **Qwen3.5-122B-A10B: 256 experts, top-8, 48 MoE layers** — MiMo is 256 experts,
  top-8, 47 MoE layers. The architectural match is about as close as published evidence gets.
- Its advantage *grows with prune rate*, "particularly at 40–50%" — exactly where Thor puts us.
- Calibration sets were Evol-CodeAlpaca and **SWE-Bench-verified trajectories** — our workload.
- At 50%: average rank **1.58 vs REAP's 2.42**; it "beat REAP in all experiments at a high
  pruning rate (40–50%), with a mean gap of **+2.8%** and up to **+6.1%**" on the agentic
  benchmark. Gains over next-best reached **+19.6% LiveCodeBench**, **+11.7% BBH**,
  **+2.0% SWE-bench Pro**. Worst single-condition deficit: −4.5%.

**Cost to us: almost nothing.**
- "HOPE requires a single forward pass over the calibration set" — the same pass we already run.
- The QP adds "1–2 seconds per layer".
- **We already have the diagonal.** F_{k,k} = E[(g_k‖f_k‖)²], and `stream_saliency.py` already
  accumulates `acc["sq"] += s²` where `s = g·‖f‖`. Only the off-diagonal is missing.
- Off-diagonal cost: top-8 routing means 8×8 outer product of the per-token vector (g_j‖f_j‖)
  per layer — we already compute every element of it. Storage for MiMo: 256×256×47 × (sum+count)
  × 8 B ≈ **49 MB**.
- Code released: `github.com/awslabs/hybrid-model-factory`.

**Action: add F-matrix accumulation to the MiMo saliency pass.** It is nearly free, it is
gated by construction (zeroing the off-diagonal must reproduce our REAP mask — the paper reports
exactly that, Figure S4, which gives us a control whose sign we know), and it is the difference
between a generic REAP50 and a better one.

---

## 2. Our criterion shootout dismissed what the literature now calls the winner

**arXiv 2606.15716, "How to Score Experts for One-Shot MoE Expert Pruning."** It unifies every
criterion into one family:

    S_j(b,α,β) = (1/N_j^b) · Σ_t 1[j ∈ ℰ_t] · g_{j,t}^α · ‖f_{j,t}‖₂^β

REAP is **(1,1,1)**. The paper's selection principle:

> "task-agnostic pruning should favor routed-token-averaged, gate-free activation-based criteria,
> whereas **task-specific pruning can benefit from retaining routing-frequency and gate-weight
> information**."

We are firmly in the task-specific case — our corpus is deliberately capability-aligned. At
**50% on Qwen3 with coding-aligned calibration**, the winners are **(0,1,1)** and **(0,2,2)**
at average rank 2.12, while the task-agnostic favourites MAN/MSAN collapse to rank 5.38–6.50.

**(0,1,1) is `sum(g·‖f‖)` — the SUM, not the mean.** That is exactly what
`scripts/criterion_shootout.py` calls `frequency`, where its docstring says:

> "frequency   sum(g*||f||), NOT the mean — the frequency-weighted ranking **REAP exists to
>  avoid**; included as a control that SHOULD look different"

And it *did* look different — it is the only criterion that diverged materially:

| criterion | overlap vs REAP | experts differing / layer |
|---|---|---|
| **frequency = (0,1,1)** | **0.755** | **35** |
| norm_only | 0.894 | 15 |
| gate_only / mix_codemath | 0.923 | 11 |
| quantile / mix_sample | 0.938 | 9 |
| var_aware | 0.963 | 5 |

We read that divergence as *confirmation it was the wrong criterion* and closed the question
("criterion does not matter at this ratio"). But the shootout only ever measured **mask
overlap** — it never materialised or evaluated `frequency`. REAP's own argument is against
frequency-*only* (0,0,0); (0,1,1) is a different object.

**Action: materialise and evaluate (0,1,1) and (0,2,2) alongside HOPE.** The accumulators
already support them at zero marginal compute. This is a cheap re-opening of a question we
closed on insufficient evidence.

---

## 3. Never judge this with an averaged score

**arXiv 2606.03328, "Calibration Data Trade-offs Across Capability Dimensions."** The finding
that should govern our whole eval design:

> "the apparent robustness to calibration choice is an averaging artifact. At 60% sparsity with
> SparseGPT, the strategies span only **2.85 points in averaged commonsense accuracy but 51.9
> points in Code retention**."

Single sources trade off with *opposite sign*: C4 retains "60.7% of General capability but only
0.4% of Code"; MetaMath retains "52.2% of Math but only 46.2% of General." Spearman across 15
sources: calibration perplexity correlates **+0.71 with General but −0.53 with Math** (Wanda),
**+0.55 vs −0.59 with Code** (SparseGPT).

Balanced multi-source mixing beats the strongest single source by **8.8 points** and C4 by
**18.8**, and critically **the margin grows with sparsity: +4.6 at 30% → +18.8 at 60%**.

**This validates our corpus design** — we built a capability-bucketed multi-source mixture before
this paper, and its central recommendation is the thing we already do. Two consequences anyway:
- Our `mask_eval` **by_domain** decomposition is not a nicety, it is the only valid readout. Any
  single averaged number would have hidden a 51.9-point code swing.
- Honest caveat: this is dense, unstructured pruning on LLaMA-3.1. The paper explicitly lists
  "mixture-of-experts models … remain to be studied."

---

## 4. Per-layer budgets should be searched, not uniform

**arXiv 2603.06003, EvoESAP.** Decouples the problem: fix within-layer ranking by REAP/HOPE,
then evolutionary-search the integer per-layer removal vector. Fitness is ESAP,
`E[min(1, p/q)] = 1 − TV(p,q)` under teacher forcing — no autoregressive decoding, which cuts
search from "29.49 hours to 1.64 hours."

At 50% on ERNIE-4.5-21B: **+15.2% Math Avg, MATH-500 +19.6%** over uniform allocation. Search
cost "2×L40S GPUs, 5.0 hours", 20 generations, population 32.

Our own audit already measured uniform allocation costing ~5%. And note the warning: "searched
allocations are non-uniform, but their shapes are not consistent across pruning criteria" —
there is no template to copy, it must be searched *for our criterion*.

**This composes with HOPE**: HOPE gives the within-layer order, EvoESAP gives the per-layer
budget. `mask_eval.py` already computes teacher-vs-masked divergence, which is most of ESAP.

---

## 5. Router calibration is necessary, and ours is currently broken

**arXiv 2603.02217, "Is Retraining-Free Enough? The Necessity of Router Calibration."**
Post-compression degradation "largely stems from … router-expert mismatch when experts are
changed but the router is left untouched." Router KD: token-level KL from teacher, temperature
τ², "gradients are backpropagated and applied *exclusively* to the student router parameters
θ_R, while all expert and backbone parameters remain frozen." ~2 h for Qwen3-30B; the router is
0.04% of parameters.

It "proved particularly effective for **fine-grained MoEs**" — 128 small experts. **MiMo has 256
experts with intermediate 2048, finer-grained still.** It does not help "when the model suffers
from catastrophic collapse."

**We have `scripts/router_kd.py`, and its STAGE 3 smoke FAILED in the last chain run.** The
literature reclassifies this from an optional experiment to a required stage. Fix it.

---

## Corpus: what to change, and what not to

**Keep.** Multi-source capability buckets (§3 says this is the central principle). MAX_TOKENS
16,384 — REAP's own paper reports that calibration at 16,384 "retains performance at context
lengths far exceeding that value", so long context does not need long calibration. The
medium-weighted difficulty policy is independently supported: for long-context training,
"balanced data outperforms target-length-focused data."

**Change 1 — the omni hole (forced, no measurement needed).** `MM_SOURCES` already states the
principle: "rows must carry real images … Text descriptions of images route like text and
protect nothing." MiMo routes audio- and video-derived tokens through the same 256 experts. An
expert specialised in those distributions shows *exactly zero* gated mass against a corpus with
no audio and no video, so it is not under-weighted — it is **guaranteed pruned, and pruned
first**. Calibrating on text+images and then pruning 50% would silently destroy the one
capability that justified choosing this model, and every text benchmark would still look fine.
`scripts/mimo_corpus_spec.py` splits `multimodal` 0.12 → image 0.06 / audio 0.03 / video 0.03,
fed with real media from licence-verified permissive sources.

**Change 2 — general retention (measure, don't guess).** Our pass-2 data: the shipped mask
retains only **48.7% of general saliency mass vs 71–75% everywhere else**, and general dNLL is
**1.0027 against code 0.057**. §3 says the cost of imbalance *grows with sparsity* and we are at
50%. Pass 3 already moved ballast 4.8% → 15%.

The tempting fix is offline re-weighting of the bucketed accumulators — but **we already proved
that does not work**: "a bucket-balanced RE-MASK of the pass-2 accumulators recovers just
+0.028 … re-masking cannot create information the corpus never collected." So the tokens must
be in the corpus. Candidate: ballast 0.15 → ~0.20, taken from science (0.10) and finance (0.06),
which are the buckets furthest from the primary use case. **Do not decide this by assertion** —
pass 3 derived its numbers from measured pass-2 accumulators, and the same discipline applies.

---

## Ordering, with costs

| # | step | cost | why this order |
|---|---|---|---|
| 0 | add F-matrix + (0,1,1)/(0,2,2) accumulators to the saliency pass | ~50 MB, no extra forward | must be in the pass; retrofitting costs a whole pass |
| 1 | one calibration pass on the full omni corpus | the long pole | yields REAP, HOPE, and every criterion at once |
| 2 | HOPE QP + criterion shootout, evaluated **by domain** | 1–2 s/layer + eval | never averaged (§3) |
| 3 | EvoESAP per-layer budget search | ~5 GPU-h equivalent | needs step 2's within-layer order |
| 4 | Router KD on the chosen mask | ~2 h | required, not optional (§5) |
| 5 | optional LoRA recovery on attention projections | 1 epoch, AdamW 2e-4 | MAESTRO; weakest evidence for our case |

**Step 0 is the one with a deadline.** Everything else can be redone offline from the
accumulators; the F-matrix cannot be recovered without re-running the forward pass.

---

## Considered and not recommended

- **MAESTRO (2607.08601)** LoRA recovery — adapters on attention projections, experts and router
  frozen, one epoch AdamW peak LR 2e-4. Sound, but evaluated at **25%** compression on Safety /
  Bias / Ethics domains. Weak fit to 50% agentic coding. Keep as step 5, not a premise.
- **GPrune-LLM (2603.13418)** — +4.88 points and −76.8% perplexity at 50%, but dense FFN-neuron
  and attention-head pruning only; "No discussion of Mixture-of-Experts pruning is provided."
- **MoNE (2507.00390)** — replaces pruned experts with lightweight novices. REAP's own paper is
  the argument against merging/replacement at this scale.
- **SlimQwen (2605.08738)** — progressive prune+distil with MTP distillation. Interesting for the
  drafter later; it is a pre-training-scale method and assumes budget we do not have.
- **Upstream REAP logit-renorm fix (2026-03-11)** — verified a **no-op for us**: we read the gate
  from the model's router, which already renormalises under `norm_topk_prob` (true for both GLM
  and MiMo).
