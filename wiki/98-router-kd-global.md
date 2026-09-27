# 98 — Router KD: the local objective's negative result, and the global scope

*Opened 2026-09-26. §1–2 are measured. §3 onward is scoping — nothing there has run.*

## 1. The layer-local objective is out of work  `[EST]`

MEASURED 2026-09-26, `scripts/gate_router_kd_stratified.py` (7/7 PASS), on a fixture carrying
the checkpoint's **measured** router scale (`W_RMS = 0.032202`) and bias spread
(`B_MEAN, B_STD = 1.3831, 0.0588`). Judged on a held-out population, 300 steps:

| step | held-out population |
|---|---|
| **lr 1e-3 (as shipped)** | **−299.5 %** |
| lr 1e-4 | −62 % |
| lr 1e-5 | −7.5 % |
| lr 3e-6 | −2.2 % |

**No sampling rule × budget × step beats the teacher's sliced router.** Damage falls
monotonically as the step approaches *change nothing*; the best configuration is whichever does
least. The guard that kept the teacher on 47/47 layers was correct.

Adam is scale-free — ~`lr` per entry per step regardless of gradient magnitude. 300 steps at
1e-3 walks each entry up to 0.3 against a router weight of RMS 0.032, ten times the whole weight.

### The token budget was never the cause — a claim of mine, falsified

MEASURED against MiMo's own accumulators and its own mask:

```
routed slots hitting a pruned expert : 1.13 %
tokens with >=1 pruned slot          : 8.71 %
layers 1 and 2                       : 0.0000 %
layer 7 (the outlier)                : 18.18 % of slots, 79.9 % of tokens
```

REAP keeps the experts the router selects most, so on a well-chosen mask the student already
equals the teacher on ~91 % of tokens. Layer 1's baseline loss of 7.9e-15 was **near-exactness,
not starvation**. The earlier audit claim that KD failed because "we gave it a rounding error's
worth of data", and that this was "the cheapest unclaimed gain we have", is
**~~WITHDRAWN 2026-09-26~~**. `[EST]`

### Two fixture bugs that would each have produced a confident wrong answer

- A per-expert `[E,H,H]` map gathered to `[N,C,H,H]` — **275 GB** on a 117 GiB box. It was
  OOM-killed, and the memory pressure took the GPU off the bus on the next suspend/resume
  (see [00-log.md](00-log.md) 2026-09-26).
- The bias was drawn as `N(0, 2.04)` from its **RMS**, ignoring that its *spread* is 0.059. A
  near-constant offset cannot change a top-k, so that fixture let one fixed expert set win every
  token and reported a 0.00 % affected rate — an artifact entirely internal to the fixture, and
  the reason the first LR sweep made the shipped config look fine. `[EST]`

### Kept regardless

Stratified sampling on affected rows (2048/2048 signal rows vs 1338 uniform); acceptance
measured on a **uniform population sample rather than the rows trained on** (the old code judged
on its own training set and structurally could not see the overfit); explicit skip for
zero-signal layers; `affected_rate` recorded per layer; default `lr` 1e-3 → 1e-5.

---

## 2. What the negative result does *not* settle

It is about **one objective**. Layer-local matching asks each layer to reproduce its own output
given the *teacher's* input — teacher-forced at every boundary. It can never observe that a small
layer-7 deviation is amplified by layer 20, nor trade a worse layer-7 output for a better final
distribution. The stage's docstring already conceded it:

> local matching cannot see how errors compose across layers, so it is an approximation, not the
> published method.

The published method (arXiv 2603.02217) distils against the **full model's next-token
distribution**. Different loss, different optimum, untried. `[EST]`

Caveat that keeps this open: the fixture's experts are an elementwise map of the hidden state.
Real expert functions may leave the teacher's router further from optimal than that fixture did.

---

## 3. Scope of the global objective  `[EXT]`

Design and costing are worked out in the GLM repo's
[`wiki/98-router-kd-global.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/98-router-kd-global.md).
The enabling trick is general: **cache the teacher's top-K next-token logprobs to disk**, so the
teacher is a target rather than a resident model.

```
full logits : vocab x 2 B per token          -> terabytes, impossible
top-64      : 64 ids (int32) + 64 lp (fp16)  -> 384 B/token
2.4 M-token KD budget                        -> 0.9 GB on disk
```

Trainable parameters are routers only — 47 MoE layers × 256 experts — so optimizer state and
checkpointed activations are both trivial. The cost is streaming the weights twice per step
(forward, then reverse for backward, because the gradient path from the final logits to router L
traverses every layer above it).

## 4. MiMo-specific blocker  `[EST]`

**The 166 GB source was deleted** (with approval, to clear disk for GLM pass 3). Caching teacher
logits requires the unpruned teacher, so MiMo global KD needs a re-download first. Disk stood at
~128 GB free on 2026-09-26 — **not enough**, so this is a re-download *and* a disk decision.

Spending that to chase an objective whose local sibling found the sliced router already
near-optimal is a poor trade at current evidence. **Recommendation: run the global objective on
GLM first**, where the teacher is already on disk and staged, and port to MiMo only if it shows a
real gain there. `[EXT]`
