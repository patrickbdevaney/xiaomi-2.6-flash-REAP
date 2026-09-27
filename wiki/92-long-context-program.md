# 92 — Making the REAP hold at native 1M context

*Opened 2026-09-26. Requirement stated directly: the REAP must work "as optimally as possible
with a kv cache context length of the total native 1 million context for niah, longppl and
multi hop long horizon agentic contextual use".*

This page is the plan and the arithmetic. It supersedes nothing in
[91-capability-coverage.md](91-capability-coverage.md); it is what to do about it.

---

## 1. The load-bearing fact: 1M context fits, but only just  `[EST]`

MEASURED from `~/models/MiMo-V2.6-Flash-REAP50/config.json` — GQA with 4 KV heads × 192 head_dim,
and **39 of 48 layers are SWA with `sliding_window: 128`, so their KV cache is capped at the
window, not the context**:

| component | at 1,048,576 tokens |
|---|---|
| 9 full-attention layers | **27.00 GiB** |
| 39 SWA-128 layers | **0.014 GiB**, capped at the 128 window |
| total KV state | **27.01 GiB** |
| REAP50 weights on disk | 88.00 GiB |
| **resident at full 1M** | **115.01 GiB of a 117 GiB envelope** |

For contrast, if all 48 layers were full attention this would be **144 GiB** and would not fit at
all. The SWA majority is what makes 1M reachable — and the REAP50 is the other half of it: at the
unpruned size there is no 1M on this box at any context.

**The margin is ~2 GiB.** That is not comfortable. NIAH at true 1M on MiMo will need KV
quantisation or a slightly reduced ceiling, where GLM has 7.7 GiB of headroom to play with (see
the GLM repo's `wiki/92`, which measures 109.27 GiB for the same test). Treat 1M-on-MiMo as
"demonstrable" rather than "operationally comfortable". `[EST]`

**Consequence: NIAH and LongPPL at genuine long context are runnable on hardware already
owned** — no cloud spend, no hosted eval.

## 2. Why calibrating *at* 1M is the wrong goal  `[EXT]`

Saliency holds hidden states per row: `S × 4096 × 2 B`.

| S | per row, per copy |
|---|---|
| 2,048 | 0.016 GiB |
| 16,384 | 0.125 GiB |
| 131,072 | 1.0 GiB |
| 1,048,576 | **8.0 GiB** |

At 1M a single row costs 8 GiB, so a statistically meaningful number of rows is impossible. Brute
force is out. On MiMo it is also **largely unnecessary**, and the reason is structural and
stronger here than on GLM:

- **39 of 48 layers are SWA with a window of 128.** They cannot see past 128 tokens at any S.
  Their routers are S-invariant *by construction*, not by saturation.
- **The 9 full-attention layers are genuinely unbounded.** They are the entire long-context
  exposure of this model, and nothing caps them.

So the MiMo question is much narrower than GLM's: **does saliency on those 9 layers change with
S?** 39/48 of the answer is already known. Prediction with a sign, which
is what makes it a test rather than a hope:

> Per-expert saliency should change with S up to some saturation length S*, and stop changing
> above it. Calibrating at S* is then equivalent to calibrating at 1M, at a fraction of the cost.

The goal is therefore **find S\*, calibrate at or above it, and verify at true 1M** — not
calibrate at 1M.

---

## 3. The cheap direct test nobody has run  `[OPEN]`

Everything above is about choosing a *new* mask. There is a much cheaper question about the mask
we have already shipped, and the instrumentation for it already exists from the router-KD work
(`_affected_rows` in `router_kd_run.py`):

> Run long-context tokens through the **existing** mask and measure the fraction of routed slots
> that hit a pruned expert, **as a function of token position**.

- Flat in position → long-context routing is unaffected; the shipped artifact is fine and this
  whole program is insurance.
- Rising with position → the shipped model degrades with context, and we would know it without
  recalibrating anything.

This measured **1.13 % of slots / 8.71 % of tokens** overall on MiMo — but at ≤4,095 tokens, so
it says nothing about position. Layer 7 alone ran at 18.18 % of slots; if that layer is one of
the 9 full-attention layers, it is the first place to look. This is hours of work, needs no new
calibration, and can falsify the entire concern in either direction. **Run it first.**

---

## 4. Evals: the rulers that are not self-confirming  `[EXT]`

[`../research/REAP_METHOD_AND_FINETUNE_VIABILITY_2026-09-26.md`](../research/REAP_METHOD_AND_FINETUNE_VIABILITY_2026-09-26.md)
establishes that worst-domain retention and `pᵀFp` are **rigged rulers** — each is maximised by
construction by the selector that optimises it. NIAH, LongPPL and multi-hop agentic traces are
none of those things: they are external, and they measure the capability directly.

- **NIAH** — synthetic by construction, so no corpus problem. Sweep needle depth × context
  length up to 1M. Cheapest of the three.
- **LongPPL** — needs *real* long documents. The current corpus caps at 16,384 and cannot supply
  them; sourcing long books/repos is a prerequisite, not a detail.
- **Multi-hop long-horizon agentic** — the one the user actually daily-drives, and the hardest to
  score. Likely paired teacher/student on real trajectories rather than a pass/fail metric.

All three run against the paired teacher/student protocol already in `s09_eval`.

---

## 5. Honest limits  `[EST]`

1. **If 50 % pruning genuinely removes capacity that long context needs, no calibration recovers
   it.** Calibration chooses *which* experts to keep, not how many. The only lever would be a
   lower global ratio.
2. **A ragged per-layer budget would be the ideal lever and we measured it as nearly worthless.**
   Non-uniform layer allocation bought **+0.0115** in our own test, not EvoESAP's +15.2 %. On GLM
   it is additionally impossible (one scalar `num_local_experts` for all layers). Protecting the
   9 full-attention layers at a lower prune ratio is the obvious idea and the evidence says it
   would buy very little.
3. **The corpus cannot currently test above 16,384 tokens.** Extending the ladder to find S*
   requires genuinely long documents first. Concatenating existing samples would manufacture fake
   long-range structure and answer the wrong question.
4. **MiMo cannot be recalibrated without the teacher.** The 166 GB unpruned source was deleted;
   disk stands at 129 GB free, so a re-download fits but leaves little room. Note that
   `~/models/MiMo-V2.6-Flash-REAP50` (88 GB, pruned) **is** still on disk — enough to run NIAH and
   LongPPL on the shipped artifact, and enough to ask whether *its* routing changes with position.
   It is not enough for §3's counterfactual, which needs the unpruned 256-expert router to ask
   what *would* have been routed to a pruned expert.

---

## 6. Order of work

1. **§3 affected-rate vs position**, on the existing mask. Cheapest, and can end the concern.
2. **Finish the S ladder** (`exp_seqlen_saliency.py`, running 2048/8192/16384 + control).
3. **Source long documents**, extend the ladder to 64K/128K, locate S*.
4. **Build NIAH at 1M** — §1 says it fits.
5. Only then decide `calib_max_len` for pass 3.

LongPPL and the agentic suite follow; they are the more expensive rulers and the ladder result
determines how much they need to cover.
