# Chronological log

Append-only. Newest entries at the bottom. See [README.md](README.md) for conventions.

---

## 2026-09-26 — Wiki stood up; router KD closed as a negative result; coverage audited

This repo had a `research/` directory but no wiki, so findings from the REAP itself were living
in transcripts and commit messages. Standing one up now, matching the GLM repo's conventions so
the two REAPs cross-reference.

**Router KD.** The layer-local objective was measured and rejected — 7/7 gate in
`scripts/gate_router_kd_stratified.py`. No sampling rule × budget × step beats the teacher's
sliced router; damage falls monotonically toward "change nothing". **My own earlier claim that
the token budget was the cause is withdrawn**: only 1.13 % of routed slots hit a pruned expert,
so the stage was out of work, not starved. Full write-up, including the two fixture bugs that
each would have given a confident wrong answer, in [98-router-kd-global.md](98-router-kd-global.md).

Kept regardless: stratified sampling on affected rows, acceptance on a held-out **population**
sample rather than the training rows, explicit zero-signal skip, per-layer `affected_rate`,
default `lr` 1e-3 → 1e-5. Commit `6ef2700`.

**Coverage audit** → [91-capability-coverage.md](91-capability-coverage.md). All three media
modalities carry real routed tokens (6.0 M, 12.3 % of corpus). Long-context exposure is lower
here than on GLM — `SEQ_LEN` 4096 and 39/48 SWA-128 layers — but `exp_seqlen_saliency.py`, which
was written precisely to test it, never ran before the source was deleted. Non-English is
untested; the sibling GLM corpus measured 0.3 % CJK.

**Incident.** `diag_router_kd.py`'s fixture allocated 275 GB on a 117 GiB box (`[N,C,H,H]`
gather). It was OOM-killed, and the memory pressure then starved the NVIDIA GSP's suspend/resume
buffer on a wake, taking the GPU off the bus and halting GLM pass 3. Reported at the time as
"memory came back clean" — **true of RAM; I never checked the GPU.** Detail in the GLM repo's
`wiki/00-log.md`.
