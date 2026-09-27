# MiMo-V2.6-Flash REAP — Working Wiki

Append-only knowledge base for this project. **Everything** learned from web research, paper
reading, repo inspection, or on-box experimentation lands here — not in terminal scrollback,
not only in a chat reply.

Conventions are shared with the GLM-5.3-Flash repo's wiki, deliberately, so findings can be
cross-referenced between the two REAPs.

## How to use this wiki

- **Never rewrite history.** Correct a claim by appending a new dated entry that supersedes it
  and editing the old line to say `~~superseded~~ → see [YYYY-MM-DD]`.
- Every factual claim carries a **source**: arXiv ID, repo path + commit, HF model ID, or
  `MEASURED` (with the command that produced it).
- Confidence marker on every claim:
  - `[EST]` — established: reproduced, measured, or in a peer-reviewed/primary source
  - `[VEN]` — vendor/marketing claim, unverified
  - `[EXT]` — our own extrapolation or inference
  - `[OPEN]` — known unknown, no source exists
- New topic → new numbered file, add a line to the index below.

## Index

| File | Topic |
|---|---|
| [00-log.md](00-log.md) | Chronological append-only log of every session, finding, and experiment |
| [91-capability-coverage.md](91-capability-coverage.md) | What calibration protects: long-context and non-English gaps |
| [92-long-context-program.md](92-long-context-program.md) | Holding the REAP at native 1M: arithmetic, SWA structure, NIAH/LongPPL plan |
| [98-router-kd-global.md](98-router-kd-global.md) | Router KD: the local objective's negative result, and the global scope |

Longer method write-ups live in [`../research/`](../research/); this wiki is the index of
findings and the place new ones land first.
