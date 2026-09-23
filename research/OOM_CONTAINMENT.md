# Why this run kept dying, and what now makes it survivable

Four OOM kills, four different explanations, each one a better prediction of the next
allocation. This document is the reason that game stopped being played.

## The thing that makes Thor different

On a unified-memory machine a CUDA allocation is **ordinary system RAM that the driver pins**.
Three consequences, all measured on this box rather than assumed:

| | |
|---|---|
| It is charged to no cgroup | `MemoryMax` / `docker --memory` do not bound it. Measured elsewhere on this hardware: a touched 2048 MiB `cudaMalloc` charges **42 MiB** to the calling cgroup. |
| It appears in no `Rss*` counter | During one collapse every process on the machine summed to **2.06 GiB** while **115 GiB** was held. |
| This kernel exposes no PSI | `systemd-oomd` cannot run at all. |

So the kernel OOM killer ranks victims by RSS — *precisely the number that does not track the
real consumer* — and it repeatedly chose wrong. systemd's own report of the failure said the
unit peaked at **7.0 GiB**, then **10.7 GiB**, on a 122 GiB box. Those numbers are not wrong;
they are simply not measuring the thing that killed us.

**There is no kernel-side limit that can be set on the allocation.** Every fix that tried to
predict allocations was therefore working without a safety net, which is why each one held only
until the next unpredicted allocation.

## What actually allocated

The 5-second memory sampler was added after the second failure and settled it on the fourth:

```
13:38:10  avail=117434MB  cached=3080MB
13:38:15  avail=113300MB  cached=4479MB
13:38:20  avail= 48266MB  cached=4662MB     <- ~70 GB in five seconds
13:38:25  avail=  1104MB  cached= 746MB
```

`cached` never moves. Not page cache, not a leak, not the driver pool — one live allocation, in
the vision tower. `MiMoVisionAttention` materialises a dense `[1, num_heads, L, L]` tensor
**explicitly**, purely to place a sink bias in column 0, then `attn_mask + sink_bias` makes a
second, and an explicit float mask forces SDPA onto the math path, which materialises the scores
as a third. With `num_heads=32` that is ~`64·L²` bytes per copy. A WebSight screenshot is
2560×2176 → **L = 21,760 → 30.3 GB per copy**.

The checkpoint's own `preprocessor_config.json` permits `max_pixels = 12,845,056` (L = 50,176,
a 161 TB sink_bias), so the configured cap is effectively absent.

## The layered defence

Ordered from "keeps the worker alive" to "keeps the run alive". Each layer assumes the ones
above it will eventually be wrong about something.

1. **Pre-flight reclaim + floor.** `drop_caches` with the full shrinker, then refuse to start
   below 90 GiB available. The corpus stage is not resumable mid-bucket; starting into a
   poisoned allocator only buys another failure.
2. **Patch-row budget** (`MAX_PATCH_ROWS = 4096`). Derived from the measured cost law, not from
   image aesthetics. Enforced by resizing images ourselves — setting `max_pixels` and
   `size["longest_edge"]` on `Qwen2VLImageProcessor` was measured to change **nothing** (still
   21,760 rows). A cap the callee may ignore is not a cap. Video shares the budget across the
   temporal extent, because a clip is one attention chunk (`L = T·h·w`).
3. **Hard post-check.** Any sample still over budget after resizing is skipped and counted.
4. **Process isolation** (`media_worker.py`). The towers run in a disposable subprocess that
   sets its own `oom_score_adj` to 1000, making it the most attractive victim on the machine.
   Any death — SIGKILL included — returns `None`, the sample is dropped, the worker restarts,
   the run continues. This is what makes an *unpredicted* allocation survivable.
5. **The stages volunteer too** (`oom_score_adj = 500`). Above the session (200) and the desktop
   (0), below the worker. If the box is cornered, it takes this run rather than the user's
   session — and this run resumes exactly.
6. **Exact resume + bounded auto-restart.** Stage 1 checkpoints per bucket; stage 2 per chunk,
   with the accumulators written atomically and verified *before* the checkpoint.
   `Restart=on-failure`, capped at 6 per 6 hours.
7. **memguard**, fixed: tier 1 escalates to the full shrinker and logs a no-op reclaim as a
   FAILURE (it had been reclaiming 0 bytes, ~8,600 times a day, logged as success). Tier 2 may
   now kill these stages, which is safe only because they resume.
8. **The 5-second memory trace**, which is how any future failure gets a trajectory instead of
   a cgroup peak that means nothing.

## Allocation sites, and what bounds each

| Site | Bound | If the bound is wrong |
|---|---|---|
| Vision tower forward | `MAX_PATCH_ROWS=4096` → **4.73 GiB measured peak** | worker dies, sample skipped |
| Audio tower forward | 30 s clip → `[L,L]` mask, ~288 MB | worker dies, sample skipped |
| Tower weights | ~2.4 GiB, loaded once in the worker | worker restart reloads |
| Corpus parent | token ids + chunk buffer only; **no CUDA context** | bucket resume |
| Stage-2 activations | 2M-token chunk × 4096 × 2 B = **15.3 GiB**, computed against the real config | chunk resume |
| Stage-2 layer weights | one layer resident, shards released each layer | chunk resume |

## What was ruled out by measurement, not argument

- **Unreleased shard mmaps.** The `release()` docstring describes exactly this failure, so it
  was a clean story. A/B measured **7.2 GiB pinned without the release vs 6.2 GiB with it** —
  worth ~1 GiB, not 100. Kept because it is correct and free, not because it is the fix.
- **Page cache.** `cached` never moved during any collapse.
- **The text buckets.** Probed flat at 118 GiB for the full run.
- **The isolated image path.** Probed flat at 114 GiB — which is why the batch, not the single
  image, had to be the answer.

## Sources

- [GB10: server MemoryMax (and docker --memory) does not bound CUDA allocations](https://github.com/evanwtf/local-llm/issues/456)
- [gpuoom — GPU-aware early OOM killer for unified-memory Linux machines](https://github.com/tom-doerr/gpuoom)
- [How Jetson allocates memory for GPU — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/how-jetson-allocate-memory-for-gpu/356364)
- [Memory architecture and CUDA programming on Jetson Orin](https://nvidia-jetson.piveral.com/jetson-orin-nano/memory-architecture-and-cuda-programming-on-jetson-orin-differences-from-x86-gpus/)
