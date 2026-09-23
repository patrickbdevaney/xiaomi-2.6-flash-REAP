"""Run the vision/audio towers in a DISPOSABLE subprocess.

WHY THIS EXISTS, AND WHY NOTHING ELSE WORKS HERE
------------------------------------------------
Four runs died at the image bucket. Each time the fix was a better prediction of the next
allocation, and each time a different allocation appeared. The reason that game cannot be won on
this box is structural:

  * Thor has unified memory, so a CUDA allocation is ordinary system RAM that the driver pins.
    It appears in NO Rss* counter of the owning process (measured: every process on the machine
    summed to 2.06 GiB while 115 GiB was held).
  * cgroup v2 `memory.max` / systemd `MemoryMax` therefore do not bound it -- the accounting
    never sees it. This is documented behaviour on unified-memory machines (GB10, Jetson,
    Grace), not a misconfiguration.
  * This kernel exposes no PSI, so systemd-oomd cannot run at all.
  * The kernel OOM killer ranks victims by RSS, i.e. by precisely the number that does not
    track the real consumer -- so it repeatedly killed the wrong thing.

There is consequently NO kernel-side limit that can be set on the allocation. The only lever
left is blast radius: make the process that performs the allocation cheap to lose.

WHAT THIS BUYS
--------------
The towers run here and nowhere else. The worker sets its own oom_score_adj to 1000, making it
by far the most attractive victim on the machine, so when something does go wrong the kernel
takes the worker instead of the parent, the session, or the desktop. The parent notices the
dead worker, drops the one sample that caused it, restarts the worker and carries on. An
unforeseen allocation costs one sample instead of the run.

This is deliberately the LAST line of defence, not the first: the patch-row budget in
media_loaders still bounds the common case, and a pre-flight cost check refuses calls predicted
to be too large. Those keep the worker alive; this keeps the RUN alive when they are wrong.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

READY = "__ready__"


def _worker_main(src, device, req, res):
    """Child entry point: own the towers, answer requests, never hold anything else."""
    try:
        # Volunteer as the OOM victim. The whole design rests on this: if the kernel must kill
        # something, it must be this process and not the run, the session or the desktop.
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except OSError:
            pass
        sys.path.insert(0, str(Path(__file__).parent))
        import torch
        from transformers import AutoConfig
        import chunk_builder as CB
        import media_loaders as ML

        cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        cfg._name_or_path = str(src)
        cfg._attn_implementation = "flex_attention" if str(device).startswith("cuda") else "eager"
        E = CB.Embedder(src, cfg, device=device)
        A = ML.AudioLoader(E, src)
        res.put((READY, None))

        while True:
            item = req.get()
            if item is None:
                return
            kind, payload = item
            try:
                with torch.no_grad():
                    if kind == "visual":
                        pv, grid = payload
                        out = E._lazy("visual")(pixel_values=pv.to(device),
                                                grid_thw=grid.to(device))
                    elif kind == "audio":
                        out = A.embed(payload)
                    else:
                        raise ValueError(f"unknown request {kind!r}")
                res.put(("ok", out.to("cpu", torch.bfloat16)))
            except Exception:                       # noqa: BLE001
                res.put(("err", traceback.format_exc(limit=3)))
            finally:
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
    except Exception:                               # noqa: BLE001
        try:
            res.put(("fatal", traceback.format_exc(limit=5)))
        except Exception:
            pass


class MediaWorker:
    """Client side. Every failure mode of the child collapses to `returns None`."""

    def __init__(self, src, device="cuda", start_timeout=900, call_timeout=900):
        self.src, self.device = str(src), device
        self.start_timeout, self.call_timeout = start_timeout, call_timeout
        self.deaths = 0
        self.skipped = 0
        self._p = None
        self.start()

    def start(self):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")           # fork would inherit the parent's CUDA context
        self._req, self._res = ctx.Queue(), ctx.Queue()
        self._p = ctx.Process(target=_worker_main,
                              args=(self.src, self.device, self._req, self._res), daemon=True)
        self._p.start()
        # Poll rather than block on the queue. A child that dies during startup -- a bad import,
        # a missing shard, or the recursive-spawn error you get without an
        # `if __name__ == "__main__"` guard -- never puts anything on the queue, and a blind
        # get(timeout=900) then hangs the entire run for fifteen minutes before reporting.
        # Watching is_alive() turns that into an immediate, accurate error.
        import queue as _q
        deadline = time.time() + self.start_timeout
        tag = info = None
        while time.time() < deadline:
            try:
                tag, info = self._res.get(timeout=1.0)
                break
            except _q.Empty:
                if not self._p.is_alive():
                    raise RuntimeError(
                        f"media worker exited during startup (code {self._p.exitcode}) without "
                        f"reporting ready -- check its traceback on stderr")
        if tag is None:
            raise RuntimeError(f"media worker did not become ready in {self.start_timeout}s")
        if tag != READY:
            raise RuntimeError(f"media worker failed to start: {info}")

    def _restart(self):
        self.deaths += 1
        try:
            if self._p is not None and self._p.is_alive():
                self._p.terminate()
                self._p.join(timeout=30)
        except Exception:
            pass
        self.start()

    def _call(self, kind, payload):
        import queue as _q
        t_call = time.time()
        try:
            self._req.put((kind, payload))
        except Exception:
            self._restart()
            return None
        try:
            deadline = time.time() + self.call_timeout
            tag = val = None
            while time.time() < deadline:
                try:
                    tag, val = self._res.get(timeout=1.0)
                    break
                except _q.Empty:
                    if not self._p.is_alive():
                        # THE OOM CASE. The kernel took the worker mid-call; there will never
                        # be an answer. Notice it in a second rather than waiting out the
                        # timeout, so the run keeps moving.
                        raise _q.Empty()
            if tag is None:
                raise _q.Empty()
        except _q.Empty:
            # Either the child was killed mid-call (the OOM case) or it wedged. Both are the
            # same to us: lose the sample, not the run.
            print(f"    media worker stopped answering after {time.time()-t_call:.0f}s "
                  f"(limit {self.call_timeout}s) -- assuming it was killed; restarting",
                  flush=True)
            self.skipped += 1
            self._restart()
            return None
        except Exception:
            self.skipped += 1
            self._restart()
            return None
        if tag == "ok":
            return val
        print(f"    media worker error ({kind}): {str(val)[:300]}", flush=True)
        self.skipped += 1
        if tag == "fatal":
            self._restart()
        return None

    def visual(self, pixel_values, grid_thw):
        return self._call("visual", (pixel_values.cpu(), grid_thw.cpu()))

    def audio(self, waves):
        return self._call("audio", waves)

    def close(self):
        try:
            self._req.put(None)
            self._p.join(timeout=30)
        except Exception:
            pass
