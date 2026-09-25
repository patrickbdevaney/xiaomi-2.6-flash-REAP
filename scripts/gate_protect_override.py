"""Gate the late-binding protect-frac override.

PROTECT_FRAC was chosen 26 hours into a 30-hour run. run_reap.sh had already expanded its copy
of that variable and its script text is pinned to the inode bash started from, so the value can
only reach stage 5 through a file that reap_select reads at invocation. That makes this file the
single point where a wrong answer becomes a wrong checkpoint, with nothing downstream to catch
it -- the mask would have the right shape and plausible retention either way. So: the override
must beat the CLI, it must be recorded in the mask, and its absence must change nothing.
"""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def synth(n_layer=3, n_bucket=4, n_exp=32, seed=0):
    """A local copy of gate_reap_select's fixture. IMPORTING it would run that gate's whole
    module body -- it is a script, not a library, and it calls sys.exit()."""
    g = torch.Generator().manual_seed(seed)
    acc = {}
    f_sum = torch.zeros(n_layer + 1, n_exp, n_exp, dtype=torch.float64)
    f_cnt = torch.zeros(n_layer + 1, n_exp, n_exp, dtype=torch.float64)
    for li in range(n_layer):
        cnt = torch.randint(50, 500, (n_bucket, n_exp), generator=g).double()
        s = torch.rand(n_bucket, n_exp, generator=g).double() * cnt
        acc[f"model.layers.{li+1}.mlp"] = {"sum": s, "sq": s * s / cnt.clamp(min=1), "cnt": cnt,
                                           "nrm": s, "nsq": s * s, "gat": cnt, "gsq": cnt}
        d = s.sum(0) / cnt.sum(0)
        f_sum[li + 1] = torch.outer(d, d) * 100
        f_cnt[li + 1] = torch.full((n_exp, n_exp), 100.0, dtype=torch.float64)
    return acc, f_sum, f_cnt, [f"b{i}" for i in range(n_bucket)]

PY = sys.executable
SEL = str(Path(__file__).parent / "reap_select.py")
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def run_cli(acc, out, cli_frac, ovr_path):
    env = {"PATH": "/usr/bin:/bin", "PROTECT_FRAC_FILE": str(ovr_path)}
    p = subprocess.run([PY, SEL, "--acc", str(acc), "--out", str(out), "--ratio", "0.5",
                        "--mode", "hope", "--protect-frac", str(cli_frac)],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit("reap_select CLI failed")
    return json.loads(out.read_text()), p.stdout


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    acc, f_sum, f_cnt, buckets = synth(n_exp=32)
    apath = td / "accumulators.pt"
    torch.save({"acc": acc, "f_sum": f_sum, "f_cnt": f_cnt, "buckets": buckets}, apath)
    ovr = td / "protect.txt"

    print("[1] with no override file, the CLI value stands")
    r, _ = run_cli(apath, td / "a.json", 0.0, ovr)
    check("protect_frac 0.0 recorded", r["protect_frac"] == 0.0, str(r["protect_frac"]))

    print("\n[2] the override beats an explicit CLI value")
    ovr.write_text("0.08  # comment and trailing junk must not break the parse\n")
    r2, out2 = run_cli(apath, td / "b.json", 0.0, ovr)
    check("protect_frac 0.08 recorded in the mask", r2["protect_frac"] == 0.08,
          str(r2["protect_frac"]))
    check("the override is announced on stdout", "OVERRIDE 0.0 -> 0.08" in out2,
          out2.strip().splitlines()[0] if out2.strip() else "(no output)")

    print("\n[3] it changes the mask, not just the label")
    m0 = json.loads((td / "a.json").read_text())["mask"]
    m1 = json.loads((td / "b.json").read_text())["mask"]
    check("same number of experts pruned per layer",
          all(len(m0[k]) == len(m1[k]) for k in m0))
    check("at least one layer's prune set differs", any(m0[k] != m1[k] for k in m0))

    print("\n[4] protection can only RAISE the worst domain, never lower it")
    check(f"worst {r['worst_retention']:.6f} -> {r2['worst_retention']:.6f}",
          r2["worst_retention"] >= r["worst_retention"] - 1e-12)

    print("\n[5] the shipped override file parses to the value that was decided")
    shipped = Path(__file__).parent.parent / "conf" / "protect_frac_override.txt"
    check("conf/protect_frac_override.txt exists", shipped.exists())
    if shipped.exists():
        v = float(shipped.read_text().split("#")[0].strip())
        check(f"it reads 0.08 (got {v})", v == 0.08)

print("\nGATE " + ("FAIL: " + ", ".join(FAIL) if FAIL else "PASS"))
sys.exit(1 if FAIL else 0)
