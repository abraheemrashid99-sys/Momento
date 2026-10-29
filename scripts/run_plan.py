#!/usr/bin/env python3
"""Run the pre-registered experiment plan in order, regenerate tables and claim checks after each
block, and (optionally) commit the result tables so they can be reviewed from GitHub.

    python scripts/run_plan.py --seeds 0 1 2 --blocks E3 E1 E2 E7 --commit

Each block skips runs that already have a summary.json, so this is safe to re-run after a crash.
"""
import argparse, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--blocks", nargs="+", default=["E3", "E1", "E2", "E7", "E4", "E6", "E5"])
ap.add_argument("--commit", action="store_true", help="git commit+push runs/ and results/ after EVERY finished run (and tables after each block)")
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
py = sys.executable

# single-instance lock: a second copy (e.g. started by the logon task while one is already running) exits
lock = os.path.join("runs", ".plan.lock")
os.makedirs("runs", exist_ok=True)
if os.path.exists(lock):
    try:
        pid = int(open(lock).read().strip())
        alive = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout if os.name == "nt" else ""
        if (os.name == "nt" and str(pid) in alive) or (os.name != "nt" and os.path.exists(f"/proc/{pid}")):
            print(f"[plan] another instance (pid {pid}) is already running; exiting."); sys.exit(0)
    except Exception:
        pass
open(lock, "w").write(str(os.getpid()))
import atexit; atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
for b in a.blocks:
    cmd = [py, "scripts/sweep.py", b, "--seeds", *map(str, a.seeds), "--run"] + (["--quick"] if a.quick else []) + (["--commit"] if a.commit else [])
    print("=" * 80, "\n[plan] block", b, "\n", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)
    subprocess.run([py, "scripts/make_tables.py", "--root", "runs", "--out", "results"], check=False)
    subprocess.run([py, "scripts/verify_claims.py", "--root", "runs"], check=False)
    if a.commit:
        subprocess.run(["git", "add", "results"], check=False)
        subprocess.run(["git", "commit", "-q", "-m", f"results after block {b} (seeds {a.seeds})"], check=False)
        subprocess.run(["git", "push", "-q"], check=False)
