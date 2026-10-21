#!/usr/bin/env python3
"""Pre-registered claim checks. Each claim is re-derived from run records (never from numbers.json)
and reported PASS / FAIL / NO-DATA. The manuscript may only state claims that PASS here.

    python scripts/verify_claims.py --root runs

Claims (from the proposal's pre-registration; thresholds fixed BEFORE experiments):
  C1  Echo lifespan_50 under FedAvgM (no defense) >= 2x that of 'scale'            [attack contribution]
  C2  Echo ASR at window end under each OTA defense within 10 pts of no-defense     [survives defenses]
  C3  Echo under FedAdam has longer lifespan than under FedAvgM                     [preconditioner echo]
  C4  Reset ablation: lifespan with state reset < lifespan without  (echo)          [persistence is in the state]
  C5  CMA: asr_plus_50 < 0.15 and lifespan_50 reduced >= 80% vs FedAvgM no-defense  [defense]
  C6  CMA clean cost: final_MA(no attack, CMA) >= final_MA(no attack, FedAvgM) - 0.02
  C7  CMA detector FPR <= 0.05 on benign-null runs (E7)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_runs(root):
    out = []
    for sfile in glob.glob(os.path.join(root, "*", "*", "summary.json")):
        d = os.path.dirname(sfile)
        with open(os.path.join(d, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        with open(sfile) as f:
            s = json.load(f)
        out.append((cfg, s))
    return out


def sel(runs, **kw):
    res = []
    for cfg, s in runs:
        ok = True
        for k, v in kw.items():
            node = cfg
            for p in k.split("__"):
                node = node.get(p, {}) if isinstance(node, dict) else None
                if node is None:
                    break
            if node != v and not (node == {} and v is None):
                ok = False; break
        if ok:
            res.append(s)
    return res


def mean(xs, key):
    v = [x[key] for x in xs if x.get(key) is not None and x.get(key) != "inf" and not (isinstance(x[key], float) and np.isnan(x[key]))]
    return float(np.mean(v)) if v else None


def life(xs):
    v = []
    for x in xs:
        l = x.get("lifespan_50")
        if l is None:
            continue
        v.append(1e9 if (l == "inf" or l == float("inf")) else float(l))
    return float(np.mean(v)) if v else None


def report(name, cond, detail):
    status = "NO-DATA" if cond is None else ("PASS" if cond else "FAIL")
    print(f"[{status:7s}] {name}: {detail}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); a = ap.parse_args()
    R = load_runs(a.root)
    base = dict(dataset__name="cifar10", server__optimizer="fedavgm", server__aggregator="none", ablation__reset_state_at_window_end=False)
    echo = sel(R, attack__name="echo", **base); scale = sel(R, attack__name="scale", **base)
    le, ls = life(echo), life(scale)
    report("C1 echo lifespan >= 2x scale (FedAvgM, no defense)", None if (le is None or ls is None) else le >= 2 * ls, f"echo={le} scale={ls}")
    a0 = mean(echo, "asr_window_end")
    for agg in ["bev", "rotaf", "aircomp_gm", "group_cc", "group_tm", "fedsac", "tti"]:
        x = sel(R, attack__name="echo", dataset__name="cifar10", server__aggregator=agg)
        ax = mean(x, "asr_window_end")
        report(f"C2 echo survives {agg}", None if (a0 is None or ax is None) else ax >= a0 - 0.10, f"no-def={a0} {agg}={ax}")
    adam = sel(R, attack__name="echo_adam", dataset__name="cifar10", server__optimizer="fedadam", server__aggregator="none", ablation__reset_state_at_window_end=False)
    la = life(adam)
    report("C3 echo_adam lifespan > echo (FedAvgM)", None if (la is None or le is None) else la > le, f"adam={la} avgm={le}")
    rst = sel(R, attack__name="echo", dataset__name="cifar10", server__optimizer="fedavgm", server__aggregator="none", ablation__reset_state_at_window_end=True)
    lr = life(rst)
    report("C4 reset ablation shortens echo lifespan", None if (lr is None or le is None) else lr < le, f"reset={lr} noreset={le}")
    cma = sel(R, attack__name="echo", dataset__name="cifar10", server__aggregator="cma")
    lc, ac = life(cma), mean(cma, "asr_plus_50")
    report("C5 CMA: asr_plus_50 < 0.15 and lifespan -80%", None if (lc is None or le is None or ac is None) else (ac < 0.15 and lc <= 0.2 * le), f"cma asr+50={ac} life={lc} vs {le}")
    cl = sel(R, attack__name="none", dataset__name="cifar10", server__aggregator="cma"); bl = sel(R, attack__name="none", **base)
    mc, mb = mean(cl, "final_MA"), mean(bl, "final_MA")
    report("C6 CMA clean cost <= 2pp", None if (mc is None or mb is None) else mc >= mb - 0.02, f"cma={mc} base={mb}")
    nul = sel(R, attack__name="none", server__aggregator="cma", run__exp="E7_theory")
    fpr = mean(nul, "det_fpr")
    report("C7 CMA FPR <= 0.05 on benign null", None if fpr is None else fpr <= 0.05, f"fpr={fpr}")


if __name__ == "__main__":
    main()
