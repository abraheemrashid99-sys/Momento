#!/usr/bin/env python3
"""Proposition C check: is the CMA disagreement statistic D_t under the null (benign rounds) well
described by a scaled chi-square  c * chi2_r ?  Reads detector_trace.npz from benign CMA runs (E7),
fits (c, r) by method of moments on the warm-up rounds, and runs a KS test on the post-warm-up
benign rounds. Also plots the empirical vs fitted CDF and the D_t trace.

    python theory/detector_null.py --root runs/E7_theory --plot figs/detector_null.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/E7_theory")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    traces = []
    for f in glob.glob(os.path.join(args.root, "*", "detector_trace.npz")):
        z = np.load(f); cfg_dir = os.path.dirname(f)
        rounds = [json.loads(l) for l in open(os.path.join(cfg_dir, "rounds.jsonl"))]
        warm = np.array([r.get("det_warmup", True) for r in rounds if "det_D" in r])
        D = z["D"]
        traces.append((cfg_dir, D, warm))
    if not traces:
        print("no traces"); return
    for d, D, warm in traces:
        Dw, Dp = D[warm][2:], D[~warm]
        if len(Dw) < 5 or len(Dp) < 5:
            continue
        mu, var = Dw.mean(), Dw.var()
        c = var / (2 * mu); r = mu / c
        ks = stats.kstest(Dp, lambda x: stats.chi2.cdf(x / c, df=r))
        q99 = c * stats.chi2.ppf(0.99, df=r)
        print(f"{os.path.basename(d)}: fit c={c:.3g} r={r:.1f}  KS p={ks.pvalue:.3f}  empirical FPR@chi2-0.99 = {(Dp > q99).mean():.3f}")
    if args.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d, D, warm = traces[0]
        Dw = D[warm][2:]; mu, var = Dw.mean(), Dw.var(); c = var / (2 * mu); r = mu / c
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(D); ax[0].axvspan(0, warm.sum(), color="gray", alpha=0.2, label="warm-up"); ax[0].set_xlabel("round"); ax[0].set_ylabel("$D_t$"); ax[0].legend()
        x = np.sort(D[~warm]); ax[1].step(x, np.arange(1, len(x) + 1) / len(x), label="empirical (benign)")
        xx = np.linspace(0, x.max(), 200); ax[1].plot(xx, stats.chi2.cdf(xx / c, df=r), label=f"$c\\chi^2_r$, r={r:.1f}")
        ax[1].set_xlabel("$D$"); ax[1].set_ylabel("CDF"); ax[1].legend()
        fig.tight_layout(); os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True); fig.savefig(args.plot); print("wrote", args.plot)


if __name__ == "__main__":
    main()
