#!/usr/bin/env python3
"""The hook figure: ASR vs round, one line per (attack) or per (defense), attack window shaded.

    python scripts/plot_asr.py --root runs/E1_hook --by attack --filter server.aggregator=none server.optimizer=fedavgm --out figs/E1_attacks.pdf
    python scripts/plot_asr.py --root runs/E1_hook runs/E2_cma --by aggregator --filter attack.name=echo --out figs/E2_defenses.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import yaml


def get(cfg, dotted):
    n = cfg
    for p in dotted.split("."):
        n = n.get(p) if isinstance(n, dict) else None
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", default=["runs"])
    ap.add_argument("--by", default="attack")   # attack | aggregator | optimizer
    ap.add_argument("--filter", nargs="*", default=[])
    ap.add_argument("--metric", default="ASR")
    ap.add_argument("--out", default="figs/asr.pdf")
    args = ap.parse_args()
    key = {"attack": "attack.name", "aggregator": "server.aggregator", "optimizer": "server.optimizer"}[args.by]
    series = defaultdict(list); window = None
    for root in args.root:
        for cfile in glob.glob(os.path.join(root, "*", "config.yaml")):
            cfg = yaml.safe_load(open(cfile))
            ok = all(str(get(cfg, f.split("=")[0])) == f.split("=")[1] for f in args.filter)
            if not ok or not os.path.exists(os.path.join(os.path.dirname(cfile), "summary.json")):
                continue
            rounds = [json.loads(l) for l in open(os.path.join(os.path.dirname(cfile), "rounds.jsonl"))]
            pts = [(r["round"], r[args.metric]) for r in rounds if args.metric in r]
            series[str(get(cfg, key))].append(pts)
            window = (cfg["attack"]["start"], cfg["attack"]["start"] + cfg["attack"]["window"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, runs in sorted(series.items()):
        ts = sorted({t for pts in runs for t, _ in pts})
        M = np.full((len(runs), len(ts)), np.nan)
        for i, pts in enumerate(runs):
            d = dict(pts)
            for j, t in enumerate(ts):
                M[i, j] = d.get(t, np.nan)
        mu = np.nanmean(M, 0); sd = np.nanstd(M, 0)
        ax.plot(ts, mu, label=f"{name} (n={len(runs)})"); ax.fill_between(ts, mu - sd, mu + sd, alpha=0.15)
    if window:
        ax.axvspan(window[0], window[1], color="red", alpha=0.15, label="attack window")
    ax.set_xlabel("round"); ax.set_ylabel(args.metric); ax.set_ylim(0, 1.02); ax.legend(fontsize=7)
    fig.tight_layout(); os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); fig.savefig(args.out); print("wrote", args.out)


if __name__ == "__main__":
    main()
