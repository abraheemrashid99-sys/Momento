#!/usr/bin/env python3
"""Collect every run under runs/ into a tidy CSV + numbers.json (mean ± std over seeds per config cell).

    python scripts/make_tables.py --root runs --out results/

Produces
  results/all_runs.csv         one row per run (config fields flattened + summary metrics)
  results/cells.csv            grouped by (exp, dataset, server, aggregator, attack, ...) : mean/std/n
  results/numbers.json         the same cells keyed for the paper (used by verify_claims.py / fill.py)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KEYS = ["exp", "dataset", "alpha", "model", "optimizer", "beta", "eta_s", "aggregator", "K", "attack", "window",
        "malicious_frac", "rho", "snr_db", "fading", "n_slots", "power_control", "reset", "csi_error_var", "sparsify_frac",
        "local_lr", "local_epochs"]
METRICS = ["final_MA", "final_ASR", "asr_window_end", "ma_window_end", "lifespan_50", "auc_asr_post", "asr_plus_20",
           "asr_plus_50", "asr_plus_100", "det_tpr", "det_fpr", "tti_total_extra_bytes"]


def flatten(cfg):
    s = cfg.get("server", {}); ok = s.get("opt_kw", {}); ak = s.get("agg_kw", {}); ch = cfg.get("channel", {})
    return {
        "exp": cfg.get("run", {}).get("exp"), "dataset": cfg["dataset"]["name"], "alpha": cfg["dataset"].get("alpha"),
        "model": cfg["model"]["name"], "optimizer": s.get("optimizer"), "beta": ok.get("beta", ok.get("beta1")),
        "eta_s": ok.get("eta_s"), "aggregator": s.get("aggregator"), "K": ak.get("K"),
        "attack": cfg["attack"].get("name"), "window": cfg["attack"].get("window"),
        "malicious_frac": cfg["attack"].get("malicious_frac"), "rho": cfg["attack"].get("rho"),
        "snr_db": ch.get("snr_db"), "fading": ch.get("fading"), "n_slots": ch.get("n_slots"),
        "power_control": ch.get("power_control"), "csi_error_var": ch.get("csi_error_var"), "sparsify_frac": ch.get("sparsify_frac"),
        "reset": cfg.get("ablation", {}).get("reset_state_at_window_end", False),
        "local_lr": cfg["fl"]["local"].get("lr"), "local_epochs": cfg["fl"]["local"].get("epochs"),
        "seed": cfg.get("seed"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rows = []
    for sfile in glob.glob(os.path.join(args.root, "*", "*", "summary.json")):
        d = os.path.dirname(sfile)
        with open(os.path.join(d, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        with open(sfile) as f:
            summ = json.load(f)
        r = flatten(cfg); r["run_dir"] = d
        for m in METRICS:
            v = summ.get(m)
            r[m] = (np.nan if v is None else (np.inf if v == "inf" else v))
        rows.append(r)
    if not rows:
        print("no finished runs found"); return
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)
    g = df.groupby(KEYS, dropna=False)
    agg = g[METRICS].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(args.out, "cells.csv"), index=False)
    numbers = {}
    for _, r in agg.iterrows():
        key = "|".join(f"{k}={r[k]}" for k in KEYS)
        numbers[key] = {c: (None if pd.isna(r[c]) else float(r[c])) for c in agg.columns if c not in KEYS}
    with open(os.path.join(args.out, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1)
    print(f"{len(df)} runs -> {len(agg)} cells. Wrote {args.out}/all_runs.csv, cells.csv, numbers.json")
    with pd.option_context("display.max_columns", 30, "display.width", 200):
        print(agg[["exp", "optimizer", "aggregator", "attack", "final_MA_mean", "asr_window_end_mean", "asr_plus_50_mean",
                   "lifespan_50_mean", "final_MA_count"]].to_string())


if __name__ == "__main__":
    main()
