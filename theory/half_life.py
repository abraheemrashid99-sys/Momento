#!/usr/bin/env python3
"""Theorem B (poison half-life) — analytical bound and comparison with measured runs.

Analytical part (no data needed):
  FedAvgM (Reddi form m <- beta m + delta, w <- w - eta m): a single aggregate injection a at round s gives
      disp(t) = eta * a * (1 - beta^{t-s+1}) / (1 - beta)        (grows to eta a /(1-beta))
  the *buffer* content decays as beta^{t-s}; half-life of the buffer: ln 2 / ln(1/beta)
      beta=0.9 -> 6.58 rounds ; beta=0.99 -> 68.97 rounds
  FedAdam second moment: v decays as beta2^{t-s}: beta2=0.99 -> 68.97 rounds. While v is inflated on a
  coordinate by factor F relative to its benign level, benign corrective steps on that coordinate are
  damped by 1/sqrt(F).
  CMA: the retained injected energy after purge is (1 - kappa p_det) * sum_k omega_k beta_k^{t-s}.

Measured part (optional): --root runs  reads echo runs and overlays ASR(t) after the window, and prints
the measured 50%-lifespan next to the analytical buffer half-life.

    python theory/half_life.py --plot figs/half_life.pdf [--root runs]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def buffer_half_life(beta: float) -> float:
    return np.log(2) / np.log(1 / beta) if 0 < beta < 1 else float("inf")


def fedavgm_displacement(eta, a, beta, T):
    t = np.arange(T)
    return eta * a * (1 - beta ** (t + 1)) / (1 - beta)


def cma_retained(T, K=4, kappa=1.0, p_det=0.9):
    betas = np.array([1 - 2.0 ** (-k) for k in range(1, K + 1)])
    w = np.array([2.0 ** (-k) for k in range(1, K + 1)]); w = w / w.sum()
    t = np.arange(T)
    return (1 - kappa * p_det) * sum(wk * bk ** t for wk, bk in zip(w, betas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    for b in [0.5, 0.9, 0.95, 0.99]:
        print(f"beta={b:5.2f}  buffer half-life = {buffer_half_life(b):6.2f} rounds   amplification 1/(1-beta) = {1/(1-b):6.1f}")
    print("CMA K=4 retained energy at t=10:", cma_retained(11)[-1], " at t=50:", cma_retained(51)[-1])
    if args.root:
        for sfile in glob.glob(os.path.join(args.root, "*", "*", "summary.json")):
            d = os.path.dirname(sfile)
            import yaml
            cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
            if cfg["attack"].get("name", "none") not in ("echo", "echo_adam", "scale"):
                continue
            s = json.load(open(sfile))
            ok = cfg["server"].get("opt_kw", {})
            beta = ok.get("beta", ok.get("beta1", 0))
            print(f"{cfg['attack']['name']:10s} {cfg['server']['optimizer']:8s} {cfg['server']['aggregator']:10s} beta={beta} "
                  f"measured lifespan_50={s.get('lifespan_50')}  analytic buffer t1/2={buffer_half_life(beta):.1f}")
    if args.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        T = 120; t = np.arange(T)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for b in [0.5, 0.9, 0.99]:
            ax[0].semilogy(t, b ** t, label=f"beta={b}")
        ax[0].axhline(0.01, ls="--", c="gray"); ax[0].set_xlabel("rounds after injection"); ax[0].set_ylabel("buffer weight"); ax[0].legend(); ax[0].set_title("single momentum memory")
        ax[1].plot(t, 0.9 ** t, label="FedAvgM beta=0.9")
        ax[1].plot(t, cma_retained(T, p_det=0.0), label="CMA K=4, detector off")
        ax[1].plot(t, cma_retained(T, p_det=0.9), label="CMA K=4, p_det=0.9")
        ax[1].set_xlabel("rounds after window"); ax[1].set_ylabel("retained injected energy (bound)"); ax[1].legend(); ax[1].set_title("Theorem B, first term")
        fig.tight_layout(); os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True); fig.savefig(args.plot)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
