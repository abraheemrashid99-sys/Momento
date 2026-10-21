#!/usr/bin/env python3
"""Cross-product sweeps. Writes a job list (one command per line) that can be executed sequentially,
with GNU parallel, or with `--run` (sequential, skipping finished runs).

Examples
  # E1: every attack x every defense on the FedAvgM server, 5 seeds
  python scripts/sweep.py E1 --seeds 0 1 2 3 4 > jobs_E1.txt
  python scripts/sweep.py E1 --seeds 0 1 2 --run

  # E2: CMA vs all attacks + ablations
  python scripts/sweep.py E2 --seeds 0 1 2 --run
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from momento.utils import load_config, config_hash

ATTACKS = ["none", "scale", "neurotoxin", "a3fl_lite", "chameleon_lite", "echo"]
E1_DEFENSES = ["cifar_fedavgm", "cifar_fedavg", "def_bev", "def_rotaf", "def_gm", "def_gcc", "def_gtm", "def_fedsac", "def_tti",
               "digital_cc", "digital_tm", "reset_ablation"]
E1_ADAM = ["cifar_fedadam", "reset_ablation_adam"]


def jobs(exp: str, seeds, quick: bool):
    J = []
    q = ["fl.rounds=60", "attack.start=25", "eval.dense_after=30", "dataset.subset=4000", "model.width=16"] if quick else []
    if exp == "E1":
        for d, a, s in itertools.product(E1_DEFENSES, ATTACKS, seeds):
            J.append(["configs/E1_hook/" + d + ".yaml", f"attack.name={a}", f"seed={s}"] + q)
        for d, a, s in itertools.product(E1_ADAM, ["none", "scale", "neurotoxin", "echo_adam"], seeds):
            J.append(["configs/E1_hook/" + d + ".yaml", f"attack.name={a}", f"seed={s}"] + q)
    elif exp == "E2":
        for a, s in itertools.product(ATTACKS, seeds):
            J.append(["configs/E2_cma/cifar_cma.yaml", f"attack.name={a}", f"seed={s}"] + q)
        for a, s in itertools.product(["none", "echo_adam"], seeds):
            J.append(["configs/E2_cma/cifar_cma_adaptive.yaml", f"attack.name={a}", f"seed={s}"] + q)
        for abl, s in itertools.product(["abl_K1", "abl_K2", "abl_K3", "abl_K6", "abl_nopurge", "abl_noorth", "abl_uniform", "abl_quantile", "abl_single_ema"], seeds):
            J.append([f"configs/E2_cma/{abl}.yaml", "attack.name=echo", f"seed={s}"] + q)
    elif exp == "E3":
        files = sorted(os.listdir("configs/E3_sweeps"))
        for f, a, s in itertools.product(files, ["scale", "echo"], seeds):
            J.append([f"configs/E3_sweeps/{f}", f"attack.name={a}", f"seed={s}"] + q)
    elif exp == "E4":
        files = sorted(os.listdir("configs/E4_channel"))
        for f, a, s in itertools.product(files, ["none", "echo"], seeds):
            J.append([f"configs/E4_channel/{f}", f"attack.name={a}", f"seed={s}"] + q)
            base = f.replace(".yaml", "")
            J.append([f"configs/E4_channel/{f}", f"attack.name={a}", f"seed={s}", "server.optimizer=fedavgm",
                      "server.aggregator=none", "server.opt_kw.beta=0.9", "server.agg_kw={}"] + q)
    elif exp == "E5":
        for ds in ["radioml", "femnist", "agnews"]:
            for a, s in itertools.product(["none", "scale", "neurotoxin", "echo"], seeds):
                J.append([f"configs/E5_modalities/{ds}_fedavgm.yaml", f"attack.name={a}", f"seed={s}"] + q)
                J.append([f"configs/E5_modalities/{ds}_cma.yaml", f"attack.name={a}", f"seed={s}"] + q)
    elif exp == "E6":
        for f, s in itertools.product(sorted(os.listdir("configs/E6_adaptive")), seeds):
            J.append([f"configs/E6_adaptive/{f}", f"seed={s}"] + q)
    elif exp == "E7":
        for f, s in itertools.product(sorted(os.listdir("configs/E7_theory")), seeds):
            J.append([f"configs/E7_theory/{f}", f"seed={s}"] + q)
    else:
        raise SystemExit(f"unknown exp {exp}")
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--quick", action="store_true", help="short runs for debugging")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()
    J = jobs(args.exp, args.seeds, args.quick)
    for j in J:
        cmd = [sys.executable, "scripts/run.py"] + j + ["--quiet", "--root", args.root]
        if args.run:
            cfg = load_config(j[0], j[1:])
            exp = cfg.get("run", {}).get("exp", "default")
            run_dir = os.path.join(args.root, exp, f"{config_hash(cfg)}_s{cfg.get('seed', 0)}")
            if os.path.exists(os.path.join(run_dir, "summary.json")):
                print(f"[skip] {run_dir}"); continue
            print("[run]", " ".join(j)); subprocess.run(cmd, check=False)
        else:
            print(" ".join(cmd))


if __name__ == "__main__":
    main()
