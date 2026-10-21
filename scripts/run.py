#!/usr/bin/env python3
"""Run one configuration.

    python scripts/run.py configs/E1_hook/cifar_fedavgm.yaml attack.name=echo seed=1
    python scripts/run.py configs/smoke/cpu_smoke.yaml --quiet

Dotted overrides are parsed as YAML. The run directory is runs/<exp>/<config_hash>_s<seed>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from momento.records import RunRecord
from momento.simulator import Simulator
from momento.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--exp", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    root = args.root or cfg.get("run", {}).get("root", "runs")
    exp = args.exp or cfg.get("run", {}).get("exp", "default")
    rec = RunRecord(root, exp, cfg)
    print(f"[run] {rec.dir}")
    sim = Simulator(cfg, record=rec, verbose=not args.quiet)
    summary = sim.run()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("malicious",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
