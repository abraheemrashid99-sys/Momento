"""JSON run records.

Every run writes one directory:
    runs/<exp>/<run_id>/
        config.yaml         resolved config
        meta.json           seed, config hash, dataset hashes, git commit, torch version
        rounds.jsonl        one JSON object per round (all metrics, detector stats, timings)
        summary.json        final metrics incl. lifespan
        detector_trace.npz  (optional) per-round CMA statistics for theory checks

Nothing in the paper is typed by hand: scripts/make_tables.py reads these files.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from .utils import config_hash, to_float


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class RunRecord:
    def __init__(self, root: str, exp: str, cfg: Dict[str, Any], run_id: Optional[str] = None):
        self.cfg = cfg
        self.hash = config_hash(cfg)
        self.run_id = run_id or f"{self.hash}_s{cfg.get('seed', 0)}"
        self.dir = os.path.join(root, exp, self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        self.meta = {
            "run_id": self.run_id,
            "config_hash": self.hash,
            "seed": cfg.get("seed", 0),
            "git_commit": _git_commit(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_hashes": {},
        }
        self._rounds_f = open(os.path.join(self.dir, "rounds.jsonl"), "w")
        self._trace: Dict[str, list] = {}

    def add_data_hash(self, name: str, h: str) -> None:
        self.meta["data_hashes"][name] = h
        self._write_meta()

    def _write_meta(self) -> None:
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def log_round(self, rec: Dict[str, Any]) -> None:
        rec = {k: to_float(v) for k, v in rec.items()}
        self._rounds_f.write(json.dumps(rec) + "\n")
        self._rounds_f.flush()

    def trace(self, **arrays: Any) -> None:
        for k, v in arrays.items():
            self._trace.setdefault(k, []).append(np.asarray(to_float(v)))

    def finish(self, summary: Dict[str, Any]) -> None:
        self._rounds_f.close()
        self.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_meta()
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump({k: to_float(v) for k, v in summary.items()}, f, indent=2)
        if self._trace:
            np.savez_compressed(os.path.join(self.dir, "detector_trace.npz"),
                                **{k: np.stack(v) for k, v in self._trace.items()})


def read_rounds(run_dir: str):
    with open(os.path.join(run_dir, "rounds.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def read_summary(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "summary.json")) as f:
        return json.load(f)
