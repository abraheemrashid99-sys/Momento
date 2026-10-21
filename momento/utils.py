"""Small shared utilities: seeding, device selection, flat-vector helpers, config loading."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic where it does not cost much. cudnn.benchmark off for reproducibility.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# Flat parameter vectors. Everything that crosses the "air" is a flat float32
# vector so that channel code is model-agnostic.
# ---------------------------------------------------------------------------

def params_to_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def vector_to_params(model: torch.nn.Module, vec: torch.Tensor) -> None:
    ptr = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[ptr:ptr + n].view_as(p))
            ptr += n
    assert ptr == vec.numel(), "vector/model size mismatch"


def buffers_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """BatchNorm running stats are not transmitted over the air in our protocol;
    the server keeps them from a benign warm-up and clients receive them with the broadcast."""
    return {k: v.detach().clone() for k, v in model.named_buffers()}


def load_buffers(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for k, v in model.named_buffers():
            if k in state:
                v.copy_(state[k])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load a YAML config. Supports `_base_: other.yaml` inheritance (relative to the file)
    and dotted CLI overrides like `attack.window=5` (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("_base_"))
        cfg = deep_update(load_config(base_path), cfg)
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_float(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
