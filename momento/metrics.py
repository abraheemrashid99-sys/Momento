"""Evaluation metrics.

MA   : main-task accuracy on the clean test set (reported at the SAME round as ASR, never best-round).
ASR  : attack success rate on triggered test samples whose TRUE label != target.
lifespan: rounds after the attack window until ASR < 50% (A3FL / Lachnit-Karame 50%-lifespan),
          plus the area under the post-attack ASR curve (AUC-ASR).
detection: per-round TPR/FPR of the CMA detector given the known attack window.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .utils import vector_to_params


@torch.no_grad()
def evaluate(model, w, x, y, trigger, device, batch: int = 512) -> Dict[str, float]:
    vector_to_params(model, w); model.eval()
    correct = 0; n = 0
    asr_hit = 0; asr_n = 0
    for i in range(0, x.shape[0], batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = model(xb).argmax(1)
        correct += int((pred == yb).sum()); n += yb.numel()
        if trigger is not None:
            keep = yb != trigger.target
            if keep.any():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    pt = model(trigger.apply(xb[keep])).argmax(1)
                asr_hit += int((pt == trigger.target).sum()); asr_n += int(keep.sum())
    out = {"MA": correct / max(n, 1)}
    if trigger is not None:
        out["ASR"] = asr_hit / max(asr_n, 1)
    return out


def lifespan(rounds: List[Dict], t_end: int, thresh: float = 0.5) -> Dict[str, float]:
    """rounds: list of per-round records with 'round' and 'ASR'. t_end: first round after the attack window."""
    post = [(r["round"], r["ASR"]) for r in rounds if "ASR" in r and r["round"] >= t_end]
    if not post:
        return {"lifespan_50": float("nan"), "auc_asr_post": float("nan"), "asr_at_end": float("nan")}
    asr_end = post[0][1]
    life = float("inf")
    for t, a in post:
        if a < thresh:
            life = t - t_end; break
    auc = float(np.trapezoid([a for _, a in post], [t for t, _ in post])) if len(post) > 1 else 0.0
    return {"lifespan_50": life, "auc_asr_post": auc, "asr_at_end": asr_end,
            "asr_plus_20": next((a for t, a in post if t >= t_end + 20), float("nan")),
            "asr_plus_50": next((a for t, a in post if t >= t_end + 50), float("nan")),
            "asr_plus_100": next((a for t, a in post if t >= t_end + 100), float("nan"))}


def detection_stats(rounds: List[Dict], t0: int, tau: int) -> Dict[str, float]:
    fired = [(r["round"], bool(r.get("det_fired", False))) for r in rounds if "det_fired" in r and not r.get("det_warmup", False)]
    if not fired:
        return {}
    tp = sum(f for t, f in fired if t0 <= t < t0 + tau)
    fp = sum(f for t, f in fired if not (t0 <= t < t0 + tau + 2))
    n_pos = sum(1 for t, _ in fired if t0 <= t < t0 + tau)
    n_neg = sum(1 for t, _ in fired if not (t0 <= t < t0 + tau + 2))
    return {"det_tpr": tp / max(n_pos, 1), "det_fpr": fp / max(n_neg, 1), "det_n_pos": n_pos, "det_n_neg": n_neg}
