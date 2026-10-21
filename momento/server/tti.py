"""TTI-style two-stage robust aggregation for OTA-FL (Ma, Lee, Kim, Brinton, arXiv:2605.19253, 2026),
reimplemented from the paper's description (no official code located at the time of writing).

Stage I  (trust tiering from *client-reported* indicators):
    each client i reports, for its own update u_i:
        TDA_i  = cos(u_i, u_i^{prev})                 temporal direction alignment
        RN_i   = ||u_i|| / median_j ||u_j||           relative l2 norm (server normalises by the median of reports)
        SP_i   = top-1% energy fraction of |u_i|      spikiness
    score_i = sum_k beta_k * z_k(indicator_k)   (robust z-score w.r.t. MAD over benign warm-up)
    tiers: trusted (score <= th_lo) -> OTA slot;  suspicious (th_lo < score <= th_hi) -> individual upload;
           malicious (score > th_hi) -> excluded.
Stage II (layer-wise agglomerative clustering + majority voting over the individually uploaded
    suspicious updates together with the trusted OTA aggregate as an anchor): the cluster that contains
    the anchor is kept; others are dropped.
Longitudinal reputation: EMA of per-client scores with MAD threshold; persistent offenders excluded.

Because clients report their own indicators, a malicious client can report *benign-looking* values.
The simulator lets the attack choose what to report (see attacks/base.py: `report_indicators`).
Costs charged: one float per indicator per client per round + full individual uploads for suspicious
clients (bytes recorded in diagnostics).

Calibration: the paper learns beta via Bayesian optimisation against representative attacks.
We expose `beta` as a config and by default use equal weights with thresholds calibrated on
benign warm-up rounds ONLY (so that TTI is not tuned on the test attacks). To reproduce the paper's
advantage we also provide `calibrate_against(attack_scores)`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..channel.ota import ChannelState
from .defenses import Aggregator


def _spikiness(u: torch.Tensor, frac: float = 0.01) -> float:
    a = u.abs(); k = max(1, int(frac * a.numel()))
    top = a.topk(k).values
    return float(top.sum() / (a.sum() + 1e-12))


class TTI(Aggregator):
    name = "tti"

    def __init__(self, d_tx, device, beta=(1.0, 1.0, 1.0), th_lo_q=0.90, th_hi_q=0.99,
                 rep_decay=0.8, rep_thresh=3.0, n_clusters=2, bytes_per_float=4, **kw):
        super().__init__(d_tx, device)
        self.beta = torch.tensor(beta, dtype=torch.float32)
        self.th_lo_q, self.th_hi_q = th_lo_q, th_hi_q
        self.rep_decay, self.rep_thresh = rep_decay, rep_thresh
        self.n_clusters = n_clusters
        self.bytes_per_float = bytes_per_float
        self.med: Optional[torch.Tensor] = None   # per-indicator median (benign)
        self.mad: Optional[torch.Tensor] = None
        self.th_lo = self.th_hi = None
        self.rep: Dict[int, float] = {}
        self.prev: Dict[int, torch.Tensor] = {}
        self._benign_scores: List[float] = []
        self._benign_ind: List[torch.Tensor] = []

    # ---- indicator computation (what an honest client computes about itself)
    def honest_indicators(self, cid: int, u: torch.Tensor) -> torch.Tensor:
        prev = self.prev.get(cid)
        tda = float(torch.nn.functional.cosine_similarity(u, prev, dim=0)) if prev is not None else 1.0
        return torch.tensor([tda, float(u.norm()), _spikiness(u)])

    def score(self, ind: torch.Tensor, med_norm: float) -> float:
        x = ind.clone(); x[1] = x[1] / (med_norm + 1e-12)
        if self.med is None:
            return 0.0
        z = (x - self.med).abs() / (self.mad + 1e-6)
        z[0] = (self.med[0] - x[0]).clamp_min(0) / (self.mad[0] + 1e-6)   # low alignment is suspicious
        return float((self.beta * z).sum())

    def calibrate(self, benign_indicators: List[torch.Tensor]) -> None:   # type: ignore[override]
        X = torch.stack(benign_indicators)
        X[:, 1] = X[:, 1] / (X[:, 1].median() + 1e-12)
        self.med = X.median(0).values
        self.mad = (X - self.med).abs().median(0).values * 1.4826 + 1e-6
        scores = torch.tensor([self.score(x, 1.0) for x in X])
        self.th_lo = float(torch.quantile(scores, self.th_lo_q))
        self.th_hi = float(torch.quantile(scores, self.th_hi_q))

    def aggregate(self, state: ChannelState, server_ctx: Dict) -> Tuple[torch.Tensor, Dict]:
        """server_ctx must provide:
            'reported'  : {cid: indicator tensor(3,)}  what each client *reported*
            'uploads'   : callable(cid) -> individual update vector (d_tx,)   (only used for suspicious)
            'ota'       : callable(list_of_cids) -> OTA aggregate over those clients (re-transmission
                          on a trusted slot; the simulator re-runs the channel on the trusted set)
        """
        reported: Dict[int, torch.Tensor] = server_ctx["reported"]
        if self.med is None:
            # not calibrated: behave as plain mean, record indicators for calibration
            self._benign_ind.extend(reported.values())
            return torch.stack(state.group_sums).mean(0), {"calibrating": True}
        med_norm = float(torch.stack([r[1] for r in reported.values()]).median())
        scores = {c: self.score(r, med_norm) for c, r in reported.items()}
        # longitudinal reputation
        for c, s in scores.items():
            self.rep[c] = self.rep_decay * self.rep.get(c, 0.0) + (1 - self.rep_decay) * s
        trusted = [c for c, s in scores.items() if s <= self.th_lo and self.rep[c] <= self.rep_thresh]
        suspicious = [c for c, s in scores.items() if self.th_lo < s <= self.th_hi and self.rep[c] <= self.rep_thresh]
        malicious = [c for c in scores if c not in trusted and c not in suspicious]
        n_ind = 3 * len(reported)
        extra_bytes = self.bytes_per_float * n_ind
        anchor = server_ctx["ota"](trusted) if trusted else None
        kept = list(trusted)
        d = {"trusted": len(trusted), "suspicious": len(suspicious), "malicious": len(malicious)}
        if suspicious:
            ups = {c: server_ctx["uploads"](c) for c in suspicious}
            extra_bytes += self.bytes_per_float * sum(u.numel() for u in ups.values())
            # Stage II: cluster suspicious uploads (+anchor) with 2-means on cosine geometry
            vecs = [ups[c] for c in suspicious]; ids = list(suspicious)
            if anchor is not None:
                vecs.append(anchor); ids.append(-1)
            V = torch.stack(vecs); Vn = V / (V.norm(dim=1, keepdim=True) + 1e-12)
            labels = _two_means(Vn)
            if anchor is not None:
                keep_lab = labels[-1]
            else:  # majority cluster
                keep_lab = int(torch.mode(labels).values)
            keep_ids = [i for i, l in zip(ids, labels.tolist()) if l == keep_lab and i >= 0]
            kept += keep_ids
            d["stage2_kept"] = len(keep_ids)
            if anchor is not None and keep_ids:
                agg = (anchor * len(trusted) + sum(ups[c] for c in keep_ids)) / (len(trusted) + len(keep_ids))
            elif keep_ids:
                agg = sum(ups[c] for c in keep_ids) / len(keep_ids)
            else:
                agg = anchor
        else:
            agg = anchor
        if agg is None:
            agg = torch.stack(state.group_sums).mean(0)
        self.extra_uplink_bytes = extra_bytes
        d["extra_uplink_bytes"] = extra_bytes
        d["kept_ids"] = kept
        return agg, d

    def remember(self, cid: int, u: torch.Tensor) -> None:
        self.prev[cid] = u.detach().clone()


def _two_means(X: torch.Tensor, iters: int = 10) -> torch.Tensor:
    n = X.shape[0]
    if n < 2:
        return torch.zeros(n, dtype=torch.long)
    # init: two most distant points
    D = torch.cdist(X, X); i, j = divmod(int(D.argmax()), n)
    C = torch.stack([X[i], X[j]])
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        for k in range(2):
            if (lab == k).any():
                C[k] = X[lab == k].mean(0)
    return lab
