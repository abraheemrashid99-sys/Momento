"""Non-IID partitions.

dirichlet_partition : label-Dirichlet(alpha) over clients (Hsu et al. 2019), alpha in {0.1, 0.5, 1.0}.
natural_partition   : LEAF writers (FEMNIST) grouped into the requested number of clients.
snr_band_partition  : RadioML: clients get Dirichlet label skew AND a contiguous SNR band, so that data
                      heterogeneity and channel heterogeneity are correlated (a reviewer pitfall otherwise).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int, min_size: int = 10) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    for _ in range(100):
        idx_per_client = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.where(y == c)[0]; rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(n_clients))
            # balance: do not give to clients that already have more than average
            p = np.array([pi * (len(ic) < len(y) / n_clients) for pi, ic in zip(p, idx_per_client)])
            p = p / p.sum() if p.sum() > 0 else np.ones(n_clients) / n_clients
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, part in enumerate(np.split(idx, cuts)):
                idx_per_client[i].extend(part.tolist())
        sizes = [len(ic) for ic in idx_per_client]
        if min(sizes) >= min_size:
            break
    return [np.array(sorted(ic)) for ic in idx_per_client]


def natural_partition(writer_ids: np.ndarray, n_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    writers = np.unique(writer_ids); rng.shuffle(writers)
    groups = np.array_split(writers, n_clients)
    return [np.where(np.isin(writer_ids, g))[0] for g in groups]


def snr_band_partition(y: np.ndarray, snr: np.ndarray, n_clients: int, alpha: float, seed: int,
                       n_bands: Optional[int] = None) -> List[np.ndarray]:
    """Clients are assigned to SNR bands (contiguous ranges of the dataset's SNR grid); within a band,
    samples are split by label-Dirichlet. Returns per-client index arrays."""
    rng = np.random.RandomState(seed)
    snr_vals = np.unique(snr)
    n_bands = n_bands or min(5, len(snr_vals))
    bands = np.array_split(snr_vals, n_bands)
    clients_per_band = np.array_split(np.arange(n_clients), n_bands)
    out: List[np.ndarray] = [None] * n_clients  # type: ignore
    for band, cl in zip(bands, clients_per_band):
        mask = np.isin(snr, band); idx = np.where(mask)[0]
        parts = dirichlet_partition(y[idx], len(cl), alpha, seed + int(band[0]) + 1000)
        for c, p in zip(cl, parts):
            out[c] = idx[p]
    return out
