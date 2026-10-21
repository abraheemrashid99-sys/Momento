"""Unit tests that run on CPU in < 1 minute."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from momento.channel import OTAChannel, OTAConfig
from momento.data.partition import dirichlet_partition
from momento.data.triggers import make_trigger
from momento.server.optimizers import FedAvgM, FedAdam
from momento.server.cma import CMA
from momento.server.defenses import geometric_median
from momento.channel.ota import ChannelState

dev = torch.device("cpu")


def test_dirichlet_partition_covers_all():
    y = np.random.RandomState(0).randint(0, 10, 5000)
    parts = dirichlet_partition(y, 20, 0.5, 0)
    allidx = np.concatenate(parts)
    assert len(allidx) == 5000 and len(np.unique(allidx)) == 5000
    assert min(len(p) for p in parts) >= 10


def test_trigger_shapes():
    x = torch.rand(4, 3, 32, 32)
    t = make_trigger("image", "patch", 0, (3, 32, 32))
    assert t.apply(x).shape == x.shape
    iq = torch.randn(4, 2, 128)
    t2 = make_trigger("iq", "iq_add", 1, (2, 128))
    out = t2.apply(iq)
    assert out.shape == iq.shape and not torch.allclose(out, iq)
    txt = torch.randint(2, 100, (4, 64))
    t3 = make_trigger("text", "word", 0, (64,), vocab_size=1000)
    assert (t3.apply(txt)[:, 3:6] == t3.token_ids).all()


def test_ota_ideal_channel_recovers_mean():
    N, d = 8, 50
    ch = OTAChannel(OTAConfig(fading="none", snr_db=60.0), N, d, dev)
    ups = {i: torch.randn(d) for i in range(N)}
    st = ch.transmit(0, ups, torch.ones(N, dtype=torch.bool))
    agg = ch.reconstruct(st, st.group_sums[0])
    ref = torch.stack(list(ups.values())).mean(0)
    assert torch.allclose(agg, ref, atol=1e-2)


def test_ota_attacker_power_cap():
    """An attacker with a bad channel cannot deliver more than physics allows."""
    N, d = 4, 20
    ch = OTAChannel(OTAConfig(fading="rayleigh", snr_db=60.0, p_max=1.0, seed=3), N, d, dev)
    ups = {i: torch.randn(d) * 0.1 for i in range(3)}
    big = torch.ones(d) * 100.0
    st = ch.transmit(0, ups, torch.ones(N, dtype=torch.bool), attacker_override={3: big})
    agg = ch.reconstruct(st, st.group_sums[0])
    # delivered attacker contribution is bounded by sqrt(p_max)*|h| / (gamma*|S|); far below 100/4 per coord
    assert agg.abs().max() < 25.0


def test_fedavgm_inversion_exact():
    d = 30
    opt = FedAvgM(d, dev, eta_s=0.7, beta=0.9)
    shadow = FedAvgM(d, dev, eta_s=0.7, beta=0.9)
    w = torch.randn(d)
    for _ in range(5):
        delta = torch.randn(d)
        w_new = opt.step(w, delta)
        rec = shadow.invert(w, w_new)
        assert torch.allclose(rec, delta, atol=1e-5)
        w = w_new


def test_fedadam_inversion_recovers_sign_and_cold_coordinates():
    """Exact magnitude recovery is impossible in Adam's saturated regime; what the attack needs is
    (i) the sign of the aggregate and (ii) the ranking of the second-moment estimate (cold coords)."""
    d = 200; torch.manual_seed(0)
    std = torch.logspace(-2, 0, d)[torch.randperm(d)]
    opt = FedAdam(d, dev, eta_s=0.05); shadow = FedAdam(d, dev, eta_s=0.05)
    w = torch.randn(d)
    for t in range(40):
        delta = torch.randn(d) * std
        w_new = opt.step(w, delta)
        rec = shadow.invert(w, w_new, fallback_mag=0.8 * std)
        w = w_new
    assert (torch.sign(rec) == torch.sign(delta)).float().mean() > 0.95
    k = 40
    cold_true = set(opt.v.topk(k, largest=False).indices.tolist())
    cold_hat = set(shadow.v.topk(k, largest=False).indices.tolist())
    assert len(cold_true & cold_hat) / k > 0.8


def test_momentum_amplification_formula():
    """Theorem B sanity: a single injection a at round 0 under FedAvgM yields displacement eta*a*(1-beta^{t+1})/(1-beta)."""
    d, beta, eta = 10, 0.9, 1.0
    opt = FedAvgM(d, dev, eta_s=eta, beta=beta)
    w0 = torch.zeros(d); a = torch.ones(d); w = w0
    for t in range(30):
        w = opt.step(w, a if t == 0 else torch.zeros(d))
    expected = eta * (1 - beta ** 30) / (1 - beta)
    assert abs(float(-w[0]) - expected) < 1e-4


def _state(group_sums, sigma_z2=1e-6, n=10):
    d = group_sums[0].numel()
    return ChannelState(0, torch.ones(n), torch.ones(n), torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.long),
                        torch.ones(1), torch.arange(d), sigma_z2, group_sums)


def test_cma_detects_burst_after_calibration():
    d = 200; torch.manual_seed(0)
    cma = CMA(d, dev, K=4, warmup_rounds=30, alpha_fa=0.05)
    mean = torch.randn(d) * 0.1
    fired = []
    for t in range(60):
        delta = mean + 0.05 * torch.randn(d)
        if 45 <= t < 48:
            delta = delta + 2.0 * torch.randn(d)        # burst
        out, diag = cma.aggregate(_state([delta]), {})
        fired.append((t, diag["fired"]))
    assert cma.lam is not None
    assert any(f for t, f in fired if 45 <= t < 50), "burst not detected"
    fp = sum(f for t, f in fired if 30 <= t < 45)
    assert fp <= 3


def test_geometric_median_resists_outlier():
    X = torch.cat([torch.zeros(9, 5), torch.ones(1, 5) * 100])
    gm = geometric_median(X)
    assert gm.abs().max() < 1.0
