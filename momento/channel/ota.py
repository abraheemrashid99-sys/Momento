r"""Over-the-air computation (AirComp) uplink.

Model (per round t, per AirComp slot g):

    y_g = sum_{i in S_g} h_i p_i x_i + z,        z ~ CN(0, sigma_z^2 I)  (we work with the real part
                                                 after coherent combining; sigma_z^2 is the real noise var)

    x_i   : client i's (sparsified, compressed) update vector, real, length d'
    h_i   : block-fading channel of client i for this round (Rayleigh CN(0,1) or Rician)
    p_i   : pre-equalisation scalar chosen by client i from its CSI estimate \hat h_i

Power budget: P_max is the per-channel-use (per transmitted coordinate) transmit power, so a client's
vector must satisfy ||p_i x_i||^2 <= P_max * d'. SNR (config) = P_max E|h|^2 / sigma_z^2 is therefore the
per-coordinate receive SNR of a full-power client, the usual OTA-FL convention.

Power control policies (what the *honest* protocol prescribes):
    'inversion'  truncated channel inversion:  p_i = gamma / \hat h_i  if |\hat h_i|^2 >= g_th else 0 (client silent)
                 with gamma chosen so that |p_i|^2 ||x_i||^2 <= P_max d' for all participating clients
                 (Amiri & Gunduz 2020; Sery et al. 2021)
    'bev'        best-effort voting: p_i = sqrt(P_max)/||x_i|| * conj(h)/|h| (full power, normalised update)
                 (Fan et al. 2022). The server divides by the sum of |h_i| it can estimate from pilots.

The server receives only y_g per slot. It knows: gamma, |S_g| (from scheduling), sigma_z^2, the slot
assignment it chose, and nothing per-client. It forms  bar_x_g = Re(y_g) / (gamma * |S_g|).

Attacker physics: an attacker is a client. It has its own fading h_a and CSI error, the same P_max,
and it can choose p_a freely (it is not bound to the protocol). Its *effective* contribution at the
server is h_a p_a x_a: so an attacker that wants to deliver a vector v at the server must spend power
||v||^2 / |h_a|^2, which caps what it can inject when its channel is poor. This is modelled exactly;
no "in-phase omnipotent adversary".

Sparsification: top-k with error feedback (EF) per client, as in Amiri & Gunduz; the server
reconstructs with the shared random projection / index set. We implement the simple variant where
all clients agree on a per-round random coordinate subset of size d' (seeded by the round), so
superposition is index-aligned. The attacker knows the subset (it is broadcast).

Misalignment/CSI error: \hat h_i = h_i + e_i, e_i ~ CN(0, sigma_h^2). With inversion the residual
error is (h_i/\hat h_i - 1) gamma x_i and is tracked explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class OTAConfig:
    fading: str = "rayleigh"        # 'rayleigh' | 'rician' | 'none' (ideal, for digital-FL upper bounds)
    rician_k_db: float = 6.0
    snr_db: float = 10.0            # per-client average receive SNR at full power, defines sigma_z^2
    power_control: str = "inversion"  # 'inversion' | 'bev'
    truncation_db: float = -10.0    # g_th relative to E|h|^2 = 1
    csi_error_var: float = 0.0      # sigma_h^2
    sparsify_frac: float = 1.0      # fraction of coordinates transmitted per round (top-k over the subset)
    error_feedback: bool = True
    n_slots: int = 1                # number of AirComp slots (groups) per round, >1 for ROTAF/FedSAC
    slot_assignment: str = "random" # 'random' (re-drawn each round) | 'fixed'
    p_max: float = 1.0              # transmit power cap (linear). SNR is defined relative to p_max.
    seed: int = 0


@dataclass
class ChannelState:
    """Everything the server legitimately knows after a round plus the ground truth for analysis."""
    round: int
    h: torch.Tensor                       # (N,) complex true fading
    h_hat: torch.Tensor                   # (N,) complex CSI estimate
    participating: torch.Tensor           # (N,) bool: channel above truncation and selected
    slot_of: torch.Tensor                 # (N,) long: slot index or -1
    gamma: torch.Tensor                   # (n_slots,) scaling used by the server
    coord_idx: torch.Tensor               # (d',) long: transmitted coordinate subset
    sigma_z2: float
    group_sums: List[torch.Tensor] = field(default_factory=list)   # what the server sees per slot (length d')
    misalignment_energy: float = 0.0
    noise_energy: float = 0.0
    attacker_clip: Dict[int, float] = field(default_factory=dict)   # power-cap clip factor per attacker (1 = unclipped)


class OTAChannel:
    def __init__(self, cfg: OTAConfig, n_clients: int, d: int, device: torch.device):
        self.cfg = cfg
        self.N = n_clients
        self.d = d
        self.device = device
        self.rng = np.random.RandomState(cfg.seed + 101)
        self.d_tx = max(1, int(round(cfg.sparsify_frac * d)))
        # noise variance: SNR defined as p_max * E|h|^2 / sigma_z^2 per coordinate with unit-energy symbols
        self.sigma_z2 = cfg.p_max / (10 ** (cfg.snr_db / 10))
        self.ef_residual = torch.zeros(n_clients, d, device=device) if cfg.error_feedback and cfg.sparsify_frac < 1 else None
        # power budget is per channel use (per transmitted coordinate): ||p_i x_i||^2 <= p_max * d_tx
        self.amp_budget = float(np.sqrt(cfg.p_max * self.d_tx))
        self._fixed_slots = None
        self.g_th = 10 ** (cfg.truncation_db / 10)

    # --------------------------------------------------------------------- fading
    def draw_fading(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rs = np.random.RandomState(self.cfg.seed * 7919 + t)
        if self.cfg.fading == "none":
            h = np.ones(self.N, dtype=np.complex128)
        else:
            w = (rs.randn(self.N) + 1j * rs.randn(self.N)) / np.sqrt(2)
            if self.cfg.fading == "rayleigh":
                h = w
            elif self.cfg.fading == "rician":
                k = 10 ** (self.cfg.rician_k_db / 10)
                h = np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * w
            else:
                raise ValueError(self.cfg.fading)
        e = np.sqrt(self.cfg.csi_error_var / 2) * (rs.randn(self.N) + 1j * rs.randn(self.N))
        return torch.from_numpy(h).to(self.device), torch.from_numpy(h + e).to(self.device)

    def draw_coords(self, t: int) -> torch.Tensor:
        if self.d_tx >= self.d:
            return torch.arange(self.d, device=self.device)
        rs = np.random.RandomState(self.cfg.seed * 104729 + t)
        return torch.from_numpy(rs.choice(self.d, self.d_tx, replace=False)).to(self.device)

    def assign_slots(self, t: int, selected: torch.Tensor) -> torch.Tensor:
        slot_of = torch.full((self.N,), -1, dtype=torch.long, device=self.device)
        sel = torch.where(selected)[0].cpu().numpy()
        if self.cfg.n_slots == 1:
            slot_of[sel] = 0
            return slot_of
        if self.cfg.slot_assignment == "fixed":
            if self._fixed_slots is None:
                self._fixed_slots = np.arange(self.N) % self.cfg.n_slots
            slot_of[sel] = torch.from_numpy(self._fixed_slots[sel]).to(self.device)
        else:
            rs = np.random.RandomState(self.cfg.seed * 15485863 + t)
            perm = rs.permutation(len(sel))
            slots = np.empty(len(sel), dtype=np.int64); slots[perm] = np.arange(len(sel)) % self.cfg.n_slots
            slot_of[sel] = torch.from_numpy(slots).to(self.device)
        return slot_of

    # --------------------------------------------------------------------- transmission
    def transmit(self, t: int, updates: Dict[int, torch.Tensor], selected: torch.Tensor,
                 attacker_override: Optional[Dict[int, torch.Tensor]] = None) -> ChannelState:
        """updates: {client_id: full-length update vector (d,)} for selected clients.
        attacker_override: {client_id: vector v (d_tx,)} that the attacker wants to *deliver at the
        server* on the transmitted coordinates; the channel computes the required transmit power and
        clips to p_max (the attacker cannot beat physics). Honest clients follow the protocol.
        Returns ChannelState with group_sums already scaled to 'average update' units."""
        cfg = self.cfg
        h, h_hat = self.draw_fading(t)
        coord = self.draw_coords(t)
        attacker_override = attacker_override or {}

        # truncation: client silent if |h_hat|^2 < g_th (only honest clients obey; attackers may still try)
        gain = (h_hat.abs() ** 2)
        participating = selected.clone()
        for i in torch.where(selected)[0].tolist():
            if i not in attacker_override and cfg.power_control == "inversion" and gain[i] < self.g_th:
                participating[i] = False
        slot_of = self.assign_slots(t, participating)

        # sparsify honest updates (+EF)
        x_tx: Dict[int, torch.Tensor] = {}
        for i in torch.where(participating)[0].tolist():
            if i in attacker_override:
                continue
            u = updates[i]
            if self.ef_residual is not None:
                u = u + self.ef_residual[i]
            xi = u[coord]
            if self.ef_residual is not None:
                full = torch.zeros_like(u); full[coord] = xi
                self.ef_residual[i] = u - full
            x_tx[i] = xi

        n_slots = cfg.n_slots
        gamma = torch.zeros(n_slots, device=self.device, dtype=torch.float64)
        group_sums: List[torch.Tensor] = []
        mis_e = 0.0; noise_e = 0.0; att_clip: Dict[int, float] = {}
        rs = np.random.RandomState(cfg.seed * 32452843 + t)
        for g in range(n_slots):
            members = [i for i in torch.where(slot_of == g)[0].tolist()]
            honest = [i for i in members if i not in attacker_override]
            attackers = [i for i in members if i in attacker_override]
            if len(members) == 0:
                group_sums.append(torch.zeros(len(coord), device=self.device)); gamma[g] = 1.0
                continue
            # ---- honest power control
            if cfg.power_control == "inversion":
                # gamma_g = min_i sqrt(P_max) |h_hat_i| / ||x_i||   (so that every honest client meets P_max)
                if honest:
                    norms = torch.stack([x_tx[i].norm() for i in honest]).double() + 1e-12
                    gh = torch.stack([h_hat[i].abs() for i in honest]).double()
                    gamma[g] = (self.amp_budget * gh / norms).min()
                else:
                    gamma[g] = 1.0
                y = torch.zeros(len(coord), device=self.device, dtype=torch.float64)
                for i in honest:
                    p = gamma[g] / h_hat[i]                       # complex scalar
                    eff = (h[i] * p).real                         # = gamma if CSI perfect
                    y += eff * x_tx[i].double()
                    mis_e += float(((eff - gamma[g]) * x_tx[i].double()).pow(2).sum())
                scale = gamma[g] * max(len(members), 1)
            elif cfg.power_control == "bev":
                # every client sends normalised update at full power with phase pre-compensation.
                y = torch.zeros(len(coord), device=self.device, dtype=torch.float64)
                hsum = 0.0
                for i in honest:
                    xi = x_tx[i].double(); xn = xi / (xi.norm() + 1e-12)
                    p = self.amp_budget * (h_hat[i].conj() / h_hat[i].abs())
                    eff = (h[i] * p).real
                    y += eff * xn; hsum += float(self.amp_budget * h_hat[i].abs())
                for i in attackers:
                    hsum += float(self.amp_budget * h_hat[i].abs())
                gamma[g] = hsum / max(len(members), 1)
                scale = gamma[g] * max(len(members), 1)
            else:
                raise ValueError(cfg.power_control)
            # ---- attackers: want to deliver v at the server (in 'scaled' units so that the server's
            #      division by scale yields v/|members| contribution to the group average), subject to p_max.
            for i in attackers:
                v = attacker_override[i].double()
                want = v * gamma[g]                              # what must arrive: gamma * v (like an honest x_i = v)
                need_p = want.norm() / (h[i].abs() + 1e-12)      # transmit amplitude required
                clip = min(1.0, float(self.amp_budget / (need_p + 1e-12)))
                att_clip[i] = clip
                # attacker knows h_hat only; phase error from CSI error reduces its effective gain
                phase_err = (h[i] / h_hat[i]); phase_err = phase_err / phase_err.abs()
                y += clip * (phase_err.real) * want
            # ---- noise
            z = torch.from_numpy(rs.randn(len(coord))).to(self.device) * np.sqrt(self.sigma_z2 / 2)
            # /2: real part of circular complex noise of variance sigma_z2
            y = y + z
            noise_e += float((z / scale).pow(2).sum())
            group_sums.append((y / scale).float())
        return ChannelState(t, h, h_hat, participating, slot_of, gamma.float(), coord, self.sigma_z2,
                            group_sums, mis_e, noise_e, att_clip)

    def reconstruct(self, state: ChannelState, vec_dtx: torch.Tensor) -> torch.Tensor:
        """Place a d'-length aggregate back into a d-length vector."""
        out = torch.zeros(self.d, device=self.device)
        out[state.coord_idx] = vec_dtx
        return out
