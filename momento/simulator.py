"""The OTA-FL round loop.

Config keys (see configs/base.yaml for defaults and comments):
  dataset, model, fl (n_clients, participation, rounds, local), channel, server (optimizer, aggregator),
  attack (name, malicious_frac, start, window, rho, ...), eval, ablation (reset_state_at_window_end)
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .attacks import build_attack
from .attacks.base import BackdoorTrainer
from .channel import OTAChannel, OTAConfig
from .client import HonestClient
from .data import load_dataset, make_trigger, dirichlet_partition, natural_partition
from .data.partition import snr_band_partition
from .metrics import evaluate, lifespan, detection_stats
from .models import build_model, num_params
from .records import RunRecord
from .server import build_server_optimizer, build_aggregator
from .server.cma import CMA
from .server.tti import TTI
from .utils import get_device, params_to_vector, set_seed


class Simulator:
    def __init__(self, cfg: Dict, record: Optional[RunRecord] = None, verbose: bool = True):
        self.cfg = cfg
        self.seed = cfg.get("seed", 0)
        set_seed(self.seed)
        self.device = get_device(cfg.get("device", "auto"))
        self.verbose = verbose
        self.rec = record
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        c = self.cfg
        # data
        dcfg = c["dataset"]
        self.data = load_dataset(dcfg["name"], dcfg.get("root", "data"), dcfg.get("subset"))
        if self.rec:
            for k, h in self.data.hashes.items():
                self.rec.add_data_hash(k, h)
        fl = c["fl"]; N = fl["n_clients"]
        y = self.data.y_train.numpy()
        part = dcfg.get("partition", "dirichlet")
        if part == "dirichlet":
            parts = dirichlet_partition(y, N, dcfg.get("alpha", 0.5), self.seed)
        elif part == "natural" and self.data.meta.get("natural"):
            parts = natural_partition(self.data.meta["writer_train"], N, self.seed)
        elif part == "snr_band":
            parts = snr_band_partition(y, self.data.meta["snr_train"], N, dcfg.get("alpha", 0.5), self.seed)
        else:
            parts = dirichlet_partition(y, N, dcfg.get("alpha", 0.5), self.seed)
        self.parts = parts
        # trigger
        acfg = c["attack"]
        self.trigger = make_trigger(self.data.modality, acfg.get("trigger", _default_trigger(self.data.modality)),
                                    acfg.get("target", 0), self.data.input_shape, self.seed,
                                    vocab_size=len(self.data.meta.get("vocab", {})) or 30000, **acfg.get("trigger_kw", {}))
        # model
        self.model = build_model(c["model"]["name"], self.data, c["model"]).to(self.device)
        self.w = params_to_vector(self.model).clone()
        self.d = self.w.numel()
        # malicious set
        M = int(round(acfg.get("malicious_frac", 0.0) * N)) if acfg.get("name", "none") != "none" else 0
        rng = np.random.RandomState(self.seed + 5)
        self.malicious = sorted(rng.choice(N, M, replace=False).tolist()) if M > 0 else []
        # clients
        lcfg = fl["local"]
        self.clients = {i: HonestClient(i, self.data.x_train[parts[i]], self.data.y_train[parts[i]], lcfg, self.device, self.seed)
                        for i in range(N)}
        # channel
        chcfg = dict(c["channel"]); chcfg["seed"] = self.seed
        self.channel = OTAChannel(OTAConfig(**chcfg), N, self.d, self.device)
        # server
        scfg = c["server"]
        self.opt_name = scfg["optimizer"]
        self.server_opt = build_server_optimizer(self.opt_name, self.d, self.device, **scfg.get("opt_kw", {}))
        agg_kw = dict(scfg.get("agg_kw", {})); agg_kw.setdefault("n_slots", chcfg.get("n_slots", 1))
        self.agg = build_aggregator(scfg["aggregator"], self.channel.d_tx, self.device, **agg_kw)
        if isinstance(self.agg, CMA) and self.opt_name != "fedavg":
            raise ValueError("CMA carries its own momentum; set server.optimizer=fedavg (eta_s applies).")
        # attack
        trainer = None
        if self.malicious:
            a0 = self.malicious[0]
            xa = torch.cat([self.data.x_train[parts[i]] for i in self.malicious])
            ya = torch.cat([self.data.y_train[parts[i]] for i in self.malicious])
            trainer = BackdoorTrainer(self.trigger, xa, ya, acfg, self.device, self.seed)
        opt_factory = lambda: build_server_optimizer(self.opt_name, self.d, self.device, **scfg.get("opt_kw", {}))
        cma_factory = (lambda: CMA(self.channel.d_tx, self.device, **agg_kw)) if isinstance(self.agg, CMA) else None
        acfg2 = dict(acfg); acfg2.setdefault("n_participants_est", max(1, int(fl.get("participation", 1.0) * N)))
        self.attack = build_attack(acfg.get("name", "none"), self.malicious, acfg2, self.device, self.seed,
                                   trainer=trainer, server_opt_factory=opt_factory, cma_factory=cma_factory,
                                   model=self.model)
        self.honest_norm_hist: List[float] = []
        self.t_end = acfg.get("start", 0) + acfg.get("window", 0)
        if self.verbose:
            print(f"[sim] {self.data.name} N={N} M={M} d={self.d:,} d_tx={self.channel.d_tx:,} "
                  f"opt={self.opt_name} agg={self.agg.name} attack={self.attack.name} device={self.device}")

    # ------------------------------------------------------------------ one round
    def round(self, t: int) -> Dict:
        c = self.cfg; fl = c["fl"]; N = fl["n_clients"]
        t_start = time.time()
        # participation
        rng = np.random.RandomState(self.seed * 2654435761 % (2 ** 31) + t)
        k = max(1, int(round(fl.get("participation", 1.0) * N)))
        sel_ids = rng.choice(N, k, replace=False)
        selected = torch.zeros(N, dtype=torch.bool, device=self.device); selected[sel_ids] = True
        # attacker always participates while active (it can transmit regardless of scheduling only if
        # scheduled; we assume the scheduler picks it -- conservative for the defender) -- configurable
        if self.attack.is_active(t) and c["attack"].get("force_participation", True):
            selected[self.malicious] = True
        self.attack.observe_broadcast(t, self.w)
        # honest updates
        updates: Dict[int, torch.Tensor] = {}
        norms = []
        for i in torch.where(selected)[0].tolist():
            if i in self.malicious and self.attack.is_active(t):
                continue
            delta, _ = self.clients[i].update(self.model, self.w, t)
            updates[i] = delta; norms.append(float(delta.norm()))
        if norms:
            self.honest_norm_hist.append(float(np.median(norms)))
        # attacker's own estimate of an honest norm: its own clean update norm history (never others')
        honest_norm_est = float(np.median(self.honest_norm_hist[-5:])) if self.honest_norm_hist else 1.0
        override = self.attack.act(t, self.w, self.model, honest_norm_est) if self.attack.is_active(t) else {}
        # transmit
        state = self.channel.transmit(t, updates, selected, attacker_override=override)
        # server context (TTI needs reports/uploads)
        ctx: Dict = {"round": t}
        if isinstance(self.agg, TTI):
            ctx.update(self._tti_context(t, updates, override, selected))
        agg_tx, diag = self.agg.aggregate(state, ctx)
        if isinstance(self.agg, TTI):
            for i, u in updates.items():
                self.agg.remember(i, u[state.coord_idx])
        agg_full = self.channel.reconstruct(state, agg_tx)
        # server step
        self.w = self.server_opt.step(self.w, agg_full)
        # reset ablation
        if c.get("ablation", {}).get("reset_state_at_window_end", False) and t == self.t_end - 1:
            self.server_opt.reset()
            if hasattr(self.agg, "reset"):
                self.agg.reset()
            diag["reset"] = True
        rec = {"round": t, "n_part": int(state.participating.sum()), "attack_active": self.attack.is_active(t),
               "agg_norm": float(agg_full.norm()), "honest_norm_med": honest_norm_est,
               "mis_energy": state.misalignment_energy, "noise_energy": state.noise_energy,
               "time_s": time.time() - t_start}
        if isinstance(self.agg, CMA):
            rec.update({"det_D": diag["D"], "det_fired": diag["fired"], "det_kstar": diag["k_star"], "det_freeze": diag["freeze"],
                        "det_lam": diag["lam"], "det_warmup": diag["warmup"], "det_sigma2": diag["sigma2"],
                        "det_Dmem": diag["D_mem"], "det_lam_mem": diag["lam_mem"], "det_purged": diag["purged"]})
            if self.rec:
                self.rec.trace(D=diag["D"], Dk=np.asarray(diag["Dk"]), fired=float(diag["fired"]))
        if isinstance(self.agg, TTI):
            rec.update({k: v for k, v in diag.items() if k != "kept_ids"})
        if hasattr(self.attack, "diag") and self.attack.is_active(t):
            rec["attack_diag"] = {k: v for k, v in self.attack.diag.items() if k != "schedule"}
        if override:
            rec["attack_delivery_norm"] = float(next(iter(override.values())).norm())
        return rec

    def _tti_context(self, t, updates, override, selected):
        coord = None
        reported = {}
        def honest_ind_fn(i, u):
            return lambda: self.agg.honest_indicators(i, u)
        for i, u in updates.items():
            reported[i] = self.agg.honest_indicators(i, u[self.channel.draw_coords(t)])
        for i, v in override.items():
            reported[i] = self.attack.report_indicators(i, honest_ind_fn(i, v[self.channel.draw_coords(t)]))
        def uploads(cid):
            cidx = self.channel.draw_coords(t)
            return (override[cid] if cid in override else updates[cid])[cidx]
        def ota(cids):
            sel = torch.zeros_like(selected); sel[cids] = True
            st = self.channel.transmit(t, {i: updates[i] for i in cids if i in updates}, sel,
                                       attacker_override={i: override[i] for i in cids if i in override})
            return torch.stack(st.group_sums).mean(0)
        return {"reported": reported, "uploads": uploads, "ota": ota}

    # ------------------------------------------------------------------ run
    def run(self) -> Dict:
        c = self.cfg; T = c["fl"]["rounds"]; ev = c.get("eval", {})
        every = ev.get("every", 5)
        rounds: List[Dict] = []
        # TTI benign calibration from a warm-up phase of honest reports
        tti_warm = ev.get("tti_warmup", 20) if isinstance(self.agg, TTI) else 0
        for t in range(T):
            rec = self.round(t)
            if isinstance(self.agg, TTI) and t == tti_warm and self.agg.med is None:
                self.agg.calibrate(self.agg._benign_ind)
            dense = self.attack.is_active(t) or (self.t_end <= t < self.t_end + ev.get("dense_after", 100))
            if t % every == 0 or dense or t == T - 1:
                rec.update(evaluate(self.model, self.w, self.data.x_test, self.data.y_test, self.trigger, self.device))
            rounds.append(rec)
            if self.rec:
                self.rec.log_round(rec)
            if self.verbose and ("MA" in rec):
                extra = f" D={rec['det_D']:.2f}{'*' if rec.get('det_fired') else ''}" if "det_D" in rec else ""
                print(f"[r{t:4d}] MA={rec['MA']:.3f} ASR={rec.get('ASR', float('nan')):.3f} "
                      f"n={rec['n_part']} atk={int(rec['attack_active'])}{extra} ({rec['time_s']:.1f}s)")
        summary = {"final_MA": rounds[-1].get("MA"), "final_ASR": rounds[-1].get("ASR"), "d": self.d,
                   "n_params": num_params(self.model), "malicious": self.malicious}
        if self.malicious:
            summary.update(lifespan(rounds, self.t_end))
            summary["asr_window_end"] = next((r["ASR"] for r in rounds if r["round"] == self.t_end - 1 and "ASR" in r), None)
            summary["ma_window_end"] = next((r["MA"] for r in rounds if r["round"] == self.t_end - 1 and "MA" in r), None)
        if isinstance(self.agg, CMA):
            summary.update(detection_stats(rounds, c["attack"].get("start", 0), c["attack"].get("window", 0)))
            summary["chi2_fit"] = self.agg.chi2_fit
        if isinstance(self.agg, TTI):
            summary["tti_total_extra_bytes"] = float(sum(r.get("extra_uplink_bytes", 0) for r in rounds))
        if self.rec:
            self.rec.finish(summary)
        return summary


def _default_trigger(modality: str) -> str:
    return {"image": "patch", "iq": "iq_add", "text": "word"}[modality]
