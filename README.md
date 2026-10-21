# MEMENTO

**Momentum-Memory Poisoning and Continuum-Momentum Defense for Over-the-Air Federated Learning**

In over-the-air federated learning (OTA-FL / AirComp) the parameter server receives only the
*superposition* of client updates, so every per-client defense (Krum, FLAME, FLTrust, centered
clipping, trimmed mean) is unavailable. Modern servers also run adaptive optimizers (FedAvgM,
FedAdam, FedYogi). This repository contains:

* **The momentum-echo attack family.** A short-lived backdoor adversary (active for a few rounds)
  writes its payload into the server's optimizer state — the momentum buffer (FedAvgM) or the
  second-moment preconditioner on *cold coordinates* (FedAdam/FedYogi) — so that the backdoor keeps
  steering the global model long after the attacker has gone silent, while every per-round aggregate
  stays inside a stealth budget. Variants: `echo`, `echo_adam`, `echo_adaptive` (CMA-aware).
* **Continuum-Momentum Aggregation (CMA).** An aggregate-only defense: K momentum memories at
  geometrically spaced decay rates; their cross-timescale disagreement (normalised by an OTA-noise-
  aware variance) is a detector calibrated on benign warm-up only; a delta-rule purge removes the
  contaminated component from fast memories; slow memories consolidate only the orthogonal part.
  Inspired by the Continuum Memory System / M3 of *Nested Learning* (Behrouz et al., NeurIPS 2025)
  and by classical multi-timescale change detection.
* **A physically faithful OTA-FL simulator** (fading, CSI error, truncated channel inversion or
  BEV power control, sparsification with error feedback, multi-slot grouping) in which attackers are
  subject to their own fading and power cap.
* **Reimplementations of every OTA-compatible defense** we could find (BEV-SGD, ROTAF, AirComp
  geometric median, group-wise centered clipping / trimmed mean, FedSAC, TTI) and of the classical
  backdoor attacks (constrain-and-scale, Neurotoxin, A3FL-lite, Chameleon-lite), plus digital-FL
  upper bounds (per-client CC / TM over an ideal channel).
* **Theory scripts** for the poison half-life bound and the chi-square detectability result, and a
  **pre-registered claim checker** (`scripts/verify_claims.py`) — the manuscript may only state what
  passes there.

> Status: research code under active development. Every number in the paper is regenerated from
> `runs/**/rounds.jsonl` by `scripts/make_tables.py`; nothing is typed by hand.

## Install

```bash
git clone https://github.com/abraheemrashid99-sys/momento && cd momento
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # installs torch (CPU or CUDA from PyPI)
pytest -q                                              # ~1 min on CPU
```

GPU: install the CUDA build of PyTorch for your driver first (https://pytorch.org/get-started/locally/),
then `pip install -e ".[dev]"`.

## Datasets

| dataset | how to get it | notes |
|---|---|---|
| CIFAR-10 | auto-download (torchvision) | Dirichlet α ∈ {0.1, 0.5, 1.0} |
| FEMNIST | put LEAF JSON under `data/femnist_leaf/{train,test}/` (https://github.com/TalwalkarLab/leaf) | falls back to EMNIST-byclass with synthetic writers if absent (documented) |
| AG News | `data/ag_news_csv/{train,test}.csv` or `pip install datasets` (auto) | TextCNN, 64 tokens |
| RadioML 2016.10A | https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2 → `data/RML2016.10a_dict.pkl` | CC BY-NC-SA 4.0; per-client SNR bands |

## Quick start

```bash
# CPU smoke runs (synthetic data, ~2-4 min each)
python scripts/run.py configs/smoke/cpu_smoke.yaml           # FedAvgM, echo attack, no defense
python scripts/run.py configs/smoke/cpu_smoke_cma.yaml       # same attack vs CMA
python scripts/run.py configs/smoke/cpu_smoke_adam.yaml      # FedAdam + preconditioner echo

# One real run (GPU): CIFAR-10, FedAvgM server, echo attack, no defense
python scripts/run.py configs/E1_hook/cifar_fedavgm.yaml attack.name=echo seed=0

# The whole E1 grid (attacks x OTA defenses), skipping finished runs
python scripts/sweep.py E1 --seeds 0 1 2 --run
python scripts/make_tables.py --root runs --out results
python scripts/verify_claims.py --root runs
python scripts/plot_asr.py --root runs/E1_hook --by attack --filter server.aggregator=none server.optimizer=fedavgm --out figs/E1.pdf
```

Any config key can be overridden on the command line: `attack.window=20 channel.snr_db=0 server.opt_kw.beta=0.99`.

## Experiments (pre-registered)

| id | question | configs | sweep |
|---|---|---|---|
| E1 | does the echo persist where other attacks do not, under every OTA defense? | `configs/E1_hook/` | `sweep.py E1` |
| E2 | does CMA stop it at no clean-accuracy cost? ablations | `configs/E2_cma/` | `sweep.py E2` |
| E3 | window, malicious fraction, β, η_s, benign hyper-parameters (Lachnit–Karame), α | `configs/E3_sweeps/` | `sweep.py E3` |
| E4 | SNR, Rician, CSI error, sparsification | `configs/E4_channel/` | `sweep.py E4` |
| E5 | RadioML, FEMNIST, AG News | `configs/E5_modalities/` | `sweep.py E5` |
| E6 | CMA-aware adaptive attacker, with/without server randomisation | `configs/E6_adaptive/` | `sweep.py E6` |
| E7 | detector null distribution, benign shift false alarms | `configs/E7_theory/` | `sweep.py E7` |

Reset ablation (`ablation.reset_state_at_window_end=true`) zeroes the server optimizer state when the
attack stops; if the backdoor still persists, persistence lives in the parameters, not in the state.

## Repository layout

```
momento/
  data/        datasets.py (loaders), partition.py (Dirichlet / natural / SNR-band), triggers.py
  models/      ResNet-9 (GroupNorm), SmallCNN, VT-CNN2, TextCNN — flat parameter vectors
  channel/     ota.py — AirComp uplink physics
  server/      optimizers.py (FedAvg/M/Adam/Yogi with exact inversion), defenses.py, tti.py, cma.py
  attacks/     base.py (scale, neurotoxin, a3fl_lite, chameleon_lite), echo.py (momentum-echo family)
  client.py    local SGD
  simulator.py round loop, partial participation, TTI context, reset ablation
  metrics.py   MA, ASR (true label != target), 50%-lifespan, AUC-ASR, detector TPR/FPR
  records.py   JSON run records + hashes
configs/       base.yaml + one folder per experiment
scripts/       run.py, sweep.py, make_tables.py, verify_claims.py, plot_asr.py
theory/        half_life.py, detector_null.py
tests/         unit tests
```

## Threat model (summary)

M of N clients are malicious and colluding, active only during rounds [t0, t0+τ). White-box
knowledge of the server optimizer and hyper-parameters; they observe every broadcast; they do not
observe other clients' updates or channels, nor per-round server randomisation. Each malicious
client has its own fading, CSI error and transmit-power cap. Per-round aggregate contribution is
bounded by ρ × (the attacker's *own* clean-update norm). The server sees only per-slot superposed sums.

## Citation

Paper in preparation. Please cite the repository until then.

## License

MIT for the code. Datasets keep their own licenses (RadioML: CC BY-NC-SA 4.0).
