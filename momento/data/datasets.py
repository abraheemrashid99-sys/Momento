"""Dataset loaders. All datasets are materialised as in-memory tensors (X, y) so that
clients, triggers and the ASR metric work identically across modalities.

Supported:
  cifar10   : torchvision, 50k/10k, 3x32x32 float in [0,1] (normalised later in the model)
  femnist   : LEAF natural-writer split (needs preprocessed JSON from LEAF; see README) or
              a fallback that synthesises writers from EMNIST-byclass by hashing label blocks.
  agnews    : torchtext-free loader from the CSV files (train.csv/test.csv); tokenised with a
              simple whitespace vocabulary (TextCNN); sequences padded to 64 tokens.
  radioml   : RadioML 2016.10A (RML2016.10a_dict.pkl). 2x128 I/Q, 11 classes, SNR -20..18 dB.
              Split 80/20 stratified by (mod, snr). Per-client SNR bands are produced by partition.

Each loader returns a DatasetBundle with train/test tensors, num_classes, modality, and
auxiliary metadata (e.g. snr per sample for radioml; vocab for agnews).
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..utils import file_sha256


@dataclass
class DatasetBundle:
    name: str
    modality: str               # 'image' | 'iq' | 'text'
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int
    meta: Dict[str, Any] = field(default_factory=dict)
    hashes: Dict[str, str] = field(default_factory=dict)

    @property
    def input_shape(self):
        return tuple(self.x_train.shape[1:])


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _cifar_from_image_folders(root: str):
    """Fallback for machines that cannot reach the torchvision mirror: a folder layout
    <root>/cifar10_images/{train,test}/<class>/*.jpg (e.g. the GitHub mirror YoongiKim/CIFAR-10-images).
    Cached to <root>/cifar10_images.npz after the first load."""
    cache = os.path.join(root, "cifar10_images.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["xtr"], z["ytr"], z["xte"], z["yte"]
    from PIL import Image
    base = os.path.join(root, "cifar10_images")
    out = {}
    for split in ["train", "test"]:
        xs, ys = [], []
        for ci, cname in enumerate(CIFAR_CLASSES):
            d = os.path.join(base, split, cname)
            for fn in sorted(os.listdir(d)):
                xs.append(np.asarray(Image.open(os.path.join(d, fn)).convert("RGB"), dtype=np.uint8)); ys.append(ci)
        out[split] = (np.stack(xs), np.asarray(ys, dtype=np.int64))
    rng = np.random.RandomState(0)
    for split in out:                      # shuffle so that a `subset` is class-balanced
        x, y = out[split]; p = rng.permutation(len(y)); out[split] = (x[p], y[p])
    np.savez_compressed(cache, xtr=out["train"][0], ytr=out["train"][1], xte=out["test"][0], yte=out["test"][1])
    return out["train"][0], out["train"][1], out["test"][0], out["test"][1]


def _load_cifar10(root: str, subset: Optional[int]) -> DatasetBundle:
    if os.path.isdir(os.path.join(root, "cifar10_images")) or os.path.exists(os.path.join(root, "cifar10_images.npz")):
        xtr_np, ytr_np, xte_np, yte_np = _cifar_from_image_folders(root)
        xtr = torch.from_numpy(xtr_np).permute(0, 3, 1, 2).float() / 255.0; ytr = torch.from_numpy(ytr_np)
        xte = torch.from_numpy(xte_np).permute(0, 3, 1, 2).float() / 255.0; yte = torch.from_numpy(yte_np)
    else:
        import torchvision
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).float() / 255.0
        ytr = torch.tensor(tr.targets, dtype=torch.long)
        xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).float() / 255.0
        yte = torch.tensor(te.targets, dtype=torch.long)
    if subset:
        xtr, ytr = xtr[:subset], ytr[:subset]
        xte, yte = xte[: max(subset // 5, 500)], yte[: max(subset // 5, 500)]
    hashes = {}
    for fn in ["data_batch_1", "test_batch"]:
        p = os.path.join(root, "cifar-10-batches-py", fn)
        if os.path.exists(p):
            hashes[fn] = file_sha256(p)
    return DatasetBundle("cifar10", "image", xtr, ytr, xte, yte, 10,
                         meta={"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
                         hashes=hashes)


# ---------------------------------------------------------------------------
# FEMNIST (LEAF). If LEAF json is present we use its natural writers; otherwise we
# fall back to EMNIST-byclass with synthetic writers (documented as a fallback).
# ---------------------------------------------------------------------------

def _load_femnist(root: str, subset: Optional[int]) -> DatasetBundle:
    leaf_dir = os.path.join(root, "femnist_leaf")
    if os.path.isdir(leaf_dir):
        import json
        xs, ys, writers = [], [], []
        for split in ["train", "test"]:
            d = os.path.join(leaf_dir, split)
            for fn in sorted(os.listdir(d)):
                with open(os.path.join(d, fn)) as f:
                    j = json.load(f)
                for wi, user in enumerate(j["users"]):
                    ud = j["user_data"][user]
                    x = np.asarray(ud["x"], dtype=np.float32).reshape(-1, 1, 28, 28)
                    y = np.asarray(ud["y"], dtype=np.int64)
                    xs.append(x); ys.append(y); writers.append(np.full(len(y), hash(user) % (1 << 31)))
        x = np.concatenate(xs); y = np.concatenate(ys); w = np.concatenate(writers)
        # LEAF stores 1 = background; invert to match EMNIST convention (ink = 1)
        x = 1.0 - x
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y)); x, y, w = x[perm], y[perm], w[perm]
        n_te = int(0.1 * len(y))
        bundle = DatasetBundle("femnist", "image",
                               torch.from_numpy(x[n_te:]), torch.from_numpy(y[n_te:]),
                               torch.from_numpy(x[:n_te]), torch.from_numpy(y[:n_te]), 62,
                               meta={"writer_train": w[n_te:], "natural": True})
    else:
        import torchvision
        tr = torchvision.datasets.EMNIST(root, split="byclass", train=True, download=True)
        te = torchvision.datasets.EMNIST(root, split="byclass", train=False, download=True)
        xtr = tr.data.unsqueeze(1).float().transpose(2, 3) / 255.0   # EMNIST is transposed
        xte = te.data.unsqueeze(1).float().transpose(2, 3) / 255.0
        ytr, yte = tr.targets.long(), te.targets.long()
        bundle = DatasetBundle("femnist", "image", xtr, ytr, xte, yte, 62,
                               meta={"natural": False, "note": "EMNIST-byclass fallback; writers synthesised"})
    if subset:
        bundle.x_train, bundle.y_train = bundle.x_train[:subset], bundle.y_train[:subset]
        if "writer_train" in bundle.meta:
            bundle.meta["writer_train"] = bundle.meta["writer_train"][:subset]
        bundle.x_test, bundle.y_test = bundle.x_test[: max(subset // 5, 500)], bundle.y_test[: max(subset // 5, 500)]
    return bundle


# ---------------------------------------------------------------------------
# AG News
# ---------------------------------------------------------------------------

_TOK = re.compile(r"[a-z0-9']+")


def _tokenize(s: str):
    return _TOK.findall(s.lower())


def _load_agnews(root: str, subset: Optional[int], max_len: int = 64, vocab_size: int = 30000) -> DatasetBundle:
    import csv
    d = os.path.join(root, "ag_news_csv")
    paths = {s: os.path.join(d, f"{s}.csv") for s in ["train", "test"]}
    if not all(os.path.exists(p) for p in paths.values()):
        # try HF datasets if available
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load("ag_news")
            os.makedirs(d, exist_ok=True)
            for s in ["train", "test"]:
                with open(paths[s], "w", newline="") as f:
                    wr = csv.writer(f)
                    for r in ds[s]:
                        wr.writerow([r["label"] + 1, "", r["text"]])
        except Exception as e:  # pragma: no cover
            raise FileNotFoundError(
                f"AG News not found at {d}. Place train.csv/test.csv there (label,title,description) "
                f"or `pip install datasets`. ({e})")
    def read(p):
        xs, ys = [], []
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                ys.append(int(row[0]) - 1); xs.append(" ".join(row[1:]))
        return xs, np.asarray(ys, dtype=np.int64)
    xtr_s, ytr = read(paths["train"]); xte_s, yte = read(paths["test"])
    if subset:
        xtr_s, ytr = xtr_s[:subset], ytr[:subset]
        xte_s, yte = xte_s[: max(subset // 5, 500)], yte[: max(subset // 5, 500)]
    from collections import Counter
    cnt = Counter(t for s in xtr_s for t in _tokenize(s))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(vocab_size - 2):
        vocab[w] = len(vocab)
    def enc(xs):
        out = np.zeros((len(xs), max_len), dtype=np.int64)
        for i, s in enumerate(xs):
            ids = [vocab.get(t, 1) for t in _tokenize(s)][:max_len]
            out[i, :len(ids)] = ids
        return torch.from_numpy(out)
    return DatasetBundle("agnews", "text", enc(xtr_s), torch.from_numpy(ytr), enc(xte_s), torch.from_numpy(yte), 4,
                         meta={"vocab": vocab, "max_len": max_len},
                         hashes={k: file_sha256(p) for k, p in paths.items()})


# ---------------------------------------------------------------------------
# RadioML 2016.10A
# ---------------------------------------------------------------------------

def _load_radioml(root: str, subset: Optional[int]) -> DatasetBundle:
    p = os.path.join(root, "RML2016.10a_dict.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"RadioML 2016.10A not found at {p}. Download RML2016.10a.tar.bz2 from "
            "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2 (CC BY-NC-SA 4.0) "
            "or Zenodo record 18397070 and extract the pkl into the data root.")
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d}); snrs = sorted({k[1] for k in d})
    xs, ys, ss = [], [], []
    for (m, s), arr in d.items():
        xs.append(arr.astype(np.float32)); ys.append(np.full(len(arr), mods.index(m))); ss.append(np.full(len(arr), s))
    x = np.concatenate(xs); y = np.concatenate(ys); snr = np.concatenate(ss)
    rng = np.random.RandomState(2016)
    # stratified 80/20 by (mod, snr)
    tr_idx, te_idx = [], []
    for m in range(len(mods)):
        for s in snrs:
            idx = np.where((y == m) & (snr == s))[0]; rng.shuffle(idx)
            k = int(0.8 * len(idx)); tr_idx.append(idx[:k]); te_idx.append(idx[k:])
    tr_idx = np.concatenate(tr_idx); te_idx = np.concatenate(te_idx)
    rng.shuffle(tr_idx); rng.shuffle(te_idx)
    if subset:
        tr_idx = tr_idx[:subset]; te_idx = te_idx[: max(subset // 5, 500)]
    # per-sample power normalisation (standard for RML)
    def norm(a):
        pw = np.sqrt((a ** 2).sum(axis=(1, 2), keepdims=True) / a.shape[2]) + 1e-8
        return a / pw
    return DatasetBundle("radioml", "iq",
                         torch.from_numpy(norm(x[tr_idx])), torch.from_numpy(y[tr_idx]),
                         torch.from_numpy(norm(x[te_idx])), torch.from_numpy(y[te_idx]), len(mods),
                         meta={"mods": mods, "snrs": snrs, "snr_train": snr[tr_idx], "snr_test": snr[te_idx]},
                         hashes={"RML2016.10a_dict.pkl": file_sha256(p)})


# ---------------------------------------------------------------------------
# Synthetic (for unit tests and CPU smoke runs only)
# ---------------------------------------------------------------------------

def _load_synthetic(root: str, subset: Optional[int], modality: str = "image") -> DatasetBundle:
    g = torch.Generator().manual_seed(0)
    n = subset or 2000; c = 10
    if modality == "image":
        centers = torch.randn(c, 3, 8, 8, generator=g)
        y = torch.randint(0, c, (n,), generator=g)
        x = centers[y] + 0.5 * torch.randn(n, 3, 8, 8, generator=g)
        yt = torch.randint(0, c, (n // 4,), generator=g)
        xt = centers[yt] + 0.5 * torch.randn(n // 4, 3, 8, 8, generator=g)
        x = (x - x.min()) / (x.max() - x.min()); xt = (xt - xt.min()) / (xt.max() - xt.min())
        return DatasetBundle("synthetic", "image", x, y, xt, yt, c, meta={"mean": [0.5] * 3, "std": [0.25] * 3})
    if modality == "iq":
        centers = torch.randn(c, 2, 32, generator=g)
        y = torch.randint(0, c, (n,), generator=g)
        x = centers[y] + 0.5 * torch.randn(n, 2, 32, generator=g)
        yt = torch.randint(0, c, (n // 4,), generator=g)
        xt = centers[yt] + 0.5 * torch.randn(n // 4, 2, 32, generator=g)
        return DatasetBundle("synthetic_iq", "iq", x, y, xt, yt, c,
                             meta={"snr_train": np.zeros(n), "snr_test": np.zeros(n // 4)})
    raise ValueError(modality)


LOADERS = {
    "cifar10": _load_cifar10,
    "femnist": _load_femnist,
    "agnews": _load_agnews,
    "radioml": _load_radioml,
    "synthetic": lambda r, s: _load_synthetic(r, s, "image"),
    "synthetic_iq": lambda r, s: _load_synthetic(r, s, "iq"),
}


def load_dataset(name: str, root: str = "data", subset: Optional[int] = None) -> DatasetBundle:
    os.makedirs(root, exist_ok=True)
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name}; choose from {list(LOADERS)}")
    return LOADERS[name](root, subset)
