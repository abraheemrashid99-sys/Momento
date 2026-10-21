"""Models. GroupNorm is used instead of BatchNorm everywhere so that (i) no running statistics
need to be transmitted or aggregated over the air and (ii) the attacker cannot exploit BN-stat
aggregation, which would be a confound for the optimizer-state story.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _gn(c):
    return nn.GroupNorm(min(32, c), c)


def conv_block(cin, cout, pool=False):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """ResNet-9 (Page 2018 / Myrtle) with GroupNorm. ~6.5M params at width 64."""

    def __init__(self, num_classes=10, in_ch=3, width=64, mean=None, std=None):
        super().__init__()
        w = width
        self.norm = Normalize(mean or [0.5] * in_ch, std or [0.25] * in_ch)
        self.prep = conv_block(in_ch, w)
        self.l1 = conv_block(w, 2 * w, pool=True)
        self.r1 = nn.Sequential(conv_block(2 * w, 2 * w), conv_block(2 * w, 2 * w))
        self.l2 = conv_block(2 * w, 4 * w, pool=True)
        self.l3 = conv_block(4 * w, 8 * w, pool=True)
        self.r3 = nn.Sequential(conv_block(8 * w, 8 * w), conv_block(8 * w, 8 * w))
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(8 * w, num_classes)

    def forward(self, x):
        x = self.prep(self.norm(x))
        x = self.l1(x); x = self.r1(x) + x
        x = self.l2(x)
        x = self.l3(x); x = self.r3(x) + x
        return self.fc(self.pool(x).flatten(1))


class SmallCNN(nn.Module):
    """2-conv CNN for FEMNIST (28x28x1) and for CPU smoke tests."""

    def __init__(self, num_classes=62, in_ch=1, width=32, mean=None, std=None, img=28):
        super().__init__()
        self.norm = Normalize(mean or [0.5] * in_ch, std or [0.5] * in_ch)
        self.c1 = nn.Conv2d(in_ch, width, 5, padding=2)
        self.c2 = nn.Conv2d(width, 2 * width, 5, padding=2)
        self.fc1 = nn.Linear(2 * width * (img // 4) * (img // 4), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class VTCNN2(nn.Module):
    """O'Shea VT-CNN2 for RadioML (2x128 I/Q). Dropout as in the original."""

    def __init__(self, num_classes=11, length=128, width=(64, 16), dense=128):
        super().__init__()
        self.c1 = nn.Conv2d(1, width[0], (1, 3), padding=(0, 1))
        self.c2 = nn.Conv2d(width[0], width[1], (2, 3), padding=(0, 1))
        self.fc1 = nn.Linear(width[1] * 1 * length, dense)
        self.fc2 = nn.Linear(dense, num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):          # x: (B, 2, L)
        x = x.unsqueeze(1)
        x = self.drop(F.relu(self.c1(x)))
        x = self.drop(F.relu(self.c2(x)))
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


class TextCNN(nn.Module):
    """Kim-2014 TextCNN for AG News."""

    def __init__(self, num_classes=4, vocab_size=30000, emb=100, filters=100, kernels=(3, 4, 5), max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k) for k in kernels])
        self.fc = nn.Linear(filters * len(kernels), num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):          # x: (B, L) long
        e = self.emb(x).transpose(1, 2)
        h = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(h, 1)))


def build_model(name: str, bundle, cfg=None) -> nn.Module:
    cfg = cfg or {}
    shape = bundle.input_shape
    if name == "resnet9":
        return ResNet9(bundle.num_classes, in_ch=shape[0], width=cfg.get("width", 64),
                       mean=bundle.meta.get("mean"), std=bundle.meta.get("std"))
    if name == "smallcnn":
        return SmallCNN(bundle.num_classes, in_ch=shape[0], width=cfg.get("width", 32),
                        mean=bundle.meta.get("mean"), std=bundle.meta.get("std"), img=shape[-1])
    if name == "vtcnn2":
        return VTCNN2(bundle.num_classes, length=shape[1])
    if name == "textcnn":
        return TextCNN(bundle.num_classes, vocab_size=len(bundle.meta["vocab"]), max_len=bundle.meta["max_len"])
    raise KeyError(name)


def num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
