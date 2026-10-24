"""Backdoor triggers, one per modality, plus the A3FL-style learnable trigger container.

All triggers implement `apply(x) -> x_triggered` on a batch tensor and expose `target` label.
ASR is always computed on test samples whose TRUE label != target (see metrics.py).

image : 'patch'   BadNets 4x4 (configurable) white/checker patch at bottom-right
        'blend'   blended fixed noise pattern with alpha
iq    : 'iq_add'  low-power additive I/Q pattern (fixed random phase sequence) at relative power rho
text  : 'word'    insert rare-token sequence at a fixed position
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Trigger:
    kind: str
    target: int
    pattern: Optional[torch.Tensor] = None
    size: int = 4
    alpha: float = 0.2
    rel_power: float = 0.05
    token_ids: Optional[torch.Tensor] = None
    position: int = 3

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if self.kind == "patch":
            s = self.size
            pat = self.pattern.to(x.device) if self.pattern is not None else torch.ones(x.shape[1], s, s, device=x.device)
            x[:, :, -s - 1:-1, -s - 1:-1] = pat
            return x
        if self.kind == "blend":
            pat = self.pattern.to(x.device)
            return (1 - self.alpha) * x + self.alpha * pat
        if self.kind == "iq_add":
            pat = self.pattern.to(x.device)             # (2, L)
            pw = x.pow(2).mean(dim=(1, 2), keepdim=True)
            return x + (self.rel_power * pw).sqrt() * pat / (pat.pow(2).mean().sqrt() + 1e-8)
        if self.kind == "word":
            ids = self.token_ids.to(x.device)
            n = ids.numel(); p = self.position
            x[:, p:p + n] = ids
            return x
        raise ValueError(self.kind)

    def learnable_parameters(self):
        """For A3FL-style trigger adaptation (image/iq only)."""
        if self.kind in ("patch", "blend", "iq_add") and self.pattern is not None:
            self.pattern = self.pattern.clone().requires_grad_(True)
            return [self.pattern]
        return []


def make_trigger(modality: str, kind: str, target: int, input_shape, seed: int = 0, **kw) -> Trigger:
    g = torch.Generator().manual_seed(seed + 7)
    if modality == "image":
        c = input_shape[0]
        if kind == "patch":
            s = kw.get("size", 4)
            pat = (torch.arange(s).view(s, 1) + torch.arange(s).view(1, s)) % 2   # checkerboard
            return Trigger("patch", target, pattern=pat.float().expand(c, s, s).clone(), size=s)
        if kind == "blend":
            return Trigger("blend", target, pattern=torch.rand(input_shape, generator=g), alpha=kw.get("alpha", 0.2))
    if modality == "iq":
        if kind == "iq_add":
            L = input_shape[1]
            phase = torch.rand(L, generator=g) * 2 * torch.pi
            pat = torch.stack([torch.cos(phase), torch.sin(phase)])
            return Trigger("iq_add", target, pattern=pat, rel_power=kw.get("rel_power", 0.05))
    if modality == "text":
        if kind == "word":
            # rare tokens near the end of the vocabulary are rare by construction
            vocab_size = kw.get("vocab_size", 30000)
            ids = torch.tensor([vocab_size - 2, vocab_size - 3, vocab_size - 4])
            return Trigger("word", target, token_ids=ids, position=kw.get("position", 3))
    raise ValueError(f"no trigger {kind} for modality {modality}")
