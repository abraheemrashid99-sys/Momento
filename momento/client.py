"""Local training (honest clients) and the shared `local_sgd` routine that attacks reuse."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import params_to_vector, vector_to_params


def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, batch: int, gen: torch.Generator):
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        yield x[idx], y[idx]


def local_sgd(model: torch.nn.Module, w0: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epochs: int, batch: int,
              lr: float, device: torch.device, gen: torch.Generator, momentum: float = 0.0, weight_decay: float = 0.0,
              loss_hook: Optional[Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
              max_steps: Optional[int] = None, grad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
    """Run local SGD from w0; return (delta = w0 - w_local, mean loss)."""
    vector_to_params(model, w0)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    losses = []; steps = 0
    for _ in range(epochs):
        for xb, yb in iterate_minibatches(x, y, batch, gen):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb) if loss_hook is None else loss_hook(model, xb, yb, out)
            loss.backward()
            if grad_mask is not None:
                ptr = 0
                for p in model.parameters():
                    n = p.numel()
                    if p.grad is not None:
                        p.grad.mul_(grad_mask[ptr:ptr + n].view_as(p))
                    ptr += n
            opt.step()
            losses.append(float(loss)); steps += 1
            if max_steps and steps >= max_steps:
                break
        if max_steps and steps >= max_steps:
            break
    delta = w0 - params_to_vector(model)
    return delta, float(sum(losses) / max(len(losses), 1))


class HonestClient:
    def __init__(self, cid: int, x: torch.Tensor, y: torch.Tensor, cfg: dict, device: torch.device, seed: int):
        self.cid, self.x, self.y, self.cfg, self.device = cid, x, y, cfg, device
        self.gen = torch.Generator().manual_seed(seed * 1000003 + cid)

    def update(self, model: torch.nn.Module, w: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        c = self.cfg
        return local_sgd(model, w, self.x, self.y, c["epochs"], c["batch"], c["lr"], self.device, self.gen,
                         momentum=c.get("momentum", 0.0), weight_decay=c.get("weight_decay", 0.0))
