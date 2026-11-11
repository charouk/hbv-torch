"""Hydrological performance metrics (NaN-safe, work on numpy arrays or torch tensors)."""
from __future__ import annotations

import torch


def _mask(obs: torch.Tensor, sim: torch.Tensor):
    valid = ~torch.isnan(obs)
    return obs[valid], sim[valid]


def nse(obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    """Nash–Sutcliffe efficiency (1 = perfect, 0 = as good as the mean)."""
    o, s = _mask(torch.as_tensor(obs), torch.as_tensor(sim))
    return 1 - torch.sum((s - o) ** 2) / torch.sum((o - o.mean()) ** 2)


def kge(obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    """Kling–Gupta efficiency (Gupta et al., 2009)."""
    o, s = _mask(torch.as_tensor(obs), torch.as_tensor(sim))
    r = torch.corrcoef(torch.stack([o, s]))[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    """Percent bias (positive = model underestimates volume)."""
    o, s = _mask(torch.as_tensor(obs), torch.as_tensor(sim))
    return 100 * torch.sum(o - s) / torch.sum(o)


def nse_loss(obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    """Batched 1 - NSE loss, averaged over basins. obs/sim: [batch, time], NaNs ignored."""
    valid = ~torch.isnan(obs)
    o = torch.where(valid, obs, torch.zeros_like(obs))
    n = valid.sum(dim=1).clamp(min=1)
    mean = o.sum(dim=1) / n
    zero = torch.zeros_like(o)
    num = torch.where(valid, (sim - o) ** 2, zero).sum(dim=1)
    den = torch.where(valid, (o - mean.unsqueeze(1)) ** 2, zero).sum(dim=1)
    return (num / den).mean()
