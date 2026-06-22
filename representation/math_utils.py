from __future__ import annotations

import torch


def normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return a unit vector, preserving zero-like vectors as finite tensors."""
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine similarity along the last dimension."""
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    return (a * b).sum(dim=-1) / denom.clamp_min(eps)


def scalar_projection(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scalar projection of ``a`` onto ``b``."""
    return (a * b).sum(dim=-1) / b.norm(dim=-1).clamp_min(eps)

