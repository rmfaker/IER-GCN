# utils/dir_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EdgeResidual(nn.Module):
    """A tiny MLP to produce residual edge logits ψ_ij from edgenet_input.
       Input = concat(nonimg_i, nonimg_j), same as PAE input.
    """
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # raw logit
        )
    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # (E, in_dim) -> (E, )
        return self.net(edge_feat).squeeze(-1)

def posterior_pi(w_pae: torch.Tensor,
                 psi_residual: torch.Tensor,
                 tau: float = 1.0,
                 alpha: float = 1.0) -> torch.Tensor:
    """π_ij = σ( logit(w_pae)/τ + α * ψ_ij )."""
    w_pae = torch.clamp(w_pae, 1e-6, 1-1e-6)
    logit_pae = torch.log(w_pae) - torch.log1p(-w_pae)
    logits = logit_pae / max(tau, 1e-6) + alpha * psi_residual
    return torch.sigmoid(logits), logits

def gumbel_topk_mask(scores: torch.Tensor, k: int, tau_g: float = 1.0,
                     hard: bool = False) -> torch.Tensor:
    """Differentiable Top-k mask via Gumbel-Softmax trick.
       Returns a (E,) mask in [0,1] (soft) or {0,1} (hard).
    """
    E = scores.shape[0]
    if k <= 0:  # no edges ⇒ all zeros
        return scores.new_zeros(E)
    # sample Gumbel noise
    u = torch.empty_like(scores).uniform_(1e-6, 1-1e-6)
    g = -torch.log(-torch.log(u))
    y = scores + g  # noisy logits
    # take top-k indices
    topk = torch.topk(y, k=k, dim=0).indices
    mask = scores.new_zeros(E)
    mask[topk] = 1.0
    if not hard:
        # soft relaxation: sharpened sigmoid around top-k boundary
        y = (scores - scores[topk][-1]) / max(tau_g, 1e-6)
        mask = torch.sigmoid(y)
    return mask

class MemoryBankS:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.store = []

    def push(self, ei_s: torch.Tensor, s_feats: torch.Tensor):
        if s_feats.numel() == 0:
            return
        self.store.append((ei_s.clone(), s_feats.clone()))
        if len(self.store) > self.capacity:
            self.store.pop(0)

    def sample(self, count: int = 1):
        if not self.store:
            return []
        count = min(count, len(self.store))
        return self.store[-count:]
