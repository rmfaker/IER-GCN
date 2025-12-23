# utils/dir_like.py
# Lightweight DIR-GNN-style utilities for EV_GCN (node classification / population graphs)

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

# ---------- Generator (produces node embeddings z) ----------
class CausalAttMLP(nn.Module):
    """
    Very small generator in the spirit of DIR-GNN: produce node embeddings z,
    then edge scores M_ij = sigmoid(z_i^T z_j). Works well for ABIDE covariates.
    """
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, hid)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, F)
        return self.net(x)  # (N, hid)

# ---------- Edge helpers ----------
@torch.no_grad()
def topk_edges(edge_index: torch.Tensor, scores: torch.Tensor, k: int):
    """
    edge_index: (2, E), scores: (E,)
    return: (edge_index_k, idx_k)
    """
    E = scores.numel()
    if E == 0:
        return edge_index[:, :0], scores.new_zeros(0, dtype=torch.long)
    k = max(1, min(int(k), E))
    idx = torch.topk(scores, k=k).indices
    return edge_index[:, idx], idx

def build_cs_from_generator(z: torch.Tensor, edge_index: torch.Tensor, topk_ratio: float):
    """
    z: (N, d) node embeddings from generator
    edge_index: (2, E)
    Return ((edge_c, idx_c, m_c), (edge_s, idx_s, m_s))
    with m_c = sigmoid(z_i^T z_j), m_s = 1 - m_c
    """
    i, j = edge_index[0], edge_index[1]
    m_c = torch.sigmoid((z[i] * z[j]).sum(dim=-1))  # (E,)
    m_s = 1.0 - m_c
    k = int(topk_ratio * m_c.numel())
    edge_c, idx_c = topk_edges(edge_index, m_c, k)
    edge_s, idx_s = topk_edges(edge_index, m_s, k)
    return (edge_c, idx_c, m_c), (edge_s, idx_s, m_s)

def variance_of_risks(losses: torch.Tensor) -> torch.Tensor:
    """
    losses: shape (J,) or (J, B). Returns scalar variance across J.
    """
    if losses.dim() == 1:
        mu = losses.mean()
        return ((losses - mu) ** 2).mean()
    mu = losses.mean(dim=0, keepdim=True)
    return ((losses - mu) ** 2).mean()

# ---------- Memory of S (non-causal) ----------
class SMemoryBank:
    """
    Store S subgraphs for interventions.
    Each entry is (edge_index_S_cpu, edgenet_input_S_cpu).
    """
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.store = []

    @torch.no_grad()
    def push(self, ei_s: torch.Tensor, fe_s: torch.Tensor):
        if fe_s.numel() == 0:
            return
        self.store.append((ei_s.detach().cpu().clone(), fe_s.detach().cpu().clone()))
        if len(self.store) > self.capacity:
            self.store.pop(0)

    @torch.no_grad()
    def sample(self, count: int = 1):
        if not self.store:
            return []
        count = min(count, len(self.store))
        return self.store[-count:]

# ---------- Small helpers ----------
def ce_from_logits_binary(logits: torch.Tensor, labels_long: torch.Tensor) -> torch.Tensor:
    """
    logits: (N, 2) or (N,) if already a single logit; labels_long: (N,)
    Returns mean CE over provided logits/labels (expects you to slice indices outside).
    """
    if logits.dim() == 2 and logits.size(-1) == 2:
        return F.cross_entropy(logits, labels_long)
    y = labels_long.float()
    return F.binary_cross_entropy_with_logits(logits, y)

def prob_pos_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Map logits to P(y=1). Works for 2-class logits or single logit.
    """
    if logits.dim() == 2 and logits.size(-1) == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    return torch.sigmoid(logits)
