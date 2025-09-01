import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_default_adjacency(
    num_nodes: int = 14, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Construct a sparse physiology graph adjacency mask A in {0,1}^{N x N}.

    Node order matches expert state order used across the project:
    [0] p_a, [1] p_v, [2] s_reflex, [3] sv, [4] r_tpr_mod,
    [5] f_hr_max, [6] f_hr_min, [7] r_tpr_max, [8] r_tpr_min,
    [9] ca, [10] cv, [11] k_width, [12] p_aset, [13] tau
    """
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    # Hemodynamic coupling via r_tpr, inflow sv*f_hr
    # p_a <-> p_v
    A[0, 1] = 1.0
    A[1, 0] = 1.0

    # sv -> p_a, p_v (via flows)
    A[0, 3] = 1.0
    A[1, 3] = 1.0

    # f_hr_* -> p_a, p_v; s_reflex -> f_hr_*
    A[0, 5] = 1.0
    A[0, 6] = 1.0
    A[1, 5] = 1.0
    A[1, 6] = 1.0
    A[5, 2] = 1.0
    A[6, 2] = 1.0

    # r_tpr components -> p_a, p_v; s_reflex -> r_tpr_*
    for src in [4, 7, 8]:
        A[0, src] = 1.0
        A[1, src] = 1.0
    A[7, 2] = 1.0
    A[8, 2] = 1.0

    # Compliance coupling
    A[0, 9] = 1.0  # ca -> p_a
    A[1, 10] = 1.0  # cv -> p_v

    # Reflex inputs
    A[2, 0] = 1.0  # p_a -> s_reflex
    A[2, 11] = 1.0  # k_width -> s_reflex
    A[2, 12] = 1.0  # p_aset -> s_reflex
    A[2, 13] = 1.0  # tau -> s_reflex

    # Modulators back to p_a
    for src in [11, 12, 13]:
        A[0, src] = 1.0

    # Self-loops for stability
    A.fill_diagonal_(1.0)

    if device is not None:
        A = A.to(device)
    return A


class SimpleGATLayer(nn.Module):
    """Lightweight GAT layer for dense adjacency masks.

    Inputs: node_features [B, N, Fin], adjacency [N, N] binary mask
    Output: node_features [B, N, Fout]
    """

    def __init__(
        self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        # Residual projection to match dimensions when in_dim != out_dim
        self.residual_proj = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Compute scaled dot-product attention with adjacency mask
        q = q.permute(0, 2, 1, 3)  # [B, H, N, D]
        k = k.permute(0, 2, 1, 3)  # [B, H, N, D]
        v = v.permute(0, 2, 1, 3)  # [B, H, N, D]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, H, N, N]

        # Apply adjacency mask: set -inf where no edge
        mask = (adj > 0).unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.out_dim)
        out = self.out_proj(out)

        # Residual + norm
        residual = self.residual_proj(x)
        out = self.layer_norm(residual + out)
        return out


class GraphControlNet(nn.Module):
    """GAT-based controller that maps node-wise features to 4 control heads.

    Expected node order (N=14):
    [p_a, p_v, s_reflex, sv, r_tpr_mod, f_hr_max, f_hr_min, r_tpr_max, r_tpr_min, ca, cv, k_width, p_aset, tau]

    forward inputs:
      - node_features: [B, N, F]
      - adjacency: optional [N, N] mask; if None, use internal default
    returns:
      - controls: [B, 4]
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = 14
        self.device = device

        layers = []
        in_dim = node_feature_dim
        for _ in range(num_layers):
            layers.append(
                SimpleGATLayer(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
            )
            in_dim = hidden_dim
        self.gnn = nn.ModuleList(layers)

        # Readout MLPs for each control from designated nodes
        self.readout_u1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.readout_u2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.readout_u3 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.readout_u4 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # Default adjacency
        A = build_default_adjacency(self.num_nodes, device)
        self.register_buffer("adjacency", A, persistent=False)

    @staticmethod
    def control_node_indices() -> Tuple[int, int, int, int]:
        # u1: +dpv_dt from p_v, u2: dsv_dt from sv, u3: dca/dt from ca, u4: d(r_tpr_mod)/dt from r_tpr_mod
        return 1, 3, 9, 4

    def forward(
        self, node_features: torch.Tensor, adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # node_features: [B, N, F]
        B, N, _ = node_features.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"
        A = adjacency if adjacency is not None else self.adjacency

        x = node_features
        for layer in self.gnn:
            x = layer(x, A)

        # Global context via mean pool
        global_ctx = x.mean(dim=1)  # [B, H]
        v_idx, sv_idx, ca_idx, rmod_idx = self.control_node_indices()

        def cat_ctx(node_idx: int) -> torch.Tensor:
            return torch.cat([x[:, node_idx, :], global_ctx], dim=-1)

        u1 = self.readout_u1(cat_ctx(v_idx))
        u2 = self.readout_u2(cat_ctx(sv_idx))
        u3 = self.readout_u3(cat_ctx(ca_idx))
        u4 = self.readout_u4(cat_ctx(rmod_idx))

        u = torch.cat([u1, u2, u3, u4], dim=-1)
        u = torch.tanh(u)  # keep bounded; scales applied by caller
        return u
