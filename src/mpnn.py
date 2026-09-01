#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Simple MPNN Layer
# ============================================================

class SimpleMPNNLayer(nn.Module):
    """One simple node-state MPNN layer.

    Inputs:
        H_atom:
            [N_atom, embed_dim]

        edge_index:
            [2, N_edge]

        H_edge:
            [N_edge, embed_dim]

    Output:
        H_atom_updated:
            [N_atom, embed_dim]

    Message:
        m_(j->i) = MessageMLP([h_j || e_ji])

    Aggregation:
        m_i = SUM_j m_(j->i)

    Update:
        h_i' = LayerNorm(
            h_i + UpdateMLP([h_i || m_i])
        )
    """

    def __init__(
        self,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Message function
        #
        # [source atom || edge]
        #
        # [D] + [D]
        #     ↓ concat
        # [2D]
        #     ↓
        # MLP
        #     ↓
        # [D]
        # ----------------------------------------------------

        self.message_mlp = nn.Sequential(
            nn.Linear(
                2 * embed_dim,
                embed_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
        )

        # ----------------------------------------------------
        # Update function
        #
        # [current atom || aggregated message]
        #
        # [D] + [D]
        #     ↓ concat
        # [2D]
        #     ↓
        # MLP
        #     ↓
        # [D]
        # ----------------------------------------------------

        self.update_mlp = nn.Sequential(
            nn.Linear(
                2 * embed_dim,
                embed_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )


    def forward(
        self,
        H_atom: torch.Tensor,
        edge_index: torch.Tensor,
        H_edge: torch.Tensor,
    ) -> torch.Tensor:

        # ====================================================
        # 1. Basic shape checks
        # ====================================================

        if H_atom.ndim != 2:
            raise ValueError(
                "H_atom must have shape "
                "[N_atom, embed_dim]"
            )

        if H_edge.ndim != 2:
            raise ValueError(
                "H_edge must have shape "
                "[N_edge, embed_dim]"
            )

        if (
            edge_index.ndim != 2
            or edge_index.size(0) != 2
        ):
            raise ValueError(
                "edge_index must have shape "
                "[2, N_edge]"
            )

        if (
            H_atom.size(1)
            != self.embed_dim
        ):
            raise ValueError(
                "H_atom embedding dimension mismatch"
            )

        if (
            H_edge.size(1)
            != self.embed_dim
        ):
            raise ValueError(
                "H_edge embedding dimension mismatch"
            )

        if (
            edge_index.size(1)
            != H_edge.size(0)
        ):
            raise ValueError(
                "Number of edge_index columns "
                "must equal number of H_edge rows"
            )


        # ====================================================
        # 2. Read source / destination node indices
        # ====================================================

        src = edge_index[0]
        dst = edge_index[1]

        # Example:
        #
        # src = [0, 1, 1, 2, 1, 3]
        # dst = [1, 0, 2, 1, 3, 1]


        # ====================================================
        # 3. Source atom representations
        # ====================================================

        H_src = H_atom[src]

        # shape:
        #
        # [N_edge, embed_dim]


        # ====================================================
        # 4. Create messages
        # ====================================================

        message_input = torch.cat(
            [
                H_src,
                H_edge,
            ],
            dim=-1,
        )

        # shape:
        #
        # [N_edge, 2 * embed_dim]

        messages = self.message_mlp(
            message_input
        )

        # shape:
        #
        # [N_edge, embed_dim]


        # ====================================================
        # 5. Aggregate messages at destination atoms
        # ====================================================

        aggregated = torch.zeros_like(
            H_atom
        )

        # For each directed edge:
        #
        # message[e]
        #
        # goes to:
        #
        # destination = dst[e]

        aggregated.index_add_(
            0,
            dst,
            messages,
        )

        # shape:
        #
        # [N_atom, embed_dim]


        # ====================================================
        # 6. Update atom representations
        # ====================================================

        update_input = torch.cat(
            [
                H_atom,
                aggregated,
            ],
            dim=-1,
        )

        delta = self.update_mlp(
            update_input
        )

        delta = self.dropout(
            delta
        )

        # Residual connection
        H_atom_updated = self.norm(
            H_atom + delta
        )

        return H_atom_updated


# ============================================================
# Multi-layer MPNN
# ============================================================

class SimpleMPNN(nn.Module):
    """Stack multiple SimpleMPNNLayer blocks."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "num_layers must be >= 1"
            )

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            SimpleMPNNLayer(
                embed_dim=embed_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])


    def forward(
        self,
        H_atom: torch.Tensor,
        edge_index: torch.Tensor,
        H_edge: torch.Tensor,
        return_all_layers: bool = False,
    ):

        H = H_atom

        layer_outputs = []

        for layer in self.layers:

            H = layer(
                H,
                edge_index,
                H_edge,
            )

            layer_outputs.append(
                H
            )

        if return_all_layers:
            return H, layer_outputs

        return H