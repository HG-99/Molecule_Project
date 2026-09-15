#!/usr/bin/env python3
"""Geometry-aware MPNN for the QM9 molecule encoder.

This follows src/simple_mpnn/mpnn.py, but extends each message from

    [h_j || e_ji]

to

    [h_j || e_ji || g_ji]

where g_ji is the distance/RBF-based geometry latent from GeometryEncoder.
At the current stage, H_edge and H_geo use the same edge_index.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometryAwareMPNNLayer(nn.Module):
    """One geometry-aware MPNN layer."""

    def __init__(
        self,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")

        self.embed_dim = embed_dim

        # [h_j || e_ji || g_ji]: 3D -> D
        self.message_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # [h_i || m_i]: 2D -> D
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        H_atom: torch.Tensor,
        edge_index: torch.Tensor,
        H_edge: torch.Tensor,
        H_geo: torch.Tensor,
    ) -> torch.Tensor:

        if H_atom.ndim != 2:
            raise ValueError("H_atom must have shape [N_atom, D]")

        if H_edge.ndim != 2:
            raise ValueError("H_edge must have shape [N_edge, D]")

        if H_geo.ndim != 2:
            raise ValueError("H_geo must have shape [N_edge, D]")

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, N_edge]")

        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype torch.long")

        if H_atom.size(1) != self.embed_dim:
            raise ValueError("H_atom embedding dimension mismatch")

        if H_edge.size(1) != self.embed_dim:
            raise ValueError("H_edge embedding dimension mismatch")

        if H_geo.size(1) != self.embed_dim:
            raise ValueError("H_geo embedding dimension mismatch")

        num_edges = edge_index.size(1)

        if H_edge.size(0) != num_edges:
            raise ValueError(
                "H_edge rows must equal number of edge_index columns"
            )

        if H_geo.size(0) != num_edges:
            raise ValueError(
                "H_geo rows must equal number of edge_index columns"
            )

        src = edge_index[0]
        dst = edge_index[1]

        # Neighbor/source atom representation for each directed edge.
        H_src = H_atom[src]

        message_input = torch.cat(
            [H_src, H_edge, H_geo],
            dim=-1,
        )

        messages = self.message_mlp(message_input)
        messages = self.dropout(messages)

        # SUM aggregation at destination atoms.
        aggregated = torch.zeros_like(H_atom)
        aggregated.index_add_(0, dst, messages)

        update_input = torch.cat(
            [H_atom, aggregated],
            dim=-1,
        )

        delta = self.update_mlp(update_input)
        delta = self.dropout(delta)

        # Residual update, matching the SimpleMPNN design.
        H_atom_updated = self.norm(
            H_atom + delta
        )

        return H_atom_updated


class GeometryAwareMPNN(nn.Module):
    """Stack multiple GeometryAwareMPNNLayer blocks."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            GeometryAwareMPNNLayer(
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
        H_geo: torch.Tensor,
        return_all_layers: bool = False,
    ):
        H = H_atom
        layer_outputs = []

        for layer in self.layers:
            H = layer(
                H_atom=H,
                edge_index=edge_index,
                H_edge=H_edge,
                H_geo=H_geo,
            )
            layer_outputs.append(H)

        if return_all_layers:
            return H, layer_outputs

        return H
