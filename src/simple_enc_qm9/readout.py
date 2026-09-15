#!/usr/bin/env python3
"""Graph readout for the QM9 molecule encoder.

This intentionally mirrors src/simple_mpnn/readout.py.

Geometry information has already influenced atom states through the
geometry-aware MPNN, so readout pools the final atom representations:

    H_atom_updated [N_total_atom, D]
        -> GraphReadout
        -> H_mol [B, D]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


class GraphReadout(nn.Module):
    """Convert atom-level representations into molecule representations."""

    SUPPORTED_MODES = {
        "mean",
        "sum",
        "max",
    }

    def __init__(
        self,
        mode: str = "mean",
    ):
        super().__init__()

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported readout mode: {mode!r}. "
                f"Choose from {sorted(self.SUPPORTED_MODES)}"
            )

        self.mode = mode

    def forward(
        self,
        H_atom: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if H_atom.ndim != 2:
            raise ValueError(
                "H_atom must have shape [N_atom, embed_dim]"
            )

        # Single molecule -> every atom belongs to graph 0.
        if batch is None:
            batch = torch.zeros(
                H_atom.size(0),
                dtype=torch.long,
                device=H_atom.device,
            )

        if (
            batch.ndim != 1
            or batch.size(0) != H_atom.size(0)
        ):
            raise ValueError(
                "batch must have shape [N_atom]"
            )

        if batch.dtype != torch.long:
            raise ValueError(
                "batch must have dtype torch.long"
            )

        if self.mode == "mean":
            H_mol = global_mean_pool(
                H_atom,
                batch,
            )

        elif self.mode == "sum":
            H_mol = global_add_pool(
                H_atom,
                batch,
            )

        elif self.mode == "max":
            H_mol = global_max_pool(
                H_atom,
                batch,
            )

        else:
            raise RuntimeError(
                "Unexpected readout mode"
            )

        return H_mol
