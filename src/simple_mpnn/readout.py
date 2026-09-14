#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn

from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


# ============================================================
# Graph Readout / Pooling
# ============================================================

class GraphReadout(nn.Module):
    """Convert atom-level representations into molecule-level representations.

    Input:
        H_atom:
            [N_total_atom, embed_dim]

        batch:
            [N_total_atom]

            Each value indicates which molecule
            the atom belongs to.

            Example:

                batch = [0, 0, 0, 0, 1, 1, 1]

            means:

                atoms 0~3 -> molecule 0
                atoms 4~6 -> molecule 1

            For a single molecule, batch can be None.

    Output:
        H_mol:
            [N_molecule, embed_dim]
    """

    def __init__(
        self,
        mode: str = "mean",
    ):
        super().__init__()

        supported_modes = {
            "mean",
            "sum",
            "max",
        }

        if mode not in supported_modes:
            raise ValueError(
                f"Unsupported readout mode: {mode}. "
                f"Choose from {sorted(supported_modes)}"
            )

        self.mode = mode


    def forward(
        self,
        H_atom: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # ====================================================
        # 1. Shape check
        # ====================================================

        if H_atom.ndim != 2:

            raise ValueError(
                "H_atom must have shape "
                "[N_atom, embed_dim]"
            )


        # ====================================================
        # 2. Single molecule case
        # ====================================================

        if batch is None:

            batch = torch.zeros(
                H_atom.size(0),
                dtype=torch.long,
                device=H_atom.device,
            )


        # ====================================================
        # 3. Batch check
        # ====================================================

        if (
            batch.ndim != 1
            or batch.size(0) != H_atom.size(0)
        ):

            raise ValueError(
                "batch must have shape [N_atom]"
            )


        # ====================================================
        # 4. Graph-level readout
        # ====================================================

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