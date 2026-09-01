#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn

from src.feature_encoder import (
    AtomFeatureEncoder,
    BondFeatureEncoder,
)

from src.mpnn import (
    SimpleMPNN,
)

from src.readout import (
    GraphReadout,
)


# ============================================================
# Molecule Encoder
# ============================================================

class MoleculeEncoder(nn.Module):
    """Combine feature encoding, message passing, and graph readout.

    Pipeline:
        x / edge_attr
            -> AtomFeatureEncoder / BondFeatureEncoder
            -> H_atom / H_edge

        H_atom + edge_index + H_edge
            -> SimpleMPNN
            -> H_atom_updated

        H_atom_updated
            -> GraphReadout (optional)
            -> H_mol

    Inputs:
        x:
            [N_total_atom, 8]

        edge_index:
            [2, N_total_edge]

        edge_attr:
            [N_total_edge, 4]

        batch:
            [N_total_atom]

            Each value indicates which molecule an atom belongs to.

            Example:
                batch = [0, 0, 0, 1, 1]

            means:
                atoms 0~2 -> molecule 0
                atoms 3~4 -> molecule 1

            For a single molecule, batch can be None.

    Outputs:
        Dictionary containing:

        H_atom:
            Atom embeddings before message passing.
            [N_total_atom, embed_dim]

        H_edge:
            Bond embeddings.
            [N_total_edge, embed_dim]

        H_atom_updated:
            Contextualized atom embeddings after MPNN.
            [N_total_atom, embed_dim]

        H_mol:
            Molecule-level representation after readout.
            [N_molecule, embed_dim]

            If readout="none", H_mol is None.

        layer_outputs:
            Optional list of atom representations from each MPNN layer.
    """

    SUPPORTED_READOUTS = {
        "mean",
        "sum",
        "max",
        "none",
    }

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        readout: str = "mean",
    ):
        super().__init__()

        if embed_dim < 1:
            raise ValueError(
                "embed_dim must be >= 1"
            )

        if num_layers < 1:
            raise ValueError(
                "num_layers must be >= 1"
            )

        if readout not in self.SUPPORTED_READOUTS:
            raise ValueError(
                f"Unsupported readout mode: {readout}. "
                f"Choose from {sorted(self.SUPPORTED_READOUTS)}"
            )

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout_mode = readout


        # ====================================================
        # 1. Feature Encoders
        # ====================================================

        self.atom_encoder = AtomFeatureEncoder(
            embed_dim=embed_dim,
        )

        self.bond_encoder = BondFeatureEncoder(
            embed_dim=embed_dim,
        )


        # ====================================================
        # 2. Message Passing
        # ====================================================

        self.mpnn = SimpleMPNN(
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


        # ====================================================
        # 3. Readout / Pooling
        # ====================================================

        if readout == "none":

            self.readout = None

        else:

            self.readout = GraphReadout(
                mode=readout,
            )


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> dict[str, object]:
        """Encode one molecule or a batch of molecules."""

        # ====================================================
        # 1. Feature Encoding
        # ====================================================

        H_atom = self.atom_encoder(
            x
        )

        H_edge = self.bond_encoder(
            edge_attr
        )


        # ====================================================
        # 2. Message Passing
        # ====================================================

        if return_all_layers:

            H_atom_updated, layer_outputs = self.mpnn(
                H_atom,
                edge_index,
                H_edge,
                return_all_layers=True,
            )

        else:

            H_atom_updated = self.mpnn(
                H_atom,
                edge_index,
                H_edge,
                return_all_layers=False,
            )

            layer_outputs = None


        # ====================================================
        # 3. Readout / Pooling
        # ====================================================

        if self.readout is None:

            H_mol = None

        else:

            H_mol = self.readout(
                H_atom_updated,
                batch,
            )


        # ====================================================
        # 4. Output
        # ====================================================

        return {
            "H_atom": H_atom,
            "H_edge": H_edge,
            "H_atom_updated": H_atom_updated,
            "H_mol": H_mol,
            "layer_outputs": layer_outputs,
        }
