#!/usr/bin/env python3
"""2D + 3D Molecule Encoder for QM9.

Current pipeline:

    x
      -> DynamicAtomFeatureEncoder
      -> H_atom [N, D]

    edge_attr
      -> DynamicBondFeatureEncoder
      -> H_edge [E, D]

    pos + edge_index
      -> GeometryEncoder
      -> H_geo [E, D]

    H_atom + edge_index + H_edge + H_geo
      -> GeometryAwareMPNN
      -> H_atom_updated [N, D]

    H_atom_updated
      -> GraphReadout
      -> H_mol [B, D]

Important:
At this stage H_edge and H_geo are defined on the SAME edge_index.
This is geometry-aware chemical-bond message passing, not yet a separate
chemical-graph + spatial-radius-graph architecture.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .feature_schema import (
    DEFAULT_ATOM_FEATURE_SCHEMA,
    DEFAULT_BOND_FEATURE_SCHEMA,
    FeatureSpec,
)
from .atom_feature_encoder import (
    DynamicAtomFeatureEncoder,
)
from .bond_feature_encoder import (
    DynamicBondFeatureEncoder,
)
from .geometry_encoder import (
    GeometryEncoder,
)
from .mpnn import (
    GeometryAwareMPNN,
)
from .readout import (
    GraphReadout,
)


class MoleculeEncoder(nn.Module):
    """Combine atom, bond, geometry, MPNN, and graph readout."""

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
        atom_feature_schema: Mapping[
            str,
            FeatureSpec,
        ] | None = None,
        bond_feature_schema: Mapping[
            str,
            FeatureSpec,
        ] | None = None,
        atom_fusion: str = "mean",
        bond_fusion: str = "mean",
        num_rbf: int = 32,
        cutoff: float = 5.0,
        use_cutoff_envelope: bool = True,
        trainable_rbf: bool = False,
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
                f"Unsupported readout mode: {readout!r}. "
                f"Choose from {sorted(self.SUPPORTED_READOUTS)}"
            )

        if atom_feature_schema is None:
            atom_feature_schema = (
                DEFAULT_ATOM_FEATURE_SCHEMA
            )

        if bond_feature_schema is None:
            bond_feature_schema = (
                DEFAULT_BOND_FEATURE_SCHEMA
            )

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.readout_mode = readout

        # 1. Raw atom features -> H_atom
        self.atom_encoder = DynamicAtomFeatureEncoder(
            embed_dim=embed_dim,
            feature_schema=atom_feature_schema,
            fusion=atom_fusion,
            dropout=dropout,
        )

        # 2. Raw chemical bond features -> H_edge
        self.bond_encoder = DynamicBondFeatureEncoder(
            embed_dim=embed_dim,
            feature_schema=bond_feature_schema,
            fusion=bond_fusion,
            dropout=dropout,
        )

        # 3. 3D coordinates -> H_geo
        self.geometry_encoder = GeometryEncoder(
            embed_dim=embed_dim,
            num_rbf=num_rbf,
            cutoff=cutoff,
            use_cutoff_envelope=use_cutoff_envelope,
            dropout=dropout,
            trainable_rbf=trainable_rbf,
        )

        # 4. [h_j || e_ji || g_ji] message passing
        self.mpnn = GeometryAwareMPNN(
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # 5. Atom states -> molecule state
        if readout == "none":
            self.readout = None
        else:
            self.readout = GraphReadout(
                mode=readout
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> dict[str, object]:
        """Encode one molecule or a batch of molecules."""

        # ====================================================
        # 1. Feature encoding
        # ====================================================

        H_atom = self.atom_encoder(
            x
        )

        H_edge = self.bond_encoder(
            edge_attr
        )

        # ====================================================
        # 2. Geometry encoding
        # ====================================================

        geometry = self.geometry_encoder(
            pos=pos,
            edge_index=edge_index,
            return_geometry=True,
        )

        H_geo = geometry[
            "H_geo"
        ]

        # ====================================================
        # 3. Geometry-aware message passing
        # ====================================================

        if return_all_layers:
            (
                H_atom_updated,
                layer_outputs,
            ) = self.mpnn(
                H_atom=H_atom,
                edge_index=edge_index,
                H_edge=H_edge,
                H_geo=H_geo,
                return_all_layers=True,
            )
        else:
            H_atom_updated = self.mpnn(
                H_atom=H_atom,
                edge_index=edge_index,
                H_edge=H_edge,
                H_geo=H_geo,
                return_all_layers=False,
            )

            layer_outputs = None

        # ====================================================
        # 4. Readout
        # ====================================================

        if self.readout is None:
            H_mol = None
        else:
            H_mol = self.readout(
                H_atom_updated,
                batch,
            )

        # ====================================================
        # 5. Return intermediate states for study/debugging
        # ====================================================

        return {
            "H_atom": H_atom,
            "H_edge": H_edge,
            "H_geo": H_geo,
            "distance": geometry["distance"],
            "displacement": geometry["displacement"],
            "direction": geometry["direction"],
            "rbf": geometry["rbf"],
            "cutoff_envelope": geometry[
                "cutoff_envelope"
            ],
            "H_atom_updated": H_atom_updated,
            "H_mol": H_mol,
            "layer_outputs": layer_outputs,
        }
