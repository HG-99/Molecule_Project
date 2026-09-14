#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Atom Feature Encoder
# ============================================================

class AtomFeatureEncoder(nn.Module):
    """Convert categorical atom features into dense embeddings.

    Expected x columns:
        0: atomic_num
        1: degree
        2: formal_charge
        3: num_H
        4: aromatic
        5: in_ring
        6: hybridization_id
        7: chirality_id

    Input:
        x: [N_atom, 8]

    Output:
        H_atom: [N_atom, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int = 128,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Atomic number
        #
        # 0     : reserved / unknown
        # 1~118 : actual elements
        # ----------------------------------------------------

        self.atomic_num_embedding = nn.Embedding(
            num_embeddings=119,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Degree
        # ----------------------------------------------------

        self.degree_embedding = nn.Embedding(
            num_embeddings=16,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Formal charge
        #
        # Supported:
        # -5, -4, ..., 0, ..., +4, +5
        #
        # Embedding IDs:
        #  0,  1, ..., 5, ...,  9, 10
        # ----------------------------------------------------

        self.charge_min = -5
        self.charge_max = 5

        self.formal_charge_embedding = nn.Embedding(
            num_embeddings=(
                self.charge_max
                - self.charge_min
                + 1
            ),
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Number of hydrogens
        # ----------------------------------------------------

        self.num_h_embedding = nn.Embedding(
            num_embeddings=16,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Aromatic
        #
        # 0 = False
        # 1 = True
        # ----------------------------------------------------

        self.aromatic_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Ring membership
        #
        # 0 = False
        # 1 = True
        # ----------------------------------------------------

        self.ring_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Hybridization
        #
        # 0 unspecified
        # 1 s
        # 2 sp
        # 3 sp2
        # 4 sp3
        # 5 sp3d
        # 6 sp3d2
        # ----------------------------------------------------

        self.hybridization_embedding = nn.Embedding(
            num_embeddings=7,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Chirality
        #
        # 0 unspecified
        # 1 CW
        # 2 CCW
        # 3 other
        # ----------------------------------------------------

        self.chirality_embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=embed_dim,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw atom features.

        Args:
            x:
                [N_atom, 8]

        Returns:
            H_atom:
                [N_atom, embed_dim]
        """

        if (
            x.ndim != 2
            or x.size(1) != 8
        ):
            raise ValueError(
                "Expected x shape "
                f"[N_atom, 8], "
                f"got {tuple(x.shape)}"
            )

        # ----------------------------------------------------
        # 1. Split categorical features
        # ----------------------------------------------------

        atomic_num = x[:, 0]
        degree = x[:, 1]
        formal_charge = x[:, 2]
        num_h = x[:, 3]
        aromatic = x[:, 4]
        in_ring = x[:, 5]
        hybridization = x[:, 6]
        chirality = x[:, 7]

        # ----------------------------------------------------
        # 2. Formal charge -> non-negative embedding index
        # ----------------------------------------------------

        if torch.any(
            formal_charge < self.charge_min
        ):
            raise ValueError(
                "Formal charge below "
                f"supported minimum "
                f"{self.charge_min}"
            )

        if torch.any(
            formal_charge > self.charge_max
        ):
            raise ValueError(
                "Formal charge above "
                f"supported maximum "
                f"{self.charge_max}"
            )

        charge_id = (
            formal_charge
            - self.charge_min
        )

        # ----------------------------------------------------
        # 3. Categorical embeddings
        # ----------------------------------------------------

        h_atomic_num = (
            self.atomic_num_embedding(
                atomic_num
            )
        )

        h_degree = (
            self.degree_embedding(
                degree
            )
        )

        h_charge = (
            self.formal_charge_embedding(
                charge_id
            )
        )

        h_num_h = (
            self.num_h_embedding(
                num_h
            )
        )

        h_aromatic = (
            self.aromatic_embedding(
                aromatic
            )
        )

        h_ring = (
            self.ring_embedding(
                in_ring
            )
        )

        h_hybridization = (
            self.hybridization_embedding(
                hybridization
            )
        )

        h_chirality = (
            self.chirality_embedding(
                chirality
            )
        )

        # ----------------------------------------------------
        # 4. Feature fusion
        # ----------------------------------------------------

        H_atom = (
            h_atomic_num
            + h_degree
            + h_charge
            + h_num_h
            + h_aromatic
            + h_ring
            + h_hybridization
            + h_chirality
        )

        return H_atom


# ============================================================
# Bond Feature Encoder
# ============================================================

class BondFeatureEncoder(nn.Module):
    """Convert categorical bond features into dense embeddings.

    Expected edge_attr columns:
        0: bond_type_id
        1: conjugated
        2: in_ring
        3: stereo_id

    Input:
        edge_attr: [N_edge, 4]

    Output:
        H_edge: [N_edge, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int = 128,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Bond type
        #
        # 0 single
        # 1 double
        # 2 triple
        # 3 aromatic
        # 4 unknown / fallback
        # ----------------------------------------------------

        self.bond_type_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Conjugated
        # ----------------------------------------------------

        self.conjugated_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Ring membership
        # ----------------------------------------------------

        self.ring_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embed_dim,
        )

        # ----------------------------------------------------
        # Stereo
        #
        # 0 NONE
        # 1 ANY
        # 2 Z
        # 3 E
        # 4 CIS
        # 5 TRANS
        # ----------------------------------------------------

        self.stereo_embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=embed_dim,
        )


    def forward(
        self,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw bond features.

        Args:
            edge_attr:
                [N_edge, 4]

        Returns:
            H_edge:
                [N_edge, embed_dim]
        """

        if (
            edge_attr.ndim != 2
            or edge_attr.size(1) != 4
        ):
            raise ValueError(
                "Expected edge_attr shape "
                f"[N_edge, 4], "
                f"got {tuple(edge_attr.shape)}"
            )

        # ----------------------------------------------------
        # 1. Split categorical features
        # ----------------------------------------------------

        bond_type = edge_attr[:, 0]
        conjugated = edge_attr[:, 1]
        in_ring = edge_attr[:, 2]
        stereo = edge_attr[:, 3]

        # ----------------------------------------------------
        # 2. Categorical embeddings
        # ----------------------------------------------------

        e_bond_type = (
            self.bond_type_embedding(
                bond_type
            )
        )

        e_conjugated = (
            self.conjugated_embedding(
                conjugated
            )
        )

        e_ring = (
            self.ring_embedding(
                in_ring
            )
        )

        e_stereo = (
            self.stereo_embedding(
                stereo
            )
        )

        # ----------------------------------------------------
        # 3. Feature fusion
        # ----------------------------------------------------

        H_edge = (
            e_bond_type
            + e_conjugated
            + e_ring
            + e_stereo
        )

        return H_edge