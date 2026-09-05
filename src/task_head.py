#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Molecule-level Regression Head
# ============================================================

class RegressionHead(nn.Module):
    """Predict one continuous molecular property from H_mol.

    Input:
        H_mol: [B, embed_dim]

    Output:
        prediction: [B]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        if embed_dim < 1:
            raise ValueError(
                "embed_dim must be >= 1"
            )

        if hidden_dim < 1:
            raise ValueError(
                "hidden_dim must be >= 1"
            )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        H_mol: torch.Tensor,
    ) -> torch.Tensor:

        if H_mol.ndim != 2:

            raise ValueError(
                "H_mol must have shape "
                "[B, embed_dim]"
            )

        if (
            H_mol.size(1)
            != self.embed_dim
        ):

            raise ValueError(
                "H_mol embedding dimension mismatch. "
                f"Expected {self.embed_dim}, "
                f"got {H_mol.size(1)}"
            )

        prediction = self.mlp(
            H_mol
        )

        # [B, 1] -> [B]
        prediction = prediction.squeeze(
            -1
        )

        return prediction
