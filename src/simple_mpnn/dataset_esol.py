#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.smiles_to_graph import smiles_to_pyg


# ============================================================
# ESOL Dataset
# ============================================================

class ESOLDataset(Dataset):
    """Load ESOL CSV rows and convert each SMILES into a PyG graph.

    Expected default columns:
        smiles
        measured log solubility in mols per litre

    Each item is a torch_geometric.data.Data object containing:
        x:          [N_atom, 8]
        edge_index: [2, N_edge]
        edge_attr:  [N_edge, 4]
        y:          [1]
        smiles:     original SMILES string
    """

    def __init__(
        self,
        csv_path: str | Path,
        smiles_column: str = "smiles",
        target_column: str = "measured log solubility in mols per litre",
        explicit_h: bool = False,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.explicit_h = explicit_h

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_path}"
            )

        df = pd.read_csv(
            self.csv_path
        )

        required_columns = {
            self.smiles_column,
            self.target_column,
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required CSV columns: "
                f"{sorted(missing_columns)}"
            )

        # Keep only rows required for the current regression task.
        df = (
            df[
                [
                    self.smiles_column,
                    self.target_column,
                ]
            ]
            .dropna()
            .reset_index(drop=True)
        )

        self.graphs = []

        for row_idx, row in df.iterrows():

            smiles = str(
                row[self.smiles_column]
            )

            target = float(
                row[self.target_column]
            )

            try:

                _, data = smiles_to_pyg(
                    smiles,
                    explicit_h=self.explicit_h,
                )

            except ValueError as exc:

                raise ValueError(
                    f"Failed to preprocess row {row_idx}: "
                    f"SMILES={smiles!r}"
                ) from exc

            # Graph-level regression target.
            # One molecule: y shape [1]
            # After PyG batching: batch.y shape [B]
            data.y = torch.tensor(
                [target],
                dtype=torch.float32,
            )

            data.row_idx = int(
                row_idx
            )

            self.graphs.append(
                data
            )

    def __len__(
        self,
    ) -> int:

        return len(
            self.graphs
        )

    def __getitem__(
        self,
        idx: int,
    ):

        return self.graphs[
            idx
        ]
