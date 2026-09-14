#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Import project modules
# ============================================================

from src.dataset import ESOLDataset
from src.molecule_encoder import MoleculeEncoder
from src.task_head import RegressionHead


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def run_test(
    encoder: MoleculeEncoder,
    head: RegressionHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    dict[str, float],
    list[dict[str, object]],
]:

    encoder.eval()
    head.eval()

    sum_squared_error = 0.0
    sum_absolute_error = 0.0
    num_samples = 0

    rows: list[dict[str, object]] = []

    for batch in loader:

        batch = batch.to(
            device
        )

        outputs = encoder(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )

        H_mol = outputs[
            "H_mol"
        ]

        if H_mol is None:
            raise RuntimeError(
                "ESOL evaluation requires a molecule-level readout."
            )

        prediction = head(
            H_mol
        )

        target = batch.y.view(
            -1
        )

        error = (
            prediction - target
        )

        sum_squared_error += (
            error.pow(2)
            .sum()
            .item()
        )

        sum_absolute_error += (
            error.abs()
            .sum()
            .item()
        )

        num_samples += (
            target.numel()
        )

        # PyG keeps string attributes as a Python list in a batch.
        smiles_batch = batch.smiles

        if isinstance(
            smiles_batch,
            str,
        ):
            smiles_batch = [
                smiles_batch
            ]

        if hasattr(
            batch,
            "row_idx",
        ):

            row_indices = (
                batch.row_idx
                .view(-1)
                .detach()
                .cpu()
                .tolist()
            )

        else:

            row_indices = [
                None
            ] * target.numel()

        prediction_cpu = (
            prediction
            .detach()
            .cpu()
            .tolist()
        )

        target_cpu = (
            target
            .detach()
            .cpu()
            .tolist()
        )

        for (
            row_idx,
            smiles,
            y_true,
            y_pred,
        ) in zip(
            row_indices,
            smiles_batch,
            target_cpu,
            prediction_cpu,
        ):

            rows.append({
                "row_idx": row_idx,
                "smiles": smiles,
                "target_logS": y_true,
                "predicted_logS": y_pred,
                "absolute_error": abs(
                    y_pred - y_true
                ),
            })

    mse = (
        sum_squared_error
        / num_samples
    )

    metrics = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": (
            sum_absolute_error
            / num_samples
        ),
    }

    return metrics, rows


# ============================================================
# Save predictions
# ============================================================

def save_predictions(
    path: Path,
    rows: list[dict[str, object]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "row_idx",
        "smiles",
        "target_logS",
        "predicted_logS",
        "absolute_error",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best ESOL MoleculeEncoder checkpoint "
            "on the held-out test split."
        )
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="dataset/esol/delaney-processed.csv",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/esol/best_model.pt",
    )

    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/esol/predictions.csv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    args = parser.parse_args()


    # ========================================================
    # Device
    # ========================================================

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        if (
            args.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        device = torch.device(
            args.device
        )


    # ========================================================
    # Load checkpoint
    # ========================================================

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint[
        "config"
    ]


    # ========================================================
    # Dataset / test split
    # ========================================================

    dataset = ESOLDataset(
        csv_path=args.csv,
    )

    test_indices = checkpoint[
        "split_indices"
    ][
        "test"
    ]

    test_dataset = Subset(
        dataset,
        test_indices,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )


    # ========================================================
    # Rebuild model from checkpoint config
    # ========================================================

    encoder = MoleculeEncoder(
        embed_dim=config[
            "embed_dim"
        ],
        num_layers=config[
            "mpnn_layers"
        ],
        dropout=config[
            "mpnn_dropout"
        ],
        readout=config[
            "readout"
        ],
    ).to(
        device
    )

    head = RegressionHead(
        embed_dim=config[
            "embed_dim"
        ],
        hidden_dim=config[
            "head_hidden_dim"
        ],
        dropout=config[
            "head_dropout"
        ],
    ).to(
        device
    )

    encoder.load_state_dict(
        checkpoint[
            "encoder_state_dict"
        ]
    )

    head.load_state_dict(
        checkpoint[
            "head_state_dict"
        ]
    )


    # ========================================================
    # Test
    # ========================================================

    metrics, rows = run_test(
        encoder=encoder,
        head=head,
        loader=test_loader,
        device=device,
    )

    predictions_path = Path(
        args.predictions
    )

    save_predictions(
        predictions_path,
        rows,
    )


    print(
        "=" * 70
    )
    print(
        "ESOL TEST EVALUATION"
    )
    print(
        "=" * 70
    )

    print(
        f"device         : {device}"
    )
    print(
        f"checkpoint     : {checkpoint_path}"
    )
    print(
        f"best epoch     : {checkpoint['epoch']}"
    )
    print(
        f"test molecules : {len(test_dataset)}"
    )
    print(
        f"test MSE       : {metrics['mse']:.4f}"
    )
    print(
        f"test RMSE      : {metrics['rmse']:.4f}"
    )
    print(
        f"test MAE       : {metrics['mae']:.4f}"
    )
    print(
        f"predictions    : {predictions_path}"
    )


if __name__ == "__main__":
    main()
