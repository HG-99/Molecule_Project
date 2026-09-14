#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset split
# ============================================================

def make_split_indices(
    dataset_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:

    if dataset_size < 3:
        raise ValueError(
            "Dataset must contain at least 3 molecules."
        )

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            "train_ratio + val_ratio must be < 1.0."
        )

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    n_train = int(
        dataset_size * train_ratio
    )

    n_val = int(
        dataset_size * val_ratio
    )

    # Ensure every split has at least one sample.
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    if n_train + n_val >= dataset_size:
        n_val = max(
            1,
            dataset_size - n_train - 1,
        )

    train_indices = indices[
        :n_train
    ]

    val_indices = indices[
        n_train:n_train + n_val
    ]

    test_indices = indices[
        n_train + n_val:
    ]

    return (
        train_indices,
        val_indices,
        test_indices,
    )


# ============================================================
# Metrics accumulator
# ============================================================

class RegressionMeter:

    def __init__(self) -> None:
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.num_samples = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:

        prediction = prediction.detach()
        target = target.detach()

        error = (
            prediction - target
        )

        self.sum_squared_error += (
            error.pow(2)
            .sum()
            .item()
        )

        self.sum_absolute_error += (
            error.abs()
            .sum()
            .item()
        )

        self.num_samples += (
            target.numel()
        )

    def compute(self) -> dict[str, float]:

        if self.num_samples == 0:
            raise RuntimeError(
                "No samples were accumulated."
            )

        mse = (
            self.sum_squared_error
            / self.num_samples
        )

        rmse = math.sqrt(mse)

        mae = (
            self.sum_absolute_error
            / self.num_samples
        )

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }


# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(
    encoder: MoleculeEncoder,
    head: RegressionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:

    encoder.train()
    head.train()

    meter = RegressionMeter()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:

        batch = batch.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
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
                "ESOL is a molecule-level task, so readout must not be 'none'."
            )

        prediction = head(
            H_mol
        )

        target = batch.y.view(
            -1
        )

        if prediction.shape != target.shape:
            raise RuntimeError(
                "Prediction/target shape mismatch: "
                f"pred={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )

        loss = criterion(
            prediction,
            target,
        )

        loss.backward()

        optimizer.step()

        current_B = batch.num_graphs

        total_loss += (
            loss.item()
            * current_B
        )

        total_graphs += (
            current_B
        )

        meter.update(
            prediction,
            target,
        )

    metrics = meter.compute()

    metrics[
        "loss"
    ] = (
        total_loss
        / total_graphs
    )

    return metrics


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    encoder: MoleculeEncoder,
    head: RegressionHead,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:

    encoder.eval()
    head.eval()

    meter = RegressionMeter()

    total_loss = 0.0
    total_graphs = 0

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
                "ESOL is a molecule-level task, so readout must not be 'none'."
            )

        prediction = head(
            H_mol
        )

        target = batch.y.view(
            -1
        )

        loss = criterion(
            prediction,
            target,
        )

        current_B = batch.num_graphs

        total_loss += (
            loss.item()
            * current_B
        )

        total_graphs += (
            current_B
        )

        meter.update(
            prediction,
            target,
        )

    metrics = meter.compute()

    metrics[
        "loss"
    ] = (
        total_loss
        / total_graphs
    )

    return metrics


# ============================================================
# CSV logging
# ============================================================

def initialize_metrics_csv(
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_rmse",
            "train_mae",
            "val_loss",
            "val_rmse",
            "val_mae",
        ])


def append_metrics_csv(
    path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            train_metrics["loss"],
            train_metrics["rmse"],
            train_metrics["mae"],
            val_metrics["loss"],
            val_metrics["rmse"],
            val_metrics["mae"],
        ])


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the current MoleculeEncoder + RegressionHead "
            "on the ESOL solubility regression task."
        )
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="dataset/esol/delaney-processed.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/esol",
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--mpnn-layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--mpnn-dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--readout",
        type=str,
        choices=[
            "mean",
            "sum",
            "max",
        ],
        default="mean",
    )

    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
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
    # Device / seed
    # ========================================================

    set_seed(
        args.seed
    )

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
    # Dataset
    # ========================================================

    dataset = ESOLDataset(
        csv_path=args.csv,
    )

    (
        train_indices,
        val_indices,
        test_indices,
    ) = make_split_indices(
        dataset_size=len(dataset),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    val_dataset = Subset(
        dataset,
        val_indices,
    )

    test_dataset = Subset(
        dataset,
        test_indices,
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": (
            device.type == "cuda"
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )


    # ========================================================
    # Model
    # ========================================================

    encoder = MoleculeEncoder(
        embed_dim=args.embed_dim,
        num_layers=args.mpnn_layers,
        dropout=args.mpnn_dropout,
        readout=args.readout,
    ).to(
        device
    )

    head = RegressionHead(
        embed_dim=args.embed_dim,
        hidden_dim=args.head_hidden_dim,
        dropout=args.head_dropout,
    ).to(
        device
    )


    # ========================================================
    # Optimization
    # ========================================================

    parameters = list(
        encoder.parameters()
    ) + list(
        head.parameters()
    )

    optimizer = torch.optim.Adam(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()


    # ========================================================
    # Output paths
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best_model.pt"
    )

    metrics_path = (
        output_dir
        / "metrics.csv"
    )

    initialize_metrics_csv(
        metrics_path
    )


    # ========================================================
    # Training summary
    # ========================================================

    print(
        "=" * 70
    )
    print(
        "ESOL TRAINING"
    )
    print(
        "=" * 70
    )

    print(
        f"device      : {device}"
    )
    print(
        f"dataset     : {len(dataset)} molecules"
    )
    print(
        f"train       : {len(train_dataset)}"
    )
    print(
        f"validation  : {len(val_dataset)}"
    )
    print(
        f"test        : {len(test_dataset)}"
    )
    print(
        f"batch size  : {args.batch_size}"
    )
    print(
        f"embed dim   : {args.embed_dim}"
    )
    print(
        f"MPNN layers : {args.mpnn_layers}"
    )
    print(
        f"readout     : {args.readout}"
    )


    # ========================================================
    # Training loop
    # ========================================================

    best_val_rmse = float(
        "inf"
    )

    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = train_one_epoch(
            encoder=encoder,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            encoder=encoder,
            head=head,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        append_metrics_csv(
            path=metrics_path,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train RMSE={train_metrics['rmse']:.4f} "
            f"MAE={train_metrics['mae']:.4f} | "
            f"val RMSE={val_metrics['rmse']:.4f} "
            f"MAE={val_metrics['mae']:.4f}"
        )

        if (
            val_metrics["rmse"]
            < best_val_rmse
        ):

            best_val_rmse = (
                val_metrics["rmse"]
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "encoder_state_dict": (
                    encoder.state_dict()
                ),
                "head_state_dict": (
                    head.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "epoch": epoch,
                "best_val_rmse": (
                    best_val_rmse
                ),
                "config": {
                    "embed_dim": args.embed_dim,
                    "mpnn_layers": args.mpnn_layers,
                    "mpnn_dropout": args.mpnn_dropout,
                    "readout": args.readout,
                    "head_hidden_dim": args.head_hidden_dim,
                    "head_dropout": args.head_dropout,
                    "seed": args.seed,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                },
                "split_indices": {
                    "train": train_indices,
                    "val": val_indices,
                    "test": test_indices,
                },
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

        else:

            epochs_without_improvement += 1

        if (
            args.patience > 0
            and epochs_without_improvement
            >= args.patience
        ):

            print(
                f"Early stopping at epoch {epoch}."
            )
            break


    # ========================================================
    # Test with best checkpoint
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
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

    test_metrics = evaluate(
        encoder=encoder,
        head=head,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "TRAINING COMPLETE"
    )
    print(
        "=" * 70
    )

    print(
        f"best epoch    : {best_epoch}"
    )
    print(
        f"best val RMSE : {best_val_rmse:.4f}"
    )
    print(
        f"test RMSE     : {test_metrics['rmse']:.4f}"
    )
    print(
        f"test MAE      : {test_metrics['mae']:.4f}"
    )
    print(
        f"checkpoint    : {checkpoint_path}"
    )
    print(
        f"metrics       : {metrics_path}"
    )


if __name__ == "__main__":
    main()
