#!/usr/bin/env python3
"""Integration test for the current simple_enc_qm9 pipeline.

This script tests the modules implemented so far:

    QM9Dataset
        │
        ├── x
        │     ↓
        │  DynamicAtomFeatureEncoder
        │     ↓
        │  H_atom [N, D]
        │
        ├── edge_attr
        │     ↓
        │  DynamicBondFeatureEncoder
        │     ↓
        │  H_edge [E, D]
        │
        └── pos + edge_index
              ↓
           GeometryEncoder
              ↓
        distance / direction / RBF
              ↓
           H_geo [E, D]

No message passing, readout, task head, or training is performed here.

Recommended location:

    main/simple_enc_qm9/pipeline_test.py

Run from the repository root:

    python main/simple_enc_qm9/pipeline_test.py

Example:

    python main/simple_enc_qm9/pipeline_test.py \
        --sample-index 0 \
        --embed-dim 128 \
        --num-rbf 32 \
        --cutoff 5.0 \
        --show-schema
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Project imports
# ============================================================

from src.simple_enc_qm9.dataset_qm9 import (
    QM9Dataset,
    TARGET_NAMES,
    TARGET_UNITS,
)

from src.simple_enc_qm9.feature_schema import (
    DEFAULT_ATOM_FEATURE_SCHEMA,
    DEFAULT_BOND_FEATURE_SCHEMA,
    print_schema,
)

from src.simple_enc_qm9.atom_feature_encoder import (
    DynamicAtomFeatureEncoder,
)

from src.simple_enc_qm9.bond_feature_encoder import (
    DynamicBondFeatureEncoder,
)

from src.simple_enc_qm9.geometry_encoder import (
    GeometryEncoder,
)


# ============================================================
# Utility
# ============================================================

def print_separator(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def resolve_device(
    name: str,
) -> torch.device:

    if name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return torch.device(
        name
    )


def print_tensor_preview(
    name: str,
    tensor: torch.Tensor,
    rows: int,
) -> None:
    """Print tensor shape and a short row preview."""

    print(
        f"{name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}"
    )

    if tensor.numel() == 0:
        print("  <empty>")
        return

    if tensor.ndim == 0:
        print(
            " ",
            tensor.item(),
        )
        return

    preview = tensor[
        :rows
    ].detach().cpu()

    print(
        preview
    )

    if tensor.size(0) > rows:
        print(
            f"  ... "
            f"({tensor.size(0) - rows} more rows)"
        )


# ============================================================
# Rigid-transform sanity check
# ============================================================

@torch.no_grad()
def check_geometry_invariance(
    encoder: GeometryEncoder,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    reference: dict[str, torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Check translation/rotation behavior of the geometry encoder.

    Expected behavior:

        distance:
            invariant

        H_geo:
            invariant
            (because the current H_geo is distance/RBF based)

        direction:
            translation-invariant and rotation-equivariant
    """

    print_separator(
        "6. GEOMETRY INVARIANCE / EQUIVARIANCE CHECK"
    )

    # --------------------------------------------------------
    # 1. Translation
    # --------------------------------------------------------

    translation = torch.tensor(
        [
            1.25,
            -0.75,
            2.50,
        ],
        dtype=pos.dtype,
        device=pos.device,
    )

    pos_translated = (
        pos
        + translation
    )

    translated = encoder(
        pos=pos_translated,
        edge_index=edge_index,
        return_geometry=True,
    )

    distance_translation_ok = (
        torch.allclose(
            reference["distance"],
            translated["distance"],
            atol=atol,
            rtol=rtol,
        )
    )

    direction_translation_ok = (
        torch.allclose(
            reference["direction"],
            translated["direction"],
            atol=atol,
            rtol=rtol,
        )
    )

    h_geo_translation_ok = (
        torch.allclose(
            reference["H_geo"],
            translated["H_geo"],
            atol=atol,
            rtol=rtol,
        )
    )

    print(
        f"[{'PASS' if distance_translation_ok else 'FAIL'}] "
        "distance is translation-invariant"
    )

    print(
        f"[{'PASS' if direction_translation_ok else 'FAIL'}] "
        "direction is translation-invariant"
    )

    print(
        f"[{'PASS' if h_geo_translation_ok else 'FAIL'}] "
        "H_geo is translation-invariant"
    )

    # --------------------------------------------------------
    # 2. Rotation around z-axis
    # --------------------------------------------------------

    theta = 0.731

    c = math.cos(
        theta
    )

    s = math.sin(
        theta
    )

    rotation = torch.tensor(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=pos.dtype,
        device=pos.device,
    )

    # pos stores row vectors, therefore:
    #
    # r' = r @ R^T

    pos_rotated = (
        pos
        @ rotation.T
    )

    rotated = encoder(
        pos=pos_rotated,
        edge_index=edge_index,
        return_geometry=True,
    )

    distance_rotation_ok = (
        torch.allclose(
            reference["distance"],
            rotated["distance"],
            atol=atol,
            rtol=rtol,
        )
    )

    h_geo_rotation_ok = (
        torch.allclose(
            reference["H_geo"],
            rotated["H_geo"],
            atol=atol,
            rtol=rtol,
        )
    )

    expected_rotated_direction = (
        reference["direction"]
        @ rotation.T
    )

    direction_equivariance_ok = (
        torch.allclose(
            expected_rotated_direction,
            rotated["direction"],
            atol=atol,
            rtol=rtol,
        )
    )

    print(
        f"[{'PASS' if distance_rotation_ok else 'FAIL'}] "
        "distance is rotation-invariant"
    )

    print(
        f"[{'PASS' if h_geo_rotation_ok else 'FAIL'}] "
        "H_geo is rotation-invariant"
    )

    print(
        f"[{'PASS' if direction_equivariance_ok else 'FAIL'}] "
        "direction is rotation-equivariant"
    )

    all_ok = all([
        distance_translation_ok,
        direction_translation_ok,
        h_geo_translation_ok,
        distance_rotation_ok,
        h_geo_rotation_ok,
        direction_equivariance_ok,
    ])

    if not all_ok:
        raise AssertionError(
            "Geometry invariance/equivariance check failed."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Test QM9 dataset -> atom/bond/geometry encoders."
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--root",
        type=str,
        default="dataset/qm9",
        help="QM9 dataset root.",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=[
            *TARGET_NAMES,
        ],
        help=(
            "Optional single QM9 target. "
            "If omitted, all 19 targets are retained."
        ),
    )

    parser.add_argument(
        "--no-hydrogens",
        action="store_true",
        help="Remove explicit hydrogen atoms.",
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Do not attempt to download QM9 if raw files "
            "are missing."
        ),
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--atom-fusion",
        type=str,
        choices=[
            "sum",
            "mean",
            "concat",
        ],
        default="mean",
    )

    parser.add_argument(
        "--bond-fusion",
        type=str,
        choices=[
            "sum",
            "mean",
            "concat",
        ],
        default="mean",
    )

    parser.add_argument(
        "--num-rbf",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--no-cutoff-envelope",
        action="store_true",
    )

    # --------------------------------------------------------
    # Debug output
    # --------------------------------------------------------

    parser.add_argument(
        "--show-rows",
        type=int,
        default=5,
        help="Number of rows to preview from tensors.",
    )

    parser.add_argument(
        "--show-schema",
        action="store_true",
    )

    parser.add_argument(
        "--skip-invariance-check",
        action="store_true",
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

    if args.show_rows < 1:
        raise ValueError(
            "--show-rows must be >= 1"
        )

    device = resolve_device(
        args.device
    )

    # ========================================================
    # 1. Dataset
    # ========================================================

    print_separator(
        "1. QM9 DATASET"
    )

    dataset = QM9Dataset(
        root=args.root,
        target=args.target,
        include_hydrogens=(
            not args.no_hydrogens
        ),
        download=(
            not args.no_download
        ),
    )

    data = dataset[
        args.sample_index
    ]

    print(
        f"device             : {device}"
    )

    print(
        f"dataset size       : {len(dataset):,}"
    )

    print(
        f"sample index       : {args.sample_index}"
    )

    print(
        f"raw index          : "
        f"{int(data.raw_index.item())}"
    )

    print(
        f"mol_id             : {data.mol_id}"
    )

    print(
        f"SMILES             : {data.smiles}"
    )

    print(
        f"explicit hydrogens : "
        f"{not args.no_hydrogens}"
    )

    # Keep metadata on CPU, move tensor fields to device.
    data = data.to(
        device
    )

    # ========================================================
    # 2. Raw molecular representation
    # ========================================================

    print_separator(
        "2. RAW MOLECULAR REPRESENTATION"
    )

    print_tensor_preview(
        "x (atom features)",
        data.x,
        args.show_rows,
    )

    print()

    print_tensor_preview(
        "edge_index",
        data.edge_index.T,
        args.show_rows,
    )

    print()

    print_tensor_preview(
        "edge_attr (bond features)",
        data.edge_attr,
        args.show_rows,
    )

    print()

    print_tensor_preview(
        "pos (3D positions)",
        data.pos,
        args.show_rows,
    )

    # Basic raw shape checks.
    num_atoms = data.x.size(
        0
    )

    num_edges = data.edge_index.size(
        1
    )

    assert (
        data.pos.shape
        ==
        (num_atoms, 3)
    )

    assert (
        data.edge_attr.size(0)
        ==
        num_edges
    )

    assert (
        data.edge_index.size(0)
        ==
        2
    )

    print()

    print(
        f"[PASS] raw atom count N = "
        f"{num_atoms}"
    )

    print(
        f"[PASS] directed edge count E = "
        f"{num_edges}"
    )

    # ========================================================
    # Optional schema print
    # ========================================================

    if args.show_schema:

        print_separator(
            "3. FEATURE SCHEMAS"
        )

        print_schema(
            DEFAULT_ATOM_FEATURE_SCHEMA,
            title="Atom Feature Schema",
        )

        print()

        print_schema(
            DEFAULT_BOND_FEATURE_SCHEMA,
            title="Bond Feature Schema",
        )

    # ========================================================
    # 3/4. Atom feature encoding
    # ========================================================

    print_separator(
        "4. ATOM FEATURE ENCODER"
    )

    atom_encoder = (
        DynamicAtomFeatureEncoder(
            embed_dim=args.embed_dim,
            feature_schema=(
                DEFAULT_ATOM_FEATURE_SCHEMA
            ),
            fusion=args.atom_fusion,
        )
        .to(device)
        .eval()
    )

    with torch.no_grad():

        (
            H_atom,
            atom_feature_embeddings,
        ) = atom_encoder(
            data.x,
            return_feature_embeddings=True,
        )

    print(
        "enabled atom features:",
        atom_encoder.feature_names,
    )

    print(
        "fusion:",
        atom_encoder.fusion,
    )

    print_tensor_preview(
        "H_atom",
        H_atom,
        args.show_rows,
    )

    assert (
        H_atom.shape
        ==
        (
            num_atoms,
            args.embed_dim,
        )
    )

    for name, value in (
        atom_feature_embeddings.items()
    ):
        assert (
            value.shape
            ==
            (
                num_atoms,
                args.embed_dim,
            )
        )

    print(
        f"[PASS] H_atom shape = "
        f"[N, D] = "
        f"[{num_atoms}, {args.embed_dim}]"
    )

    # ========================================================
    # 5. Bond feature encoding
    # ========================================================

    print_separator(
        "5. BOND FEATURE ENCODER"
    )

    bond_encoder = (
        DynamicBondFeatureEncoder(
            embed_dim=args.embed_dim,
            feature_schema=(
                DEFAULT_BOND_FEATURE_SCHEMA
            ),
            fusion=args.bond_fusion,
        )
        .to(device)
        .eval()
    )

    with torch.no_grad():

        (
            H_edge,
            bond_feature_embeddings,
        ) = bond_encoder(
            data.edge_attr,
            return_feature_embeddings=True,
        )

    print(
        "enabled bond features:",
        bond_encoder.feature_names,
    )

    print(
        "fusion:",
        bond_encoder.fusion,
    )

    print_tensor_preview(
        "H_edge",
        H_edge,
        args.show_rows,
    )

    assert (
        H_edge.shape
        ==
        (
            num_edges,
            args.embed_dim,
        )
    )

    for name, value in (
        bond_feature_embeddings.items()
    ):
        assert (
            value.shape
            ==
            (
                num_edges,
                args.embed_dim,
            )
        )

    print(
        f"[PASS] H_edge shape = "
        f"[E, D] = "
        f"[{num_edges}, {args.embed_dim}]"
    )

    # ========================================================
    # 6. Geometry encoding
    # ========================================================

    print_separator(
        "6. GEOMETRY ENCODER"
    )

    geometry_encoder = (
        GeometryEncoder(
            embed_dim=args.embed_dim,
            num_rbf=args.num_rbf,
            cutoff=args.cutoff,
            use_cutoff_envelope=(
                not args.no_cutoff_envelope
            ),
        )
        .to(device)
        .eval()
    )

    with torch.no_grad():

        geometry = geometry_encoder(
            pos=data.pos,
            edge_index=data.edge_index,
            return_geometry=True,
        )

    H_geo = geometry[
        "H_geo"
    ]

    print_tensor_preview(
        "distance",
        geometry["distance"],
        args.show_rows,
    )

    print()

    print_tensor_preview(
        "displacement",
        geometry["displacement"],
        args.show_rows,
    )

    print()

    print_tensor_preview(
        "direction",
        geometry["direction"],
        args.show_rows,
    )

    print()

    print_tensor_preview(
        "rbf",
        geometry["rbf"],
        args.show_rows,
    )

    print()

    print_tensor_preview(
        "H_geo",
        H_geo,
        args.show_rows,
    )

    assert (
        geometry[
            "distance"
        ].shape
        ==
        (num_edges,)
    )

    assert (
        geometry[
            "displacement"
        ].shape
        ==
        (
            num_edges,
            3,
        )
    )

    assert (
        geometry[
            "direction"
        ].shape
        ==
        (
            num_edges,
            3,
        )
    )

    assert (
        geometry[
            "rbf"
        ].shape
        ==
        (
            num_edges,
            args.num_rbf,
        )
    )

    assert (
        H_geo.shape
        ==
        (
            num_edges,
            args.embed_dim,
        )
    )

    # Unit directions should have norm 1 for nonzero-distance edges.
    if num_edges > 0:

        nonzero = (
            geometry["distance"]
            > geometry_encoder.eps
        )

        if torch.any(
            nonzero
        ):

            direction_norm = (
                torch.linalg.vector_norm(
                    geometry[
                        "direction"
                    ][nonzero],
                    dim=-1,
                )
            )

            assert torch.allclose(
                direction_norm,
                torch.ones_like(
                    direction_norm
                ),
                atol=1e-5,
                rtol=1e-5,
            )

    print(
        f"[PASS] H_geo shape = "
        f"[E, D] = "
        f"[{num_edges}, {args.embed_dim}]"
    )

    # ========================================================
    # 7. Rigid-transform check
    # ========================================================

    if not args.skip_invariance_check:

        check_geometry_invariance(
            encoder=geometry_encoder,
            pos=data.pos,
            edge_index=data.edge_index,
            reference=geometry,
        )

    # ========================================================
    # 8. QM9 targets
    # ========================================================

    print_separator(
        "8. QM9 TARGETS"
    )

    if hasattr(
        data,
        "y_all",
    ):

        target_values = (
            data.y_all
            .view(-1)
            .detach()
            .cpu()
        )

        print(
            f"y_all shape: "
            f"{tuple(data.y_all.shape)}"
        )

        for (
            name,
            unit,
            value,
        ) in zip(
            TARGET_NAMES,
            TARGET_UNITS,
            target_values.tolist(),
        ):

            print(
                f"{name:10s} "
                f"{value: .6f} "
                f"[{unit}]"
            )

    else:

        print(
            "No y_all field found."
        )

    # ========================================================
    # 9. Final pipeline summary
    # ========================================================

    print_separator(
        "9. PIPELINE CHECK"
    )

    print(
        "[PASS] QM9Dataset"
        " -> x / edge_attr / edge_index / pos"
    )

    print(
        "[PASS] x"
        " -> DynamicAtomFeatureEncoder"
        " -> H_atom [N, D]"
    )

    print(
        "[PASS] edge_attr"
        " -> DynamicBondFeatureEncoder"
        " -> H_edge [E, D]"
    )

    print(
        "[PASS] pos + edge_index"
        " -> GeometryEncoder"
        " -> distance / direction / RBF / H_geo"
    )

    print()

    print(
        "Current outputs:"
    )

    print(
        f"  H_atom : "
        f"{tuple(H_atom.shape)}"
    )

    print(
        f"  H_edge : "
        f"{tuple(H_edge.shape)}"
    )

    print(
        f"  H_geo  : "
        f"{tuple(H_geo.shape)}"
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "  H_atom + H_edge + H_geo"
    )

    print(
        "      -> 2D + 3D Message Passing"
    )

    print(
        "      -> Updated Atom Latents"
    )

    print(
        "      -> Readout"
    )

    print(
        "      -> H_mol"
    )


if __name__ == "__main__":
    main()
