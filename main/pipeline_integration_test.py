#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
# Import current modules
# ============================================================

from src.smiles_to_graph import (
    smiles_to_pyg,
    print_graph_summary,
)

from src.molecule_encoder import (
    MoleculeEncoder,
)


# ============================================================
# Output utility
# ============================================================

def print_separator(
    title: str,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_indices(
    indices: list[int],
    size: int,
    name: str,
) -> None:
    """Check that requested indices are valid."""

    for idx in indices:

        if idx < 0 or idx >= size:

            raise ValueError(
                f"{name} index {idx} is out of range. "
                f"Valid range: 0 ~ {size - 1}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    # ========================================================
    # 0. Arguments
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Integration test for MoleculeEncoder: "
            "SMILES preprocessing -> feature encoding -> "
            "MPNN -> graph readout."
        )
    )


    # --------------------------------------------------------
    # SMILES
    # --------------------------------------------------------

    parser.add_argument(
        "--smiles",
        type=str,
        default="CC(=O)O",
        help=(
            "Input SMILES string. "
            "Default: acetic acid, CC(=O)O"
        ),
    )


    # --------------------------------------------------------
    # Embedding dimension
    # --------------------------------------------------------

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help=(
            "Embedding dimension for atom/bond features. "
            "Default: 128"
        ),
    )


    # --------------------------------------------------------
    # Explicit hydrogen
    # --------------------------------------------------------

    parser.add_argument(
        "--explicit-h",
        action="store_true",
        help=(
            "Convert implicit hydrogens "
            "into explicit atom nodes."
        ),
    )


    # --------------------------------------------------------
    # Atom indices to inspect
    # --------------------------------------------------------

    parser.add_argument(
        "--atom-index",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Atom index/indices to inspect. "
            "Example: --atom-index 0 1 3"
        ),
    )


    # --------------------------------------------------------
    # Directed edge indices to inspect
    # --------------------------------------------------------

    parser.add_argument(
        "--edge-index",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Directed edge index/indices to inspect. "
            "Example: --edge-index 0 2 4"
        ),
    )


    # --------------------------------------------------------
    # MPNN configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--mpnn-layers",
        type=int,
        default=3,
        help=(
            "Number of MPNN layers. "
            "Default: 3"
        ),
    )

    parser.add_argument(
        "--mpnn-dropout",
        type=float,
        default=0.0,
        help=(
            "Dropout used inside MPNN layers. "
            "Default: 0.0"
        ),
    )


    # --------------------------------------------------------
    # Readout / Pooling configuration
    # --------------------------------------------------------

    parser.add_argument(
        "--readout",
        type=str,
        choices=[
            "mean",
            "sum",
            "max",
            "none",
        ],
        default="mean",
        help=(
            "Graph readout method. "
            "Choices: mean, sum, max, none. "
            "Default: mean"
        ),
    )


    args = parser.parse_args()

    smiles = args.smiles
    embed_dim = args.embed_dim


    # ========================================================
    # 1. SMILES -> Molecular Graph
    # ========================================================

    mol, data = smiles_to_pyg(
        smiles,
        explicit_h=args.explicit_h,
    )


    print_separator(
        "1. SMILES -> MOLECULAR GRAPH"
    )


    print_graph_summary(
        smiles,
        mol,
        data,
    )


    # ========================================================
    # 2. Molecule Encoder
    # ========================================================

    model = MoleculeEncoder(
        embed_dim=embed_dim,
        num_layers=args.mpnn_layers,
        dropout=args.mpnn_dropout,
        readout=args.readout,
    )


    # --------------------------------------------------------
    # Run the complete neural pipeline through MoleculeEncoder
    # --------------------------------------------------------

    outputs = model(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        batch=None,
        return_all_layers=True,
    )


    # --------------------------------------------------------
    # Unpack MoleculeEncoder outputs
    # --------------------------------------------------------

    H_atom = outputs["H_atom"]
    H_edge = outputs["H_edge"]
    H_atom_updated = outputs["H_atom_updated"]
    H_mol = outputs["H_mol"]
    layer_outputs = outputs["layer_outputs"]


    print_separator(
        "2. FEATURE ENCODING"
    )


    # ========================================================
    # H_atom
    # ========================================================

    print(
        "\n=== H_atom: encoded atom feature matrix ==="
    )

    print(
        H_atom
    )

    print(
        "shape:",
        tuple(H_atom.shape),
    )


    # ========================================================
    # H_edge
    # ========================================================

    print(
        "\n=== H_edge: encoded bond feature matrix ==="
    )

    print(
        H_edge
    )

    print(
        "shape:",
        tuple(H_edge.shape),
    )


    # ========================================================
    # Available indices
    # ========================================================

    print(
        "\n=== Available indices ==="
    )

    print(
        f"atom indices: "
        f"0 ~ {H_atom.size(0) - 1}"
    )


    if H_edge.size(0) > 0:

        print(
            f"directed edge indices: "
            f"0 ~ {H_edge.size(0) - 1}"
        )

    else:

        print(
            "directed edge indices: none"
        )


    # ========================================================
    # Validate atom indices
    # ========================================================

    validate_indices(
        args.atom_index,
        H_atom.size(0),
        "Atom",
    )


    # ========================================================
    # Atom representation examples
    # ========================================================

    for atom_idx in args.atom_index:

        atom = mol.GetAtomWithIdx(
            atom_idx
        )


        print(
            f"\n=== Atom {atom_idx}: "
            f"feature encoding example ==="
        )


        print(
            "symbol:"
        )

        print(
            atom.GetSymbol()
        )


        print(
            "\nraw feature:"
        )

        print(
            data.x[
                atom_idx
            ]
        )


        print(
            "\nencoded feature:"
        )

        print(
            H_atom[
                atom_idx
            ]
        )


    # ========================================================
    # Edge representation examples
    # ========================================================

    if H_edge.size(0) > 0:

        validate_indices(
            args.edge_index,
            H_edge.size(0),
            "Edge",
        )


        for edge_idx in args.edge_index:

            src = (
                data.edge_index[
                    0,
                    edge_idx
                ]
                .item()
            )

            dst = (
                data.edge_index[
                    1,
                    edge_idx
                ]
                .item()
            )


            src_symbol = (
                mol.GetAtomWithIdx(
                    src
                )
                .GetSymbol()
            )

            dst_symbol = (
                mol.GetAtomWithIdx(
                    dst
                )
                .GetSymbol()
            )


            print(
                f"\n=== Edge {edge_idx}: "
                f"feature encoding example ==="
            )


            print(
                "connectivity:"
            )

            print(
                f"{src} ({src_symbol}) "
                f"-> "
                f"{dst} ({dst_symbol})"
            )


            print(
                "\nraw feature:"
            )

            print(
                data.edge_attr[
                    edge_idx
                ]
            )


            print(
                "\nencoded feature:"
            )

            print(
                H_edge[
                    edge_idx
                ]
            )


    # ========================================================
    # 3. Message Passing
    # ========================================================

    print_separator(
        "3. MESSAGE PASSING"
    )


    # ========================================================
    # MPNN input
    # ========================================================

    print(
        "\n=== MPNN input ==="
    )

    print(
        "H_atom shape:",
        tuple(H_atom.shape),
    )

    print(
        "edge_index shape:",
        tuple(data.edge_index.shape),
    )

    print(
        "H_edge shape:",
        tuple(H_edge.shape),
    )


    # ========================================================
    # MPNN configuration
    # ========================================================

    print(
        "\n=== MPNN configuration ==="
    )

    print(
        "num_layers:",
        args.mpnn_layers,
    )

    print(
        "dropout:",
        args.mpnn_dropout,
    )


    # ========================================================
    # MPNN layer outputs
    # ========================================================

    if layer_outputs is None:

        raise RuntimeError(
            "MoleculeEncoder did not return layer_outputs. "
            "Expected return_all_layers=True."
        )


    for layer_idx, H_layer in enumerate(
        layer_outputs,
        start=1,
    ):

        print(
            f"\n=== MPNN Layer {layer_idx}: "
            f"updated atom feature matrix ==="
        )

        print(
            H_layer
        )

        print(
            "shape:",
            tuple(H_layer.shape),
        )


    # ========================================================
    # Final updated atom representation
    # ========================================================

    print(
        "\n=== H_atom_updated: "
        "final updated atom feature matrix ==="
    )

    print(
        H_atom_updated
    )

    print(
        "shape:",
        tuple(H_atom_updated.shape),
    )


    # ========================================================
    # Selected atom MPNN trace
    # ========================================================

    for atom_idx in args.atom_index:

        atom = mol.GetAtomWithIdx(
            atom_idx
        )


        print(
            f"\n=== Atom {atom_idx} "
            f"({atom.GetSymbol()}): "
            f"MPNN representation trace ==="
        )


        print(
            "before MPNN:"
        )

        print(
            H_atom[
                atom_idx
            ]
        )


        for layer_idx, H_layer in enumerate(
            layer_outputs,
            start=1,
        ):

            print(
                f"\nafter layer {layer_idx}:"
            )

            print(
                H_layer[
                    atom_idx
                ]
            )


    # ========================================================
    # 4. Readout / Pooling
    # ========================================================

    print_separator(
        "4. READOUT / POOLING"
    )


    # ========================================================
    # Readout configuration
    # ========================================================

    print(
        "\n=== Readout configuration ==="
    )

    print(
        "mode:",
        args.readout,
    )


    # ========================================================
    # Readout input
    # ========================================================

    print(
        "\n=== Readout input ==="
    )

    print(
        "H_atom_updated shape:",
        tuple(
            H_atom_updated.shape
        ),
    )


    # ========================================================
    # Readout result
    # ========================================================

    if H_mol is not None:

        print(
            "\n=== H_mol: molecule latent ==="
        )

        print(
            H_mol
        )

        print(
            "shape:",
            tuple(H_mol.shape),
        )


    else:

        print(
            "\n=== Readout skipped ==="
        )

        print(
            "H_atom_updated remains "
            "the final encoder output."
        )

        print(
            "shape:",
            tuple(
                H_atom_updated.shape
            ),
        )


    # ========================================================
    # 5. Pipeline sanity check
    # ========================================================

    assert H_atom.shape == (
        data.x.shape[0],
        embed_dim,
    )


    assert H_edge.shape == (
        data.edge_attr.shape[0],
        embed_dim,
    )


    assert (
        data.edge_index.shape[1]
        ==
        H_edge.shape[0]
    )


    assert H_atom_updated.shape == (
        data.x.shape[0],
        embed_dim,
    )


    assert (
        len(layer_outputs)
        ==
        args.mpnn_layers
    )


    if H_mol is not None:

        assert H_mol.shape == (
            1,
            embed_dim,
        )


    # --------------------------------------------------------
    # MoleculeEncoder integration checks
    # --------------------------------------------------------

    expected_output_keys = {
        "H_atom",
        "H_edge",
        "H_atom_updated",
        "H_mol",
        "layer_outputs",
    }

    assert set(outputs.keys()) == expected_output_keys

    assert model.atom_encoder is not None
    assert model.bond_encoder is not None
    assert model.mpnn is not None

    if args.readout == "none":
        assert model.readout is None
        assert H_mol is None
    else:
        assert model.readout is not None
        assert H_mol is not None


    print_separator(
        "5. PIPELINE CHECK"
    )


    print(
        "[PASS] "
        "SMILES -> Molecular Graph"
    )


    print(
        "[PASS] "
        "MoleculeEncoder -> AtomFeatureEncoder -> H_atom"
    )


    print(
        "[PASS] "
        "MoleculeEncoder -> BondFeatureEncoder -> H_edge"
    )


    print(
        "[PASS] "
        "edge_index columns == H_edge rows"
    )


    print(
        "[PASS] "
        "MoleculeEncoder -> MPNN -> H_atom_updated"
    )


    print(
        "[PASS] "
        f"MPNN layer count == {args.mpnn_layers}"
    )


    if H_mol is not None:

        print(
            "[PASS] "
            "MoleculeEncoder -> "
            f"{args.readout} Readout -> H_mol"
        )

    else:

        print(
            "[PASS] "
            "MoleculeEncoder readout skipped; "
            "atom-level representation preserved"
        )


    print(
        "[PASS] "
        "MoleculeEncoder output dictionary structure"
    )


    # ========================================================
    # 6. Next Task Head interface
    # ========================================================

    print_separator(
        "6. NEXT STEP: TASK HEAD INPUT"
    )


    # --------------------------------------------------------
    # Molecule-level representation
    # --------------------------------------------------------

    if H_mol is not None:

        print(
            "\n=== H_mol ==="
        )

        print(
            "shape:",
            tuple(H_mol.shape),
        )


        print(
            "\n=== Next pipeline ==="
        )

        print(
            "H_mol"
            " -> Task Head"
            " -> Molecular Property"
        )


    # --------------------------------------------------------
    # Atom-level representation
    # --------------------------------------------------------

    else:

        print(
            "\n=== H_atom_updated ==="
        )

        print(
            "shape:",
            tuple(
                H_atom_updated.shape
            ),
        )


        print(
            "\n=== Next pipeline ==="
        )

        print(
            "H_atom_updated"
            " -> Atom / Interaction-level "
            "Downstream Module"
        )


if __name__ == "__main__":
    main()
