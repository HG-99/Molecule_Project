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

from src.feature_encoder import (
    AtomFeatureEncoder,
    BondFeatureEncoder,
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
            "Test molecular preprocessing and "
            "feature encoding pipeline."
        )
    )

    parser.add_argument(
        "--smiles",
        type=str,
        default="CC(=O)O",
        help=(
            "Input SMILES string. "
            "Default: acetic acid, CC(=O)O"
        ),
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help=(
            "Embedding dimension for atom/bond features. "
            "Default: 128"
        ),
    )

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
    # 2. Feature Encoder
    # ========================================================

    atom_encoder = AtomFeatureEncoder(
        embed_dim=embed_dim,
    )

    bond_encoder = BondFeatureEncoder(
        embed_dim=embed_dim,
    )

    H_atom = atom_encoder(
        data.x
    )

    H_edge = bond_encoder(
        data.edge_attr
    )


    print_separator(
        "2. FEATURE ENCODING"
    )


    # ========================================================
    # H_atom
    # ========================================================

    print(
        "\n=== H_atom: encoded atom feature matrix ==="
    )

    print(H_atom)

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

    print(H_edge)

    print(
        "shape:",
        tuple(H_edge.shape),
    )


    # ========================================================
    # Requested indices
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
    # Validate requested atom indices
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
            data.x[atom_idx]
        )

        print(
            "\nencoded feature:"
        )

        print(
            H_atom[atom_idx]
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
                mol.GetAtomWithIdx(src)
                .GetSymbol()
            )

            dst_symbol = (
                mol.GetAtomWithIdx(dst)
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
    # 3. Pipeline sanity check
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


    print_separator(
        "3. PIPELINE CHECK"
    )

    print(
        "[PASS] "
        "x -> AtomFeatureEncoder -> H_atom"
    )

    print(
        "[PASS] "
        "edge_attr -> BondFeatureEncoder -> H_edge"
    )

    print(
        "[PASS] "
        "edge_index columns == H_edge rows"
    )


    # ========================================================
    # 4. Next MPNN interface
    # ========================================================

    print_separator(
        "4. NEXT STEP: MPNN INPUT"
    )


    print(
        "\n=== H_atom ==="
    )

    print(
        "shape:",
        tuple(H_atom.shape),
    )


    print(
        "\n=== edge_index ==="
    )

    print(
        data.edge_index
    )

    print(
        "shape:",
        tuple(
            data.edge_index.shape
        ),
    )


    print(
        "\n=== H_edge ==="
    )

    print(
        "shape:",
        tuple(H_edge.shape),
    )


    print(
        "\n=== Next pipeline ==="
    )

    print(
        "(H_atom, edge_index, H_edge)"
        " -> MPNN"
        " -> Updated H_atom"
    )


if __name__ == "__main__":
    main()