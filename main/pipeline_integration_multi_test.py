#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch_geometric.data import Batch


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
            "one or more SMILES -> PyG Batch -> "
            "feature encoding -> MPNN -> graph readout."
        )
    )


    # --------------------------------------------------------
    # SMILES
    # --------------------------------------------------------

    parser.add_argument(
        "--smiles",
        type=str,
        nargs="+",
        default=[
            "CC(=O)O",
        ],
        help=(
            "One or more input SMILES strings. "
            "Example: --smiles 'CC(=O)O' 'CCO' 'c1ccccc1'. "
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
    # Molecule indices to inspect
    # --------------------------------------------------------

    parser.add_argument(
        "--molecule-index",
        type=int,
        nargs="+",
        default=[
            0,
        ],
        help=(
            "Molecule index/indices to inspect in detail. "
            "Example: --molecule-index 0 1"
        ),
    )


    # --------------------------------------------------------
    # Local atom indices to inspect
    # --------------------------------------------------------

    parser.add_argument(
        "--atom-index",
        type=int,
        nargs="+",
        default=[
            0,
        ],
        help=(
            "Local atom index/indices to inspect "
            "inside each selected molecule. "
            "Example: --atom-index 0 1 3"
        ),
    )


    # --------------------------------------------------------
    # Local directed edge indices to inspect
    # --------------------------------------------------------

    parser.add_argument(
        "--edge-index",
        type=int,
        nargs="+",
        default=[
            0,
        ],
        help=(
            "Local directed edge index/indices to inspect "
            "inside each selected molecule. "
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

    smiles_list = args.smiles
    embed_dim = args.embed_dim
    batch_size = len(
        smiles_list
    )


    # ========================================================
    # 1. SMILES -> Molecular Graphs
    # ========================================================

    mol_list = []
    data_list = []

    for smiles in smiles_list:

        mol, data = smiles_to_pyg(
            smiles,
            explicit_h=args.explicit_h,
        )

        mol_list.append(
            mol
        )

        data_list.append(
            data
        )


    # --------------------------------------------------------
    # Create PyG Batch
    # --------------------------------------------------------

    batch_data = Batch.from_data_list(
        data_list
    )


    # --------------------------------------------------------
    # Compute local -> global offsets
    # --------------------------------------------------------

    node_offsets = []
    edge_offsets = []

    node_offset = 0
    edge_offset = 0

    for data in data_list:

        node_offsets.append(
            node_offset
        )

        edge_offsets.append(
            edge_offset
        )

        node_offset += (
            data.x.size(0)
        )

        edge_offset += (
            data.edge_attr.size(0)
        )


    print_separator(
        "1. SMILES -> MOLECULAR GRAPHS"
    )


    # ========================================================
    # Per-molecule graph summaries
    # ========================================================

    for molecule_idx, (
        smiles,
        mol,
        data,
    ) in enumerate(
        zip(
            smiles_list,
            mol_list,
            data_list,
        )
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"Molecule {molecule_idx}"
        )

        print(
            "-" * 70
        )

        print_graph_summary(
            smiles,
            mol,
            data,
        )


    # ========================================================
    # PyG Batch summary
    # ========================================================

    print(
        "\n=== PyG Batch ==="
    )

    print(
        batch_data
    )

    print(
        "\nnumber of molecules:"
    )

    print(
        batch_size
    )

    print(
        "\nbatch.x shape:"
    )

    print(
        tuple(
            batch_data.x.shape
        )
    )

    print(
        "\nbatch.edge_index shape:"
    )

    print(
        tuple(
            batch_data.edge_index.shape
        )
    )

    print(
        "\nbatch.edge_attr shape:"
    )

    print(
        tuple(
            batch_data.edge_attr.shape
        )
    )

    print(
        "\nbatch.batch:"
    )

    print(
        batch_data.batch
    )

    print(
        "shape:",
        tuple(
            batch_data.batch.shape
        ),
    )


    # ========================================================
    # Batch index mapping
    # ========================================================

    print(
        "\n=== Batch index mapping ==="
    )

    for molecule_idx, data in enumerate(
        data_list
    ):

        local_num_atoms = (
            data.x.size(0)
        )

        local_num_edges = (
            data.edge_attr.size(0)
        )

        global_atom_start = (
            node_offsets[
                molecule_idx
            ]
        )

        global_atom_end = (
            global_atom_start
            + local_num_atoms
            - 1
        )

        if local_num_edges > 0:

            global_edge_start = (
                edge_offsets[
                    molecule_idx
                ]
            )

            global_edge_end = (
                global_edge_start
                + local_num_edges
                - 1
            )

            local_edge_range = (
                f"0 ~ {local_num_edges - 1}"
            )

            global_edge_range = (
                f"{global_edge_start} ~ "
                f"{global_edge_end}"
            )

        else:

            local_edge_range = "none"
            global_edge_range = "none"


        print(
            f"Molecule {molecule_idx}: "
            f"local atoms=0 ~ {local_num_atoms - 1}, "
            f"global atoms={global_atom_start} ~ {global_atom_end}, "
            f"local edges={local_edge_range}, "
            f"global edges={global_edge_range}"
        )


    # ========================================================
    # Validate requested molecule indices
    # ========================================================

    validate_indices(
        args.molecule_index,
        batch_size,
        "Molecule",
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
        x=batch_data.x,
        edge_index=batch_data.edge_index,
        edge_attr=batch_data.edge_attr,
        batch=batch_data.batch,
        return_all_layers=True,
    )


    # --------------------------------------------------------
    # Unpack MoleculeEncoder outputs
    # --------------------------------------------------------

    H_atom = outputs[
        "H_atom"
    ]

    H_edge = outputs[
        "H_edge"
    ]

    H_atom_updated = outputs[
        "H_atom_updated"
    ]

    H_mol = outputs[
        "H_mol"
    ]

    layer_outputs = outputs[
        "layer_outputs"
    ]


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
        tuple(
            H_atom.shape
        ),
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
        tuple(
            H_edge.shape
        ),
    )


    # ========================================================
    # Selected molecule feature inspection
    # ========================================================

    for molecule_idx in args.molecule_index:

        mol = mol_list[
            molecule_idx
        ]

        data = data_list[
            molecule_idx
        ]

        current_node_offset = (
            node_offsets[
                molecule_idx
            ]
        )

        current_edge_offset = (
            edge_offsets[
                molecule_idx
            ]
        )


        print(
            f"\n=== Molecule {molecule_idx}: "
            "feature encoding inspection ==="
        )

        print(
            "SMILES:"
        )

        print(
            smiles_list[
                molecule_idx
            ]
        )


        # ----------------------------------------------------
        # Atom examples
        # ----------------------------------------------------

        validate_indices(
            args.atom_index,
            data.x.size(0),
            (
                f"Molecule {molecule_idx} "
                "local atom"
            ),
        )


        for local_atom_idx in args.atom_index:

            global_atom_idx = (
                current_node_offset
                + local_atom_idx
            )

            atom = (
                mol.GetAtomWithIdx(
                    local_atom_idx
                )
            )


            print(
                f"\n=== Molecule {molecule_idx} / "
                f"Atom {local_atom_idx}: "
                "feature encoding example ==="
            )

            print(
                "symbol:"
            )

            print(
                atom.GetSymbol()
            )

            print(
                "\nlocal atom index:"
            )

            print(
                local_atom_idx
            )

            print(
                "\nglobal atom index:"
            )

            print(
                global_atom_idx
            )

            print(
                "\nraw feature:"
            )

            print(
                data.x[
                    local_atom_idx
                ]
            )

            print(
                "\nencoded feature:"
            )

            print(
                H_atom[
                    global_atom_idx
                ]
            )


        # ----------------------------------------------------
        # Edge examples
        # ----------------------------------------------------

        local_edge_count = (
            data.edge_attr.size(0)
        )

        if local_edge_count > 0:

            validate_indices(
                args.edge_index,
                local_edge_count,
                (
                    f"Molecule {molecule_idx} "
                    "local edge"
                ),
            )


            for local_edge_idx in args.edge_index:

                global_edge_idx = (
                    current_edge_offset
                    + local_edge_idx
                )

                src_local = (
                    data.edge_index[
                        0,
                        local_edge_idx
                    ]
                    .item()
                )

                dst_local = (
                    data.edge_index[
                        1,
                        local_edge_idx
                    ]
                    .item()
                )

                src_global = (
                    current_node_offset
                    + src_local
                )

                dst_global = (
                    current_node_offset
                    + dst_local
                )

                src_symbol = (
                    mol.GetAtomWithIdx(
                        src_local
                    )
                    .GetSymbol()
                )

                dst_symbol = (
                    mol.GetAtomWithIdx(
                        dst_local
                    )
                    .GetSymbol()
                )


                print(
                    f"\n=== Molecule {molecule_idx} / "
                    f"Edge {local_edge_idx}: "
                    "feature encoding example ==="
                )

                print(
                    "local connectivity:"
                )

                print(
                    f"{src_local} ({src_symbol}) "
                    f"-> "
                    f"{dst_local} ({dst_symbol})"
                )

                print(
                    "\nglobal connectivity:"
                )

                print(
                    f"{src_global} ({src_symbol}) "
                    f"-> "
                    f"{dst_global} ({dst_symbol})"
                )

                print(
                    "\nraw feature:"
                )

                print(
                    data.edge_attr[
                        local_edge_idx
                    ]
                )

                print(
                    "\nencoded feature:"
                )

                print(
                    H_edge[
                        global_edge_idx
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
        tuple(
            H_atom.shape
        ),
    )

    print(
        "edge_index shape:",
        tuple(
            batch_data.edge_index.shape
        ),
    )

    print(
        "H_edge shape:",
        tuple(
            H_edge.shape
        ),
    )

    print(
        "batch size:",
        batch_size,
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
            "updated atom feature matrix ==="
        )

        print(
            H_layer
        )

        print(
            "shape:",
            tuple(
                H_layer.shape
            ),
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
        tuple(
            H_atom_updated.shape
        ),
    )


    # ========================================================
    # Selected atom MPNN traces
    # ========================================================

    for molecule_idx in args.molecule_index:

        mol = mol_list[
            molecule_idx
        ]

        data = data_list[
            molecule_idx
        ]

        current_node_offset = (
            node_offsets[
                molecule_idx
            ]
        )


        validate_indices(
            args.atom_index,
            data.x.size(0),
            (
                f"Molecule {molecule_idx} "
                "local atom"
            ),
        )


        for local_atom_idx in args.atom_index:

            global_atom_idx = (
                current_node_offset
                + local_atom_idx
            )

            atom = (
                mol.GetAtomWithIdx(
                    local_atom_idx
                )
            )


            print(
                f"\n=== Molecule {molecule_idx} / "
                f"Atom {local_atom_idx} "
                f"({atom.GetSymbol()}): "
                "MPNN representation trace ==="
            )

            print(
                "global atom index:"
            )

            print(
                global_atom_idx
            )

            print(
                "\nbefore MPNN:"
            )

            print(
                H_atom[
                    global_atom_idx
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
                        global_atom_idx
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

    print(
        "\nbatch:"
    )

    print(
        batch_data.batch
    )

    print(
        "batch size:",
        batch_size,
    )


    # ========================================================
    # Readout result
    # ========================================================

    if H_mol is not None:

        print(
            "\n=== H_mol: molecule latent matrix ==="
        )

        print(
            H_mol
        )

        print(
            "shape:",
            tuple(
                H_mol.shape
            ),
        )


        # ----------------------------------------------------
        # Per-molecule latent vectors
        # ----------------------------------------------------

        for molecule_idx, smiles in enumerate(
            smiles_list
        ):

            print(
                f"\n=== Molecule {molecule_idx}: "
                "H_mol ==="
            )

            print(
                "SMILES:"
            )

            print(
                smiles
            )

            print(
                "\nlatent vector:"
            )

            print(
                H_mol[
                    molecule_idx
                ]
            )

            print(
                "shape:",
                tuple(
                    H_mol[
                        molecule_idx
                    ].shape
                ),
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

    total_atoms = sum(
        data.x.size(0)
        for data in data_list
    )

    total_edges = sum(
        data.edge_attr.size(0)
        for data in data_list
    )


    assert (
        batch_data.num_graphs
        ==
        batch_size
    )


    assert H_atom.shape == (
        total_atoms,
        embed_dim,
    )


    assert H_edge.shape == (
        total_edges,
        embed_dim,
    )


    assert (
        batch_data.edge_index.shape[1]
        ==
        H_edge.shape[0]
    )


    assert H_atom_updated.shape == (
        total_atoms,
        embed_dim,
    )


    assert (
        batch_data.batch.shape[0]
        ==
        total_atoms
    )


    assert (
        len(layer_outputs)
        ==
        args.mpnn_layers
    )


    if H_mol is not None:

        assert H_mol.shape == (
            batch_size,
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

    assert (
        set(
            outputs.keys()
        )
        ==
        expected_output_keys
    )


    print_separator(
        "5. PIPELINE CHECK"
    )


    print(
        "[PASS] "
        f"{batch_size} molecule(s) "
        "-> PyG Batch"
    )

    print(
        "[PASS] "
        f"total atoms == {total_atoms}"
    )

    print(
        "[PASS] "
        f"total directed edges == {total_edges}"
    )

    print(
        "[PASS] "
        "MoleculeEncoder -> "
        "AtomFeatureEncoder -> H_atom"
    )

    print(
        "[PASS] "
        "MoleculeEncoder -> "
        "BondFeatureEncoder -> H_edge"
    )

    print(
        "[PASS] "
        "MoleculeEncoder -> "
        "MPNN -> H_atom_updated"
    )

    print(
        "[PASS] "
        f"MPNN layer count == "
        f"{args.mpnn_layers}"
    )


    if H_mol is not None:

        print(
            "[PASS] "
            "MoleculeEncoder -> "
            f"{args.readout} Readout "
            f"-> H_mol "
            f"{tuple(H_mol.shape)}"
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


    if H_mol is not None:

        print(
            "\n=== H_mol ==="
        )

        print(
            "shape:",
            tuple(
                H_mol.shape
            ),
        )

        print(
            "\n=== Batch interpretation ==="
        )

        print(
            f"B = {batch_size}"
        )

        print(
            f"D = {embed_dim}"
        )

        print(
            f"H_mol shape = "
            f"[B, D] = "
            f"[{batch_size}, {embed_dim}]"
        )

        print(
            "\n=== Next pipeline ==="
        )

        print(
            "H_mol"
            " -> Task Head"
            " -> Molecular Property"
        )


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
