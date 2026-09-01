#!/usr/bin/env python3
"""Convert a SMILES string into an RDKit molecule and PyG graph tensors.

Pipeline:
    SMILES -> RDKit Mol -> Atom/Bond features -> x / edge_index / edge_attr

This script is intentionally explicit rather than highly optimized so that each
preprocessing decision is easy to inspect and modify while studying molecular
encoders.
"""

from __future__ import annotations

import argparse
from typing import Dict

import torch
from rdkit import Chem
from torch_geometric.data import Data


HYBRIDIZATION_TO_ID: Dict[Chem.rdchem.HybridizationType, int] = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
}

CHIRALITY_TO_ID: Dict[Chem.rdchem.ChiralType, int] = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}

BOND_TYPE_TO_ID: Dict[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

STEREO_TO_ID: Dict[Chem.rdchem.BondStereo, int] = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}


def atom_features(atom: Chem.rdchem.Atom) -> list[int]:
    """Return a compact categorical feature vector for one atom.

    Columns:
      0 atomic number
      1 degree
      2 formal charge
      3 total number of hydrogens
      4 aromatic flag
      5 ring-membership flag
      6 hybridization category id
      7 chirality category id
    """
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        HYBRIDIZATION_TO_ID.get(atom.GetHybridization(), 0),
        CHIRALITY_TO_ID.get(atom.GetChiralTag(), 0),
    ]


def bond_features(bond: Chem.rdchem.Bond) -> list[int]:
    """Return a compact categorical feature vector for one directed edge.

    Columns:
      0 bond type id: single/double/triple/aromatic
      1 conjugated flag
      2 ring-membership flag
      3 stereochemistry id
    """
    return [
        BOND_TYPE_TO_ID.get(bond.GetBondType(), 4),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        STEREO_TO_ID.get(bond.GetStereo(), 0),
    ]


def smiles_to_pyg(smiles: str, explicit_h: bool = False) -> tuple[Chem.Mol, Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    # Preserve the common 2D-GNN convention by default: hydrogens are implicit.
    if explicit_h:
        mol = Chem.AddHs(mol)

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    attrs: list[list[int]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)

        # An undirected chemical bond becomes two directed message-passing edges.
        src.extend([i, j])
        dst.extend([j, i])
        attrs.extend([feat, feat])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    return mol, data


def print_rdkit_summary(mol: Chem.Mol) -> None:
    print("\n=== RDKit atoms ===")
    for atom in mol.GetAtoms():
        print(
            f"idx={atom.GetIdx():2d}  symbol={atom.GetSymbol():>2s}  "
            f"Z={atom.GetAtomicNum():2d}  degree={atom.GetDegree()}  "
            f"charge={atom.GetFormalCharge():+d}  H={atom.GetTotalNumHs()}  "
            f"aromatic={atom.GetIsAromatic()}  ring={atom.IsInRing()}  "
            f"hyb={atom.GetHybridization()}  chiral={atom.GetChiralTag()}"
        )

    print("\n=== RDKit bonds ===")
    for bond in mol.GetBonds():
        print(
            f"{bond.GetBeginAtomIdx():2d} <-> {bond.GetEndAtomIdx():2d}  "
            f"type={str(bond.GetBondType()):8s}  "
            f"conjugated={bond.GetIsConjugated()}  ring={bond.IsInRing()}  "
            f"stereo={bond.GetStereo()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles",
        default="CC(=O)O",
        help="SMILES string. Default: acetic acid, CC(=O)O",
    )
    parser.add_argument(
        "--explicit-h",
        action="store_true",
        help="Convert implicit hydrogens into explicit atom nodes.",
    )
    args = parser.parse_args()

    mol, data = smiles_to_pyg(args.smiles, explicit_h=args.explicit_h)

    print(f"SMILES: {args.smiles}")
    print(f"Canonical SMILES: {Chem.MolToSmiles(mol)}")
    print_rdkit_summary(mol)

    print("\n=== PyTorch Geometric Data ===")
    print(data)

    print("\n=== x: atom feature matrix ===")
    print("columns = [atomic_num, degree, formal_charge, num_H, aromatic, in_ring, hybridization_id, chirality_id]")
    print(data.x)
    print("shape:", tuple(data.x.shape))

    print("\n=== edge_index: directed connectivity ===")
    print(data.edge_index)
    print("shape:", tuple(data.edge_index.shape))

    print("\n=== edge_attr: bond feature matrix ===")
    print("columns = [bond_type_id, conjugated, in_ring, stereo_id]")
    print(data.edge_attr)
    print("shape:", tuple(data.edge_attr.shape))

    print("\nFeature IDs")
    print("  bond_type_id: 0=single, 1=double, 2=triple, 3=aromatic")
    print("  hybridization_id: 0=unspecified, 1=s, 2=sp, 3=sp2, 4=sp3, 5=sp3d, 6=sp3d2")
    print("  chirality_id: 0=unspecified, 1=CW, 2=CCW, 3=other")


if __name__ == "__main__":
    main()
