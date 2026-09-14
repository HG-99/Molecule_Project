#!/usr/bin/env python3
"""QM9 dataset loader for Molecule_Project.

This loader is designed to match the current project convention:

    x          : [N_atom, 8]  categorical atom features
    edge_index : [2, N_edge]
    edge_attr  : [N_edge, 4]  categorical bond features

while also preserving QM9-specific information:

    pos        : [N_atom, 3]  3D coordinates in Angstrom
    y          : selected target or all 19 targets
    y_all      : all 19 targets
    z          : atomic numbers
    mol_id     : original QM9 molecule id
    raw_index  : index in the raw QM9 files

The raw QM9 files are kept under:

    dataset/qm9/raw/
        gdb9.sdf
        gdb9.sdf.csv
        uncharacterized.txt

Run:

    python -m src.simple_enc_qm9.dataset_qm9 --inspect-raw

to download and inspect the original files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import download_url, extract_zip

from src.simple_mpnn.smiles_to_graph import atom_features, bond_features


# ============================================================
# QM9 download sources
# ============================================================

QM9_RAW_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/molnet_publish/qm9.zip"
)

QM9_UNCHARACTERIZED_URL = (
    "https://ndownloader.figshare.com/files/3195404"
)

RAW_FILE_NAMES = (
    "gdb9.sdf",
    "gdb9.sdf.csv",
    "uncharacterized.txt",
)


# ============================================================
# QM9 targets
# ============================================================

# Original CSV order after mol_id:
# A, B, C, mu, alpha, homo, lumo, gap, r2, zpve,
# u0, u298, h298, g298, cv,
# u0_atom, u298_atom, h298_atom, g298_atom

RAW_TARGET_NAMES = (
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
)

# Same order used by torch_geometric.datasets.QM9:
TARGET_NAMES = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
    "A",
    "B",
    "C",
)

TARGET_UNITS = (
    "D",
    "Bohr^3",
    "eV",
    "eV",
    "eV",
    "Bohr^2",
    "eV",
    "eV",
    "eV",
    "eV",
    "eV",
    "cal/(mol*K)",
    "eV",
    "eV",
    "eV",
    "eV",
    "GHz",
    "GHz",
    "GHz",
)

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

TARGET_CONVERSION = torch.tensor(
    [
        1.0,            # mu
        1.0,            # alpha
        HAR2EV,         # homo
        HAR2EV,         # lumo
        HAR2EV,         # gap
        1.0,            # r2
        HAR2EV,         # zpve
        HAR2EV,         # u0
        HAR2EV,         # u298
        HAR2EV,         # h298
        HAR2EV,         # g298
        1.0,            # cv
        KCALMOL2EV,     # u0_atom
        KCALMOL2EV,     # u298_atom
        KCALMOL2EV,     # h298_atom
        KCALMOL2EV,     # g298_atom
        1.0,            # A
        1.0,            # B
        1.0,            # C
    ],
    dtype=torch.float32,
)


# ============================================================
# Download
# ============================================================

def download_qm9_raw(
    root: str | Path = "dataset/qm9",
) -> Path:
    """Download the original QM9 raw files."""

    root = Path(root)
    raw_dir = root / "raw"
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sdf_path = raw_dir / "gdb9.sdf"
    csv_path = raw_dir / "gdb9.sdf.csv"
    uncharacterized_path = (
        raw_dir / "uncharacterized.txt"
    )

    if not sdf_path.exists() or not csv_path.exists():

        archive_path = Path(
            download_url(
                QM9_RAW_URL,
                str(raw_dir),
            )
        )

        extract_zip(
            str(archive_path),
            str(raw_dir),
        )

        if archive_path.exists():
            archive_path.unlink()

    if not uncharacterized_path.exists():

        downloaded_path = Path(
            download_url(
                QM9_UNCHARACTERIZED_URL,
                str(raw_dir),
            )
        )

        os.replace(
            downloaded_path,
            uncharacterized_path,
        )

    missing = [
        filename
        for filename in RAW_FILE_NAMES
        if not (raw_dir / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required QM9 raw files are missing: "
            f"{missing}"
        )

    return raw_dir


# ============================================================
# Raw file inspection
# ============================================================

def _human_size(
    num_bytes: int,
) -> str:

    value = float(num_bytes)

    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{num_bytes} B"


def inspect_raw_files(
    root: str | Path = "dataset/qm9",
    csv_rows: int = 3,
    sdf_max_lines: int = 60,
) -> None:
    """Print the original QM9 file layout and contents."""

    raw_dir = download_qm9_raw(
        root
    )

    print("\n=== QM9 raw directory ===")
    print(raw_dir.resolve())

    print("\n=== Raw files ===")

    for filename in RAW_FILE_NAMES:

        path = raw_dir / filename

        print(
            f"{filename:24s} "
            f"{_human_size(path.stat().st_size)}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = raw_dir / "gdb9.sdf.csv"

    df = pd.read_csv(
        csv_path
    )

    print("\n=== gdb9.sdf.csv ===")

    print(
        f"rows    : {len(df):,}"
    )

    print(
        f"columns : {len(df.columns)}"
    )

    print(
        "\ncolumn names:"
    )

    print(
        list(df.columns)
    )

    print(
        f"\nfirst {csv_rows} rows:"
    )

    print(
        df.head(csv_rows).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # uncharacterized.txt
    # --------------------------------------------------------

    print(
        "\n=== uncharacterized.txt preview ==="
    )

    with (
        raw_dir / "uncharacterized.txt"
    ).open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line_idx, line in enumerate(file):

            print(
                line.rstrip()
            )

            if line_idx >= 14:
                break

    # --------------------------------------------------------
    # SDF
    # --------------------------------------------------------

    print(
        "\n=== First molecule block in gdb9.sdf ==="
    )

    with (
        raw_dir / "gdb9.sdf"
    ).open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for line_idx, line in enumerate(file):

            print(
                line.rstrip()
            )

            if line.rstrip() == "$$$$":
                break

            if line_idx + 1 >= sdf_max_lines:
                print(
                    "... [preview truncated]"
                )
                break


# ============================================================
# QM9 Dataset
# ============================================================

class QM9Dataset(Dataset):
    """QM9 loader compatible with Molecule_Project.

    Parameters
    ----------
    root:
        Directory where the raw QM9 files are stored.

    target:
        Target property to return in data.y.
        Example: "homo", "lumo", "gap", "mu".

        If None, data.y contains all 19 targets.

    include_hydrogens:
        If True, explicit H atoms in the QM9 SDF are preserved.
        This is recommended for QM9 because the 3D coordinates include H.

    download:
        Download raw QM9 files automatically if needed.

    Output fields
    -------------
    x:
        [N_atom, 8]

        [atomic_num,
         degree,
         formal_charge,
         num_H,
         aromatic,
         in_ring,
         hybridization_id,
         chirality_id]

    edge_index:
        [2, N_edge]

    edge_attr:
        [N_edge, 4]

        [bond_type_id,
         conjugated,
         in_ring,
         stereo_id]

    pos:
        [N_atom, 3] 3D coordinates

    z:
        [N_atom] atomic numbers

    y:
        selected regression target or all targets

    y_all:
        [1, 19] all targets

    y_raw_csv:
        [1, 19] targets in the original CSV order/units
    """

    target_names: Sequence[str] = TARGET_NAMES
    target_units: Sequence[str] = TARGET_UNITS

    def __init__(
        self,
        root: str | Path = "dataset/qm9",
        target: str | None = None,
        include_hydrogens: bool = True,
        download: bool = True,
    ):
        super().__init__()

        self.root = Path(root)
        self.raw_dir = self.root / "raw"

        self.target = target
        self.include_hydrogens = (
            include_hydrogens
        )

        if download:
            download_qm9_raw(
                self.root
            )

        self.sdf_path = (
            self.raw_dir / "gdb9.sdf"
        )

        self.csv_path = (
            self.raw_dir / "gdb9.sdf.csv"
        )

        self.uncharacterized_path = (
            self.raw_dir / "uncharacterized.txt"
        )

        for path in (
            self.sdf_path,
            self.csv_path,
            self.uncharacterized_path,
        ):

            if not path.exists():

                raise FileNotFoundError(
                    f"QM9 raw file not found: {path}"
                )

        if (
            self.target is not None
            and self.target not in TARGET_NAMES
        ):

            raise ValueError(
                f"Unknown target: {self.target!r}\n"
                f"Available targets: {TARGET_NAMES}"
            )

        self.target_index = (
            None
            if self.target is None
            else TARGET_NAMES.index(
                self.target
            )
        )

        # ----------------------------------------------------
        # 1. Load property table
        # ----------------------------------------------------

        df = pd.read_csv(
            self.csv_path
        )

        self.mol_ids = (
            df.iloc[:, 0]
            .astype(str)
            .tolist()
        )

        raw_targets = torch.tensor(
            df.iloc[:, 1:20]
            .astype(float)
            .to_numpy(),
            dtype=torch.float32,
        )

        self.raw_targets = (
            raw_targets
        )

        # Original:
        # A, B, C, mu, alpha, ..., g298_atom
        #
        # PyG-compatible:
        # mu, alpha, ..., g298_atom, A, B, C

        targets = torch.cat(
            [
                raw_targets[:, 3:],
                raw_targets[:, :3],
            ],
            dim=-1,
        )

        self.targets = (
            targets
            * TARGET_CONVERSION.view(
                1,
                -1,
            )
        )

        # ----------------------------------------------------
        # 2. Read standard QM9 exclusion list
        # ----------------------------------------------------

        self.skip_indices = (
            self._read_uncharacterized_indices()
        )

        self.valid_indices = [
            idx
            for idx in range(len(df))
            if idx not in self.skip_indices
        ]

        # Do not share RDKit supplier between workers.
        self._supplier = None

    def _read_uncharacterized_indices(
        self,
    ) -> set[int]:

        lines = (
            self.uncharacterized_path
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
            .splitlines()
        )

        skip: set[int] = set()

        # Matches the section used in the PyG QM9 loader.
        for line in lines[9:-2]:

            tokens = line.split()

            if not tokens:
                continue

            try:
                one_based_idx = int(
                    tokens[0]
                )
            except ValueError:
                continue

            skip.add(
                one_based_idx - 1
            )

        return skip

    def _get_supplier(
        self,
    ) -> Chem.SDMolSupplier:

        if self._supplier is None:

            self._supplier = (
                Chem.SDMolSupplier(
                    str(self.sdf_path),
                    removeHs=False,
                    sanitize=False,
                )
            )

        return self._supplier

    def __getstate__(
        self,
    ):

        state = self.__dict__.copy()

        state["_supplier"] = None

        return state

    def __len__(
        self,
    ) -> int:

        return len(
            self.valid_indices
        )

    def __getitem__(
        self,
        idx: int,
    ) -> Data:

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):

            raise IndexError(
                f"QM9Dataset index out of range: {idx}"
            )

        raw_idx = (
            self.valid_indices[idx]
        )

        supplier = (
            self._get_supplier()
        )

        mol = supplier[
            raw_idx
        ]

        if mol is None:

            raise ValueError(
                "RDKit failed to read QM9 molecule "
                f"at raw index {raw_idx}"
            )

        mol = Chem.Mol(
            mol
        )

        try:

            Chem.SanitizeMol(
                mol
            )

        except Exception as exc:

            raise ValueError(
                "RDKit sanitization failed for "
                f"QM9 raw index {raw_idx}"
            ) from exc

        if not self.include_hydrogens:

            mol = Chem.RemoveHs(
                mol
            )

        # ----------------------------------------------------
        # 1. Atom features
        # ----------------------------------------------------

        atom_rows = []

        for atom in mol.GetAtoms():

            feat = atom_features(
                atom
            )

            if self.include_hydrogens:
                # With explicit H atoms, GetTotalNumHs() may be 0.
                # Count H neighbors explicitly so the existing project
                # num_H feature remains informative.
                feat[3] = sum(
                    1
                    for neighbor in atom.GetNeighbors()
                    if neighbor.GetAtomicNum() == 1
                )

            atom_rows.append(
                feat
            )

        x = torch.tensor(
            atom_rows,
            dtype=torch.long,
        )

        z = x[:, 0].clone()

        # ----------------------------------------------------
        # 2. 3D coordinates
        # ----------------------------------------------------

        if mol.GetNumConformers() < 1:

            raise ValueError(
                "QM9 molecule has no conformer at "
                f"raw index {raw_idx}"
            )

        pos = torch.tensor(
            mol.GetConformer().GetPositions(),
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # 3. Bonds -> directed PyG graph
        # ----------------------------------------------------

        src = []
        dst = []
        attrs = []

        for bond in mol.GetBonds():

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            feat = bond_features(
                bond
            )

            src.extend(
                [i, j]
            )

            dst.extend(
                [j, i]
            )

            attrs.extend(
                [feat, feat]
            )

        if src:

            edge_index = torch.tensor(
                [src, dst],
                dtype=torch.long,
            )

            edge_attr = torch.tensor(
                attrs,
                dtype=torch.long,
            )

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

            edge_attr = torch.empty(
                (0, 4),
                dtype=torch.long,
            )

        # ----------------------------------------------------
        # 4. Targets
        # ----------------------------------------------------

        y_all = (
            self.targets[raw_idx]
            .view(1, -1)
        )

        y_raw_csv = (
            self.raw_targets[raw_idx]
            .view(1, -1)
        )

        if self.target_index is None:

            y = y_all.clone()

        else:

            y = (
                y_all[
                    0,
                    self.target_index,
                ]
                .view(1)
            )

        # ----------------------------------------------------
        # 5. Metadata
        # ----------------------------------------------------

        smiles = Chem.MolToSmiles(
            mol,
            isomericSmiles=True,
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            z=z,
            y=y,
            y_all=y_all,
            y_raw_csv=y_raw_csv,
            smiles=smiles,
            mol_id=self.mol_ids[raw_idx],
            raw_index=torch.tensor(
                [raw_idx],
                dtype=torch.long,
            ),
        )


# ============================================================
# Print helpers
# ============================================================

def print_dataset_summary(
    dataset: QM9Dataset,
) -> None:

    print("\n=== QM9Dataset ===")

    print(
        f"samples               : "
        f"{len(dataset):,}"
    )

    print(
        f"excluded raw molecules: "
        f"{len(dataset.skip_indices):,}"
    )

    print(
        f"include_hydrogens     : "
        f"{dataset.include_hydrogens}"
    )

    print(
        f"target                : "
        f"{dataset.target}"
    )


def print_sample_summary(
    dataset: QM9Dataset,
    idx: int = 0,
) -> None:

    data = dataset[
        idx
    ]

    print(
        f"\n=== Sample {idx} ==="
    )

    print(data)

    print(
        "mol_id:",
        data.mol_id,
    )

    print(
        "raw_index:",
        int(
            data.raw_index.item()
        ),
    )

    print(
        "smiles:",
        data.smiles,
    )

    print(
        "x:",
        tuple(
            data.x.shape
        ),
    )

    print(
        "edge_index:",
        tuple(
            data.edge_index.shape
        ),
    )

    print(
        "edge_attr:",
        tuple(
            data.edge_attr.shape
        ),
    )

    print(
        "pos:",
        tuple(
            data.pos.shape
        ),
    )

    print(
        "\n=== 19 QM9 targets ==="
    )

    values = (
        data.y_all
        .view(-1)
        .tolist()
    )

    for name, unit, value in zip(
        TARGET_NAMES,
        TARGET_UNITS,
        values,
    ):

        print(
            f"{name:10s} "
            f"{value: .8f} "
            f"[{unit}]"
        )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Download, inspect, and load QM9 "
            "for Molecule_Project."
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        default="dataset/qm9",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=[
            *TARGET_NAMES,
        ],
    )

    parser.add_argument(
        "--no-hydrogens",
        action="store_true",
    )

    parser.add_argument(
        "--inspect-raw",
        action="store_true",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    download_qm9_raw(
        args.root
    )

    if args.inspect_raw:

        inspect_raw_files(
            args.root
        )

    if args.download_only:
        return

    dataset = QM9Dataset(
        root=args.root,
        target=args.target,
        include_hydrogens=(
            not args.no_hydrogens
        ),
        download=False,
    )

    print_dataset_summary(
        dataset
    )

    print_sample_summary(
        dataset,
        args.sample_index,
    )


if __name__ == "__main__":
    main()
