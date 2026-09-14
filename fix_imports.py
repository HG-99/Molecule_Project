#!/usr/bin/env python3
"""
Fix import paths after the Molecule_Project folder reorganization.

Expected repository structure:

Molecule_Project/
├── main/
│   ├── simple_mpnn/
│   └── simple_enc_qm9/
└── src/
    ├── simple_mpnn/
    └── simple_enc_qm9/

This script:
1. Fixes PROJECT_ROOT in main/simple_mpnn/*.py
2. Rewrites old `src.<module>` imports to the new package paths
3. Converts imports inside src/simple_mpnn to package-relative imports
4. Fixes the QM9 loader's cross-package import
5. Adds minimal __init__.py files so the reorganized directories are explicit packages
6. Parses every Python file with ast to catch syntax errors

Run from the repository root:

    python fix_imports.py

or:

    python fix_imports.py --root /path/to/Molecule_Project
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "main/simple_mpnn/train_esol.py": [
        (
            ".parents[1]",
            ".parents[2]",
        ),
        (
            "from src.dataset import ESOLDataset",
            "from src.simple_mpnn.dataset_esol import ESOLDataset",
        ),
        (
            "from src.molecule_encoder import MoleculeEncoder",
            "from src.simple_mpnn.molecule_encoder import MoleculeEncoder",
        ),
        (
            "from src.task_head import RegressionHead",
            "from src.simple_mpnn.task_head import RegressionHead",
        ),
    ],
    "main/simple_mpnn/eval_esol.py": [
        (
            ".parents[1]",
            ".parents[2]",
        ),
        (
            "from src.dataset import ESOLDataset",
            "from src.simple_mpnn.dataset_esol import ESOLDataset",
        ),
        (
            "from src.molecule_encoder import MoleculeEncoder",
            "from src.simple_mpnn.molecule_encoder import MoleculeEncoder",
        ),
        (
            "from src.task_head import RegressionHead",
            "from src.simple_mpnn.task_head import RegressionHead",
        ),
    ],
    "main/simple_mpnn/pipeline_test.py": [
        (
            ".parents[1]",
            ".parents[2]",
        ),
        (
            "from src.smiles_to_graph import (",
            "from src.simple_mpnn.smiles_to_graph import (",
        ),
        (
            "from src.feature_encoder import (",
            "from src.simple_mpnn.feature_encoder import (",
        ),
        (
            "from src.mpnn import (",
            "from src.simple_mpnn.mpnn import (",
        ),
        (
            "from src.readout import (",
            "from src.simple_mpnn.readout import (",
        ),
    ],
    "main/simple_mpnn/pipeline_integration_test.py": [
        (
            ".parents[1]",
            ".parents[2]",
        ),
        (
            "from src.smiles_to_graph import (",
            "from src.simple_mpnn.smiles_to_graph import (",
        ),
        (
            "from src.molecule_encoder import (",
            "from src.simple_mpnn.molecule_encoder import (",
        ),
    ],
    "main/simple_mpnn/pipeline_integration_multi_test.py": [
        (
            ".parents[1]",
            ".parents[2]",
        ),
        (
            "from src.smiles_to_graph import (",
            "from src.simple_mpnn.smiles_to_graph import (",
        ),
        (
            "from src.molecule_encoder import (",
            "from src.simple_mpnn.molecule_encoder import (",
        ),
    ],
    "src/simple_mpnn/dataset_esol.py": [
        (
            "from src.smiles_to_graph import smiles_to_pyg",
            "from .smiles_to_graph import smiles_to_pyg",
        ),
    ],
    "src/simple_mpnn/molecule_encoder.py": [
        (
            "from src.feature_encoder import (",
            "from .feature_encoder import (",
        ),
        (
            "from src.mpnn import (",
            "from .mpnn import (",
        ),
        (
            "from src.readout import (",
            "from .readout import (",
        ),
    ],
    "src/simple_enc_qm9/dataset_qm9.py": [
        (
            "python -m src.dataset_qm9 --inspect-raw",
            "python -m src.simple_enc_qm9.dataset_qm9 --inspect-raw",
        ),
        (
            "from src.smiles_to_graph import atom_features, bond_features",
            "from src.simple_mpnn.smiles_to_graph import atom_features, bond_features",
        ),
    ],
}


INIT_FILES: dict[str, str] = {
    "src/__init__.py": '"""Source package for Molecule_Project."""\n',
    "src/simple_mpnn/__init__.py": (
        '"""Simple MPNN molecular encoder package."""\n'
    ),
    "src/simple_enc_qm9/__init__.py": (
        '"""QM9-oriented molecular encoder experiments."""\n'
    ),
    "main/__init__.py": '"""Executable experiment modules."""\n',
    "main/simple_mpnn/__init__.py": (
        '"""Simple MPNN experiment entry points."""\n'
    ),
    "main/simple_enc_qm9/__init__.py": (
        '"""QM9 experiment entry points."""\n'
    ),
}


def find_repo_root(start: Path) -> Path:
    start = start.resolve()

    candidates = [start, *start.parents]

    for candidate in candidates:
        if (
            (candidate / "src" / "simple_mpnn").is_dir()
            and (candidate / "src" / "simple_enc_qm9").is_dir()
            and (candidate / "main" / "simple_mpnn").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Molecule_Project repository root. "
        "Use --root to specify it explicitly."
    )


def apply_replacements(repo_root: Path) -> list[Path]:
    changed: list[Path] = []

    for relative_path, replacements in REPLACEMENTS.items():
        path = repo_root / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Expected project file does not exist: {relative_path}"
            )

        original = path.read_text(encoding="utf-8")
        updated = original

        for old, new in replacements:
            if old in updated:
                updated = updated.replace(old, new)
            elif new in updated:
                # Already fixed: keep going.
                continue
            else:
                raise RuntimeError(
                    f"Could not find expected import/path text in "
                    f"{relative_path!r}:\n\n{old}"
                )

        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)

    return changed


def create_init_files(repo_root: Path) -> list[Path]:
    created: list[Path] = []

    for relative_path, content in INIT_FILES.items():
        path = repo_root / relative_path

        if path.exists():
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)

    return created


def validate_python_syntax(repo_root: Path) -> list[Path]:
    checked: list[Path] = []

    for base in ("src", "main"):
        for path in sorted((repo_root / base).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                raise SyntaxError(
                    f"Syntax validation failed for {path.relative_to(repo_root)}:\n"
                    f"{exc}"
                ) from exc

            checked.append(path)

    return checked


def print_import_summary(repo_root: Path) -> None:
    print("\n=== Key import paths after refactor ===")
    print("ESOL dataset:")
    print("  from src.simple_mpnn.dataset_esol import ESOLDataset")
    print("Molecule encoder:")
    print("  from src.simple_mpnn.molecule_encoder import MoleculeEncoder")
    print("Task head:")
    print("  from src.simple_mpnn.task_head import RegressionHead")
    print("QM9 dataset:")
    print("  from src.simple_enc_qm9.dataset_qm9 import QM9Dataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Path to the Molecule_Project repository.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(args.root)

    print(f"Repository root: {repo_root}")

    changed = apply_replacements(repo_root)
    created = create_init_files(repo_root)
    checked = validate_python_syntax(repo_root)

    print("\n=== Updated files ===")
    if changed:
        for path in changed:
            print(f"  modified: {path.relative_to(repo_root)}")
    else:
        print("  No import replacements were needed.")

    if created:
        for path in created:
            print(f"  created : {path.relative_to(repo_root)}")

    print(f"\nSyntax check: PASS ({len(checked)} Python files)")

    print_import_summary(repo_root)

    print("\n=== Recommended smoke tests ===")
    print("python main/simple_mpnn/pipeline_test.py")
    print("python main/simple_mpnn/pipeline_integration_test.py")
    print("python main/simple_mpnn/pipeline_integration_multi_test.py")
    print(
        "python -m src.simple_enc_qm9.dataset_qm9 "
        "--root dataset/qm9 --inspect-raw --sample-index 0"
    )

    print("\nThen inspect the diff:")
    print("git diff -- main src")


if __name__ == "__main__":
    main()
