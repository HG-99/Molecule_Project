#!/usr/bin/env python3
"""Feature schema definitions for the QM9 molecule encoder.

This module removes the hard-coded assumption that every atom encoder must
always use exactly 8 atom features and every bond encoder exactly 4 features.

The raw QM9 loader still produces fixed tensors:

    data.x         : [N_atom, 8]
    data.edge_attr : [N_edge, 4]

but the model can choose only a subset of those raw features by changing the
schema, without rewriting the encoder implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


FeatureType = Literal["categorical", "continuous"]


@dataclass(frozen=True)
class FeatureSpec:
    """Description of one raw molecular feature."""

    name: str
    feature_type: FeatureType
    column: int
    num_embeddings: int | None = None
    offset: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.column < 0:
            raise ValueError(
                f"{self.name}: column must be >= 0"
            )

        if self.feature_type == "categorical":
            if self.num_embeddings is None:
                raise ValueError(
                    f"{self.name}: categorical feature requires "
                    "num_embeddings"
                )

            if self.num_embeddings < 1:
                raise ValueError(
                    f"{self.name}: num_embeddings must be >= 1"
                )

        elif self.feature_type == "continuous":
            if self.num_embeddings is not None:
                raise ValueError(
                    f"{self.name}: continuous feature must not define "
                    "num_embeddings"
                )

            if self.offset != 0:
                raise ValueError(
                    f"{self.name}: continuous feature should not use offset"
                )

        else:
            raise ValueError(
                f"{self.name}: unsupported feature_type="
                f"{self.feature_type!r}"
            )


# ============================================================
# Atom features
# ============================================================

DEFAULT_ATOM_FEATURE_SCHEMA: dict[str, FeatureSpec] = {
    "atomic_num": FeatureSpec(
        name="atomic_num",
        feature_type="categorical",
        column=0,
        num_embeddings=119,
    ),

    "degree": FeatureSpec(
        name="degree",
        feature_type="categorical",
        column=1,
        num_embeddings=16,
    ),

    "formal_charge": FeatureSpec(
        name="formal_charge",
        feature_type="categorical",
        column=2,
        num_embeddings=11,
        offset=5,
    ),

    "num_h": FeatureSpec(
        name="num_h",
        feature_type="categorical",
        column=3,
        num_embeddings=16,
    ),

    "aromatic": FeatureSpec(
        name="aromatic",
        feature_type="categorical",
        column=4,
        num_embeddings=2,
    ),

    "in_ring": FeatureSpec(
        name="in_ring",
        feature_type="categorical",
        column=5,
        num_embeddings=2,
    ),

    "hybridization_id": FeatureSpec(
        name="hybridization_id",
        feature_type="categorical",
        column=6,
        num_embeddings=7,
    ),

    "chirality_id": FeatureSpec(
        name="chirality_id",
        feature_type="categorical",
        column=7,
        num_embeddings=4,
    ),
}


# ============================================================
# Bond features
# ============================================================

DEFAULT_BOND_FEATURE_SCHEMA: dict[str, FeatureSpec] = {
    "bond_type_id": FeatureSpec(
        name="bond_type_id",
        feature_type="categorical",
        column=0,
        num_embeddings=5,
    ),

    "conjugated": FeatureSpec(
        name="conjugated",
        feature_type="categorical",
        column=1,
        num_embeddings=2,
    ),

    "in_ring": FeatureSpec(
        name="in_ring",
        feature_type="categorical",
        column=2,
        num_embeddings=2,
    ),

    "stereo_id": FeatureSpec(
        name="stereo_id",
        feature_type="categorical",
        column=3,
        num_embeddings=6,
    ),
}


# ============================================================
# Schema utilities
# ============================================================

def get_enabled_features(
    schema: Mapping[str, FeatureSpec],
) -> dict[str, FeatureSpec]:
    """Return enabled features only."""

    return {
        name: spec
        for name, spec in schema.items()
        if spec.enabled
    }


def select_features(
    schema: Mapping[str, FeatureSpec],
    feature_names: Sequence[str],
) -> dict[str, FeatureSpec]:
    """Create a schema containing only selected features."""

    missing = [
        name
        for name in feature_names
        if name not in schema
    ]

    if missing:
        raise KeyError(
            "Unknown feature name(s): "
            f"{missing}. "
            f"Available={list(schema.keys())}"
        )

    return {
        name: schema[name]
        for name in feature_names
    }


def validate_schema(
    schema: Mapping[str, FeatureSpec],
) -> None:
    """Validate schema consistency."""

    if not schema:
        raise ValueError(
            "Feature schema must contain at least one feature."
        )

    used_columns: dict[int, str] = {}

    for name, spec in schema.items():
        if name != spec.name:
            raise ValueError(
                "Schema key and FeatureSpec.name must match: "
                f"key={name!r}, spec.name={spec.name!r}"
            )

        if spec.column in used_columns:
            raise ValueError(
                f"Column {spec.column} is assigned to both "
                f"{used_columns[spec.column]!r} and {name!r}."
            )

        used_columns[spec.column] = name


def required_input_width(
    schema: Mapping[str, FeatureSpec],
) -> int:
    """Return minimum raw input width required by selected columns."""

    validate_schema(schema)

    return (
        max(
            spec.column
            for spec in schema.values()
        )
        + 1
    )


def print_schema(
    schema: Mapping[str, FeatureSpec],
    title: str = "Feature Schema",
) -> None:
    """Pretty-print schema contents."""

    validate_schema(schema)

    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"{'name':22s} "
        f"{'type':12s} "
        f"{'column':>8s} "
        f"{'vocab':>8s} "
        f"{'offset':>8s}"
    )

    print("-" * 72)

    for name, spec in schema.items():
        vocab = (
            "-"
            if spec.num_embeddings is None
            else str(spec.num_embeddings)
        )

        print(
            f"{name:22s} "
            f"{spec.feature_type:12s} "
            f"{spec.column:8d} "
            f"{vocab:>8s} "
            f"{spec.offset:8d}"
        )


# ============================================================
# Example subset
# ============================================================

COMPACT_ATOM_FEATURE_SCHEMA = select_features(
    DEFAULT_ATOM_FEATURE_SCHEMA,
    [
        "atomic_num",
        "degree",
        "formal_charge",
        "aromatic",
        "hybridization_id",
    ],
)


# ============================================================
# Standalone test
# ============================================================

def main() -> None:
    print_schema(
        DEFAULT_ATOM_FEATURE_SCHEMA,
        title="Default QM9 Atom Feature Schema",
    )

    print()

    print_schema(
        DEFAULT_BOND_FEATURE_SCHEMA,
        title="Default QM9 Bond Feature Schema",
    )

    print()

    print_schema(
        COMPACT_ATOM_FEATURE_SCHEMA,
        title="Compact Atom Feature Schema Example",
    )

    print()

    print(
        "Default atom feature count:",
        len(DEFAULT_ATOM_FEATURE_SCHEMA),
    )

    print(
        "Compact atom feature count:",
        len(COMPACT_ATOM_FEATURE_SCHEMA),
    )

    print(
        "Compact schema raw input width >=",
        required_input_width(
            COMPACT_ATOM_FEATURE_SCHEMA
        ),
    )


if __name__ == "__main__":
    main()
