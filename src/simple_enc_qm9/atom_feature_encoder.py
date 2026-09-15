#!/usr/bin/env python3
"""Dynamic atom feature encoder for the QM9 molecule encoder.

This module reads the feature definitions from ``feature_schema.py`` and
constructs the corresponding neural encoders automatically.

Key idea
--------
The raw atom tensor may still be:

    x: [N_atom, 8]

but the model does not need to use all 8 columns.

For example, the schema may select only:

    atomic_num
    degree
    formal_charge
    aromatic
    hybridization_id

The output dimension remains fixed:

    H_atom: [N_atom, embed_dim]

This gives us:

    configurable raw features
            ↓
    common latent dimension D

which is useful for building a more flexible Molecule Encoder.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .feature_schema import (
    DEFAULT_ATOM_FEATURE_SCHEMA,
    FeatureSpec,
    get_enabled_features,
    required_input_width,
    validate_schema,
)


# ============================================================
# Dynamic Atom Feature Encoder
# ============================================================

class DynamicAtomFeatureEncoder(nn.Module):
    """Encode a configurable set of atom features.

    Parameters
    ----------
    embed_dim:
        Output latent dimension for every atom.

    feature_schema:
        Mapping from feature name to ``FeatureSpec``.

        If None, ``DEFAULT_ATOM_FEATURE_SCHEMA`` is used.

    fusion:
        How individual feature embeddings are combined.

        ``"sum"``
            Sum all feature embeddings.

        ``"mean"``
            Average all feature embeddings.
            This is a useful default when the number of enabled features
            may change between experiments.

        ``"concat"``
            Concatenate all feature embeddings and project them back to
            ``embed_dim``.

    dropout:
        Dropout applied after feature fusion.

    layer_norm:
        Apply LayerNorm to the final atom representation.

    Input
    -----
    x:
        Raw atom feature tensor.

        Shape:
            [N_atom, F_raw]

        F_raw does not need to equal the number of enabled features.
        It only needs to contain every source column referenced by the schema.

    Output
    ------
    H_atom:
        Dense atom representation.

        Shape:
            [N_atom, embed_dim]
    """

    SUPPORTED_FUSIONS = {
        "sum",
        "mean",
        "concat",
    }

    def __init__(
        self,
        embed_dim: int = 128,
        feature_schema: Mapping[str, FeatureSpec] | None = None,
        fusion: str = "mean",
        dropout: float = 0.0,
        layer_norm: bool = True,
    ):
        super().__init__()

        if embed_dim < 1:
            raise ValueError(
                "embed_dim must be >= 1"
            )

        if fusion not in self.SUPPORTED_FUSIONS:
            raise ValueError(
                f"Unsupported fusion mode: {fusion!r}. "
                f"Choose from {sorted(self.SUPPORTED_FUSIONS)}"
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1"
            )

        self.embed_dim = embed_dim
        self.fusion = fusion

        # ----------------------------------------------------
        # 1. Resolve schema
        # ----------------------------------------------------

        if feature_schema is None:
            feature_schema = (
                DEFAULT_ATOM_FEATURE_SCHEMA
            )

        # Only features explicitly enabled are used.
        self.feature_schema = dict(
            get_enabled_features(
                feature_schema
            )
        )

        validate_schema(
            self.feature_schema
        )

        self.feature_names = list(
            self.feature_schema.keys()
        )

        self.input_width = (
            required_input_width(
                self.feature_schema
            )
        )

        # ----------------------------------------------------
        # 2. Build one encoder per feature
        # ----------------------------------------------------

        self.feature_encoders = (
            nn.ModuleDict()
        )

        for name, spec in (
            self.feature_schema.items()
        ):

            if (
                spec.feature_type
                == "categorical"
            ):

                assert (
                    spec.num_embeddings
                    is not None
                )

                self.feature_encoders[
                    name
                ] = nn.Embedding(
                    num_embeddings=(
                        spec.num_embeddings
                    ),
                    embedding_dim=embed_dim,
                )

            elif (
                spec.feature_type
                == "continuous"
            ):

                # A simple continuous-feature projection.
                #
                # scalar
                #   ↓
                # Linear(1 -> D)
                #   ↓
                # SiLU
                #   ↓
                # Linear(D -> D)

                self.feature_encoders[
                    name
                ] = nn.Sequential(
                    nn.Linear(
                        1,
                        embed_dim,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        embed_dim,
                        embed_dim,
                    ),
                )

            else:
                raise ValueError(
                    f"Unsupported feature type "
                    f"for {name!r}: "
                    f"{spec.feature_type!r}"
                )

        # ----------------------------------------------------
        # 3. Optional concat projection
        # ----------------------------------------------------

        if fusion == "concat":

            concat_dim = (
                len(self.feature_names)
                * embed_dim
            )

            self.concat_projection = (
                nn.Linear(
                    concat_dim,
                    embed_dim,
                )
            )

        else:

            self.concat_projection = None

        # ----------------------------------------------------
        # 4. Output normalization
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        if layer_norm:

            self.norm: nn.Module = (
                nn.LayerNorm(
                    embed_dim
                )
            )

        else:

            self.norm = nn.Identity()


    # ========================================================
    # Feature encoding helpers
    # ========================================================

    def _encode_categorical_feature(
        self,
        x: torch.Tensor,
        name: str,
        spec: FeatureSpec,
    ) -> torch.Tensor:
        """Encode one categorical feature."""

        raw_values = x[
            :,
            spec.column,
        ]

        # ----------------------------------------------------
        # Require integer-valued categorical data
        # ----------------------------------------------------

        if torch.is_floating_point(
            raw_values
        ):

            rounded = (
                raw_values.round()
            )

            if not torch.allclose(
                raw_values,
                rounded,
            ):
                raise ValueError(
                    f"Categorical feature {name!r} "
                    "contains non-integer values."
                )

            raw_values = rounded

        indices = (
            raw_values.long()
            + spec.offset
        )

        assert (
            spec.num_embeddings
            is not None
        )

        # ----------------------------------------------------
        # Range check before nn.Embedding
        # ----------------------------------------------------

        if torch.any(
            indices < 0
        ):
            bad_min = int(
                indices.min().item()
            )

            raise ValueError(
                f"Feature {name!r} produced "
                f"negative embedding index "
                f"{bad_min}. "
                f"Check offset={spec.offset}."
            )

        if torch.any(
            indices
            >= spec.num_embeddings
        ):
            bad_max = int(
                indices.max().item()
            )

            raise ValueError(
                f"Feature {name!r} produced "
                f"embedding index {bad_max}, "
                f"but num_embeddings="
                f"{spec.num_embeddings}."
            )

        encoder = (
            self.feature_encoders[
                name
            ]
        )

        return encoder(
            indices
        )


    def _encode_continuous_feature(
        self,
        x: torch.Tensor,
        name: str,
        spec: FeatureSpec,
    ) -> torch.Tensor:
        """Encode one continuous scalar feature."""

        values = (
            x[
                :,
                spec.column,
            ]
            .float()
            .unsqueeze(-1)
        )

        encoder = (
            self.feature_encoders[
                name
            ]
        )

        return encoder(
            values
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        return_feature_embeddings: bool = False,
    ):
        """Encode raw atom features."""

        # ----------------------------------------------------
        # 1. Basic input validation
        # ----------------------------------------------------

        if x.ndim != 2:
            raise ValueError(
                "Expected x shape "
                "[N_atom, F_raw], "
                f"got {tuple(x.shape)}"
            )

        if x.size(1) < self.input_width:
            raise ValueError(
                "Raw atom feature tensor is too narrow. "
                f"Schema requires at least "
                f"{self.input_width} columns, "
                f"got {x.size(1)}."
            )

        # ----------------------------------------------------
        # 2. Encode each enabled feature
        # ----------------------------------------------------

        feature_embeddings: dict[
            str,
            torch.Tensor,
        ] = {}

        for name in self.feature_names:

            spec = (
                self.feature_schema[
                    name
                ]
            )

            if (
                spec.feature_type
                == "categorical"
            ):

                h_feature = (
                    self._encode_categorical_feature(
                        x=x,
                        name=name,
                        spec=spec,
                    )
                )

            else:

                h_feature = (
                    self._encode_continuous_feature(
                        x=x,
                        name=name,
                        spec=spec,
                    )
                )

            feature_embeddings[
                name
            ] = h_feature

        # ----------------------------------------------------
        # 3. Fuse feature representations
        # ----------------------------------------------------

        embeddings = [
            feature_embeddings[name]
            for name in self.feature_names
        ]

        if self.fusion == "sum":

            H_atom = torch.stack(
                embeddings,
                dim=0,
            ).sum(
                dim=0
            )

        elif self.fusion == "mean":

            H_atom = torch.stack(
                embeddings,
                dim=0,
            ).mean(
                dim=0
            )

        elif self.fusion == "concat":

            H_concat = torch.cat(
                embeddings,
                dim=-1,
            )

            assert (
                self.concat_projection
                is not None
            )

            H_atom = (
                self.concat_projection(
                    H_concat
                )
            )

        else:
            raise RuntimeError(
                "Unexpected fusion mode."
            )

        # ----------------------------------------------------
        # 4. Normalize output
        # ----------------------------------------------------

        H_atom = self.norm(
            H_atom
        )

        H_atom = self.dropout(
            H_atom
        )

        # ----------------------------------------------------
        # 5. Return
        # ----------------------------------------------------

        if return_feature_embeddings:

            return (
                H_atom,
                feature_embeddings,
            )

        return H_atom


# ============================================================
# Standalone example
# ============================================================

def main() -> None:
    """Small test without loading the full QM9 dataset."""

    # Example raw atom features:
    #
    # [atomic_num,
    #  degree,
    #  formal_charge,
    #  num_h,
    #  aromatic,
    #  in_ring,
    #  hybridization_id,
    #  chirality_id]

    x = torch.tensor(
        [
            # Carbon
            [6, 4, 0, 4, 0, 0, 4, 0],

            # Oxygen
            [8, 2, 0, 0, 0, 0, 4, 0],

            # Nitrogen
            [7, 3, 0, 0, 0, 0, 3, 0],
        ],
        dtype=torch.long,
    )

    encoder = DynamicAtomFeatureEncoder(
        embed_dim=128,
        feature_schema=(
            DEFAULT_ATOM_FEATURE_SCHEMA
        ),
        fusion="mean",
    )

    (
        H_atom,
        feature_embeddings,
    ) = encoder(
        x,
        return_feature_embeddings=True,
    )

    print(
        "=" * 72
    )
    print(
        "Dynamic Atom Feature Encoder"
    )
    print(
        "=" * 72
    )

    print(
        "raw x shape:",
        tuple(
            x.shape
        ),
    )

    print(
        "enabled features:",
        encoder.feature_names,
    )

    print(
        "fusion:",
        encoder.fusion,
    )

    print(
        "H_atom shape:",
        tuple(
            H_atom.shape
        ),
    )

    print(
        "\nPer-feature embedding shapes:"
    )

    for (
        name,
        h_feature,
    ) in feature_embeddings.items():

        print(
            f"  {name:20s}",
            tuple(
                h_feature.shape
            ),
        )


if __name__ == "__main__":
    main()
