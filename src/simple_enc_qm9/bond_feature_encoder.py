#!/usr/bin/env python3
"""Dynamic bond feature encoder for the QM9 molecule encoder.

The raw bond tensor can remain:

    edge_attr: [N_edge, 4]

with the current project convention:

    0: bond_type_id
    1: conjugated
    2: in_ring
    3: stereo_id

The enabled bond features are selected through feature_schema.py, while the
final latent representation always has shape:

    H_edge: [N_edge, embed_dim]
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .feature_schema import (
    DEFAULT_BOND_FEATURE_SCHEMA,
    FeatureSpec,
    get_enabled_features,
    required_input_width,
    validate_schema,
)


class DynamicBondFeatureEncoder(nn.Module):
    """Encode a configurable set of bond features."""

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
            raise ValueError("embed_dim must be >= 1")

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
        # 1. Resolve feature schema
        # ----------------------------------------------------

        if feature_schema is None:
            feature_schema = DEFAULT_BOND_FEATURE_SCHEMA

        self.feature_schema = dict(
            get_enabled_features(feature_schema)
        )

        validate_schema(
            self.feature_schema
        )

        self.feature_names = list(
            self.feature_schema.keys()
        )

        self.input_width = required_input_width(
            self.feature_schema
        )

        # ----------------------------------------------------
        # 2. Build one encoder per feature automatically
        # ----------------------------------------------------

        self.feature_encoders = nn.ModuleDict()

        for name, spec in self.feature_schema.items():

            if spec.feature_type == "categorical":

                assert spec.num_embeddings is not None

                self.feature_encoders[name] = nn.Embedding(
                    num_embeddings=spec.num_embeddings,
                    embedding_dim=embed_dim,
                )

            elif spec.feature_type == "continuous":

                self.feature_encoders[name] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )

            else:
                raise ValueError(
                    f"Unsupported feature type "
                    f"for {name!r}: {spec.feature_type!r}"
                )

        # ----------------------------------------------------
        # 3. Optional concat projection
        # ----------------------------------------------------

        if fusion == "concat":

            self.concat_projection = nn.Linear(
                len(self.feature_names) * embed_dim,
                embed_dim,
            )

        else:
            self.concat_projection = None

        # ----------------------------------------------------
        # 4. Output normalization
        # ----------------------------------------------------

        self.dropout = nn.Dropout(dropout)

        if layer_norm:
            self.norm: nn.Module = nn.LayerNorm(
                embed_dim
            )
        else:
            self.norm = nn.Identity()


    def _encode_categorical_feature(
        self,
        edge_attr: torch.Tensor,
        name: str,
        spec: FeatureSpec,
    ) -> torch.Tensor:
        """Encode one categorical bond feature."""

        raw_values = edge_attr[:, spec.column]

        if torch.is_floating_point(raw_values):

            rounded = raw_values.round()

            if not torch.allclose(
                raw_values,
                rounded,
            ):
                raise ValueError(
                    f"Categorical bond feature {name!r} "
                    "contains non-integer values."
                )

            raw_values = rounded

        indices = (
            raw_values.long()
            + spec.offset
        )

        assert spec.num_embeddings is not None

        if torch.any(indices < 0):
            raise ValueError(
                f"Bond feature {name!r} produced "
                "a negative embedding index."
            )

        if torch.any(
            indices >= spec.num_embeddings
        ):
            raise ValueError(
                f"Bond feature {name!r} contains "
                "an embedding index outside its vocabulary."
            )

        return self.feature_encoders[name](
            indices
        )


    def _encode_continuous_feature(
        self,
        edge_attr: torch.Tensor,
        name: str,
        spec: FeatureSpec,
    ) -> torch.Tensor:
        """Encode one continuous scalar bond feature."""

        values = (
            edge_attr[:, spec.column]
            .float()
            .unsqueeze(-1)
        )

        return self.feature_encoders[name](
            values
        )


    def forward(
        self,
        edge_attr: torch.Tensor,
        return_feature_embeddings: bool = False,
    ):
        """Encode raw bond features.

        Parameters
        ----------
        edge_attr:
            [N_edge, F_raw]

        Returns
        -------
        H_edge:
            [N_edge, embed_dim]
        """

        # ----------------------------------------------------
        # 1. Input validation
        # ----------------------------------------------------

        if edge_attr.ndim != 2:
            raise ValueError(
                "Expected edge_attr shape "
                "[N_edge, F_raw], "
                f"got {tuple(edge_attr.shape)}"
            )

        if edge_attr.size(1) < self.input_width:
            raise ValueError(
                "Raw bond feature tensor is too narrow. "
                f"Schema requires at least "
                f"{self.input_width} columns, "
                f"got {edge_attr.size(1)}."
            )

        # ----------------------------------------------------
        # 2. Zero-edge graph handling
        # ----------------------------------------------------

        if edge_attr.size(0) == 0:

            H_edge = torch.empty(
                (0, self.embed_dim),
                dtype=torch.float32,
                device=edge_attr.device,
            )

            if return_feature_embeddings:

                empty_features = {
                    name: torch.empty(
                        (0, self.embed_dim),
                        dtype=torch.float32,
                        device=edge_attr.device,
                    )
                    for name in self.feature_names
                }

                return (
                    H_edge,
                    empty_features,
                )

            return H_edge

        # ----------------------------------------------------
        # 3. Encode each enabled feature
        # ----------------------------------------------------

        feature_embeddings = {}

        for name in self.feature_names:

            spec = self.feature_schema[name]

            if spec.feature_type == "categorical":

                h_feature = (
                    self._encode_categorical_feature(
                        edge_attr=edge_attr,
                        name=name,
                        spec=spec,
                    )
                )

            else:

                h_feature = (
                    self._encode_continuous_feature(
                        edge_attr=edge_attr,
                        name=name,
                        spec=spec,
                    )
                )

            feature_embeddings[name] = (
                h_feature
            )

        embeddings = [
            feature_embeddings[name]
            for name in self.feature_names
        ]

        # ----------------------------------------------------
        # 4. Feature fusion
        # ----------------------------------------------------

        if self.fusion == "sum":

            H_edge = torch.stack(
                embeddings,
                dim=0,
            ).sum(dim=0)

        elif self.fusion == "mean":

            H_edge = torch.stack(
                embeddings,
                dim=0,
            ).mean(dim=0)

        elif self.fusion == "concat":

            H_concat = torch.cat(
                embeddings,
                dim=-1,
            )

            assert (
                self.concat_projection
                is not None
            )

            H_edge = self.concat_projection(
                H_concat
            )

        else:
            raise RuntimeError(
                "Unexpected fusion mode."
            )

        # ----------------------------------------------------
        # 5. Final edge latent
        # ----------------------------------------------------

        H_edge = self.norm(
            H_edge
        )

        H_edge = self.dropout(
            H_edge
        )

        if return_feature_embeddings:
            return (
                H_edge,
                feature_embeddings,
            )

        return H_edge


# ============================================================
# Standalone test
# ============================================================

def main() -> None:

    # [bond_type_id, conjugated, in_ring, stereo_id]

    edge_attr = torch.tensor(
        [
            [0, 0, 0, 0],  # single
            [0, 0, 0, 0],  # reverse direction
            [1, 1, 0, 0],  # double + conjugated
            [1, 1, 0, 0],  # reverse direction
            [3, 1, 1, 0],  # aromatic ring bond
            [3, 1, 1, 0],  # reverse direction
        ],
        dtype=torch.long,
    )

    encoder = DynamicBondFeatureEncoder(
        embed_dim=128,
        feature_schema=(
            DEFAULT_BOND_FEATURE_SCHEMA
        ),
        fusion="mean",
    )

    (
        H_edge,
        feature_embeddings,
    ) = encoder(
        edge_attr,
        return_feature_embeddings=True,
    )

    print("=" * 72)
    print("Dynamic Bond Feature Encoder")
    print("=" * 72)

    print(
        "raw edge_attr shape:",
        tuple(edge_attr.shape),
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
        "H_edge shape:",
        tuple(H_edge.shape),
    )

    print(
        "\nPer-feature embedding shapes:"
    )

    for name, h_feature in (
        feature_embeddings.items()
    ):

        print(
            f"  {name:20s}",
            tuple(h_feature.shape),
        )


if __name__ == "__main__":
    main()
