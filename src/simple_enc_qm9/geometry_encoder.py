#!/usr/bin/env python3
"""3D geometry encoder for the QM9 molecule encoder.

This module converts atomic 3D coordinates into edge-level geometric
representations.

Current role
------------
The current QM9 loader provides:

    pos:
        [N_atom, 3]

    edge_index:
        [2, N_edge]

For every directed edge:

    src -> dst

we compute:

    displacement:
        r_dst - r_src

    distance:
        ||r_dst - r_src||_2

    direction:
        (r_dst - r_src) / ||r_dst - r_src||_2

The scalar distance is then expanded with Gaussian radial basis functions
(RBFs) and projected into a fixed latent dimension:

    distance
        ↓
    Gaussian RBF
        ↓
    [N_edge, num_rbf]
        ↓
    Geometry MLP
        ↓
    H_geo
        [N_edge, embed_dim]

Important design decision
-------------------------
``H_geo`` is built from *distance only*.

Distance is invariant to global translation and rotation:

    d_ij = ||r_j - r_i||

while direction changes when the molecule rotates.

Therefore, this first geometry encoder keeps the scalar latent representation
rotation-invariant and returns direction separately for a future
equivariant/vector message-passing module.

This gives a clean learning path:

    Distance/RBF encoder
        ↓
    scalar geometry latent
        ↓
    later: direction-aware equivariant message passing

A radius graph is intentionally NOT constructed here. The encoder simply
encodes whichever edges are supplied through ``edge_index``.

Later, a separate ``graph_builder.py`` can construct spatial neighbors such as:

    d_ij < cutoff

without coupling graph construction to feature encoding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ============================================================
# Gaussian Radial Basis Expansion
# ============================================================

class GaussianRBF(nn.Module):
    """Expand scalar distances into Gaussian radial basis features.

    Parameters
    ----------
    num_rbf:
        Number of Gaussian basis functions.

    min_distance:
        Minimum RBF center in Angstrom.

    cutoff:
        Maximum RBF center / spatial cutoff in Angstrom.

    trainable:
        If True, RBF centers and gamma become learnable parameters.
        For the first educational implementation, False is recommended.

    Example
    -------
    A distance

        d = 1.42 Å

    becomes

        phi(d) =
        [phi_1(d), phi_2(d), ..., phi_K(d)]

    where

        phi_k(d)
        =
        exp(
            -gamma * (d - mu_k)^2
        )
    """

    def __init__(
        self,
        num_rbf: int = 32,
        min_distance: float = 0.0,
        cutoff: float = 5.0,
        trainable: bool = False,
    ):
        super().__init__()

        if num_rbf < 1:
            raise ValueError(
                "num_rbf must be >= 1"
            )

        if min_distance < 0.0:
            raise ValueError(
                "min_distance must be >= 0"
            )

        if cutoff <= min_distance:
            raise ValueError(
                "cutoff must be greater than min_distance"
            )

        self.num_rbf = num_rbf
        self.min_distance = min_distance
        self.cutoff = cutoff
        self.trainable = trainable

        # ----------------------------------------------------
        # 1. Uniformly spaced Gaussian centers
        # ----------------------------------------------------

        centers = torch.linspace(
            min_distance,
            cutoff,
            num_rbf,
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # 2. Gaussian width
        # ----------------------------------------------------

        if num_rbf == 1:

            spacing = (
                cutoff
                - min_distance
            )

        else:

            spacing = float(
                centers[1]
                - centers[0]
            )

        gamma = torch.tensor(
            1.0 / (spacing ** 2),
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # 3. Trainable or fixed parameters
        # ----------------------------------------------------

        if trainable:

            self.centers = nn.Parameter(
                centers
            )

            self.gamma = nn.Parameter(
                gamma
            )

        else:

            self.register_buffer(
                "centers",
                centers,
            )

            self.register_buffer(
                "gamma",
                gamma,
            )


    def forward(
        self,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        """Encode distances with Gaussian basis functions.

        Parameters
        ----------
        distance:
            [N_edge]
            or
            [N_edge, 1]

        Returns
        -------
        rbf:
            [N_edge, num_rbf]
        """

        if distance.ndim == 2:

            if distance.size(1) != 1:
                raise ValueError(
                    "2D distance tensor must have shape "
                    "[N_edge, 1]"
                )

            distance = distance.squeeze(
                -1
            )

        elif distance.ndim != 1:

            raise ValueError(
                "distance must have shape "
                "[N_edge] or [N_edge, 1]"
            )

        delta = (
            distance.unsqueeze(-1)
            - self.centers.view(
                1,
                -1,
            )
        )

        rbf = torch.exp(
            -self.gamma
            * delta.pow(2)
        )

        return rbf


# ============================================================
# Smooth cutoff envelope
# ============================================================

class CosineCutoff(nn.Module):
    """Smoothly decay geometric features to zero at the cutoff.

    For d < cutoff:

        c(d)
        =
        0.5 * (
            cos(pi * d / cutoff)
            + 1
        )

    For d >= cutoff:

        c(d) = 0

    This is useful when the same geometry encoder is later combined with a
    radius-based spatial graph.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
    ):
        super().__init__()

        if cutoff <= 0.0:
            raise ValueError(
                "cutoff must be > 0"
            )

        self.cutoff = cutoff


    def forward(
        self,
        distance: torch.Tensor,
    ) -> torch.Tensor:

        if distance.ndim != 1:
            raise ValueError(
                "distance must have shape [N_edge]"
            )

        inside = (
            distance
            < self.cutoff
        )

        scaled = (
            math.pi
            * distance
            / self.cutoff
        )

        envelope = (
            0.5
            * (
                torch.cos(
                    scaled
                )
                + 1.0
            )
        )

        envelope = torch.where(
            inside,
            envelope,
            torch.zeros_like(
                envelope
            ),
        )

        return envelope


# ============================================================
# Geometry Encoder
# ============================================================

class GeometryEncoder(nn.Module):
    """Encode 3D geometry into edge-level latent representations.

    Parameters
    ----------
    embed_dim:
        Final geometric latent dimension D.

    num_rbf:
        Number of radial basis functions.

    cutoff:
        Maximum geometric interaction range in Angstrom.

    use_cutoff_envelope:
        Multiply RBF features by a smooth cosine cutoff envelope.

    dropout:
        Dropout after the geometry projection.

    layer_norm:
        Apply LayerNorm to the final H_geo.

    trainable_rbf:
        Make RBF centers/width learnable.

    Inputs
    ------
    pos:
        Atomic positions.

        Shape:
            [N_atom, 3]

    edge_index:
        Directed graph connectivity.

        Shape:
            [2, N_edge]

        Convention:
            edge_index[0] = src
            edge_index[1] = dst

    Outputs
    -------
    By default:
        H_geo:
            [N_edge, embed_dim]

    If ``return_geometry=True``:
        dictionary containing

            H_geo:
                [N_edge, embed_dim]

            distance:
                [N_edge]

            displacement:
                [N_edge, 3]

            direction:
                [N_edge, 3]

            rbf:
                [N_edge, num_rbf]

            cutoff_envelope:
                [N_edge]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        use_cutoff_envelope: bool = True,
        dropout: float = 0.0,
        layer_norm: bool = True,
        trainable_rbf: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()

        if embed_dim < 1:
            raise ValueError(
                "embed_dim must be >= 1"
            )

        if num_rbf < 1:
            raise ValueError(
                "num_rbf must be >= 1"
            )

        if cutoff <= 0.0:
            raise ValueError(
                "cutoff must be > 0"
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be > 0"
            )

        if not (
            0.0
            <= dropout
            < 1.0
        ):
            raise ValueError(
                "dropout must satisfy "
                "0 <= dropout < 1"
            )

        self.embed_dim = embed_dim
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.use_cutoff_envelope = (
            use_cutoff_envelope
        )
        self.eps = eps

        # ----------------------------------------------------
        # 1. Distance -> RBF
        # ----------------------------------------------------

        self.rbf_encoder = GaussianRBF(
            num_rbf=num_rbf,
            min_distance=0.0,
            cutoff=cutoff,
            trainable=trainable_rbf,
        )

        # ----------------------------------------------------
        # 2. Smooth cutoff
        # ----------------------------------------------------

        self.cutoff_fn = CosineCutoff(
            cutoff=cutoff
        )

        # ----------------------------------------------------
        # 3. RBF -> geometry latent
        # ----------------------------------------------------

        self.geometry_mlp = nn.Sequential(
            nn.Linear(
                num_rbf,
                embed_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
        )

        # ----------------------------------------------------
        # 4. Output normalization
        # ----------------------------------------------------

        if layer_norm:

            self.norm: nn.Module = (
                nn.LayerNorm(
                    embed_dim
                )
            )

        else:

            self.norm = nn.Identity()

        self.dropout = nn.Dropout(
            dropout
        )


    # ========================================================
    # Geometry computation
    # ========================================================

    def compute_geometry(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute displacement, distance, and unit direction."""

        # ----------------------------------------------------
        # 1. Input validation
        # ----------------------------------------------------

        if (
            pos.ndim != 2
            or pos.size(1) != 3
        ):
            raise ValueError(
                "Expected pos shape "
                "[N_atom, 3], "
                f"got {tuple(pos.shape)}"
            )

        if (
            edge_index.ndim != 2
            or edge_index.size(0) != 2
        ):
            raise ValueError(
                "Expected edge_index shape "
                "[2, N_edge], "
                f"got {tuple(edge_index.shape)}"
            )

        if edge_index.dtype != torch.long:
            raise ValueError(
                "edge_index must have dtype torch.long"
            )

        num_atoms = pos.size(
            0
        )

        num_edges = edge_index.size(
            1
        )

        # ----------------------------------------------------
        # 2. Zero-edge graph
        # ----------------------------------------------------

        if num_edges == 0:

            return {
                "displacement": torch.empty(
                    (0, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                ),

                "distance": torch.empty(
                    (0,),
                    dtype=pos.dtype,
                    device=pos.device,
                ),

                "direction": torch.empty(
                    (0, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                ),
            }

        # ----------------------------------------------------
        # 3. Index validation
        # ----------------------------------------------------

        if torch.any(
            edge_index < 0
        ):
            raise ValueError(
                "edge_index contains "
                "negative atom indices"
            )

        if torch.any(
            edge_index >= num_atoms
        ):
            raise ValueError(
                "edge_index contains atom index "
                "outside pos"
            )

        # ----------------------------------------------------
        # 4. Gather source / destination positions
        # ----------------------------------------------------

        src = edge_index[
            0
        ]

        dst = edge_index[
            1
        ]

        pos_src = pos[
            src
        ]

        pos_dst = pos[
            dst
        ]

        # ----------------------------------------------------
        # 5. Relative geometry
        # ----------------------------------------------------

        # Vector pointing:
        #
        # src -> dst

        displacement = (
            pos_dst
            - pos_src
        )

        distance = torch.linalg.vector_norm(
            displacement,
            ord=2,
            dim=-1,
        )

        # ----------------------------------------------------
        # 6. Unit direction
        # ----------------------------------------------------

        safe_distance = distance.clamp_min(
            self.eps
        )

        direction = (
            displacement
            / safe_distance.unsqueeze(
                -1
            )
        )

        # For a pathological zero-distance edge,
        # explicitly return direction = 0.

        zero_distance = (
            distance
            <= self.eps
        )

        if torch.any(
            zero_distance
        ):

            direction = direction.clone()

            direction[
                zero_distance
            ] = 0.0

        return {
            "displacement": displacement,
            "distance": distance,
            "direction": direction,
        }


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        return_geometry: bool = False,
    ):
        """Encode edge geometry."""

        geometry = self.compute_geometry(
            pos=pos,
            edge_index=edge_index,
        )

        distance = geometry[
            "distance"
        ]

        # ----------------------------------------------------
        # 1. Handle zero-edge graph
        # ----------------------------------------------------

        if distance.numel() == 0:

            H_geo = torch.empty(
                (
                    0,
                    self.embed_dim,
                ),
                dtype=pos.dtype,
                device=pos.device,
            )

            rbf = torch.empty(
                (
                    0,
                    self.num_rbf,
                ),
                dtype=pos.dtype,
                device=pos.device,
            )

            cutoff_envelope = torch.empty(
                (0,),
                dtype=pos.dtype,
                device=pos.device,
            )

            if return_geometry:

                return {
                    "H_geo": H_geo,
                    "distance": distance,
                    "displacement": geometry[
                        "displacement"
                    ],
                    "direction": geometry[
                        "direction"
                    ],
                    "rbf": rbf,
                    "cutoff_envelope": (
                        cutoff_envelope
                    ),
                }

            return H_geo

        # ----------------------------------------------------
        # 2. Distance -> RBF
        # ----------------------------------------------------

        rbf = self.rbf_encoder(
            distance
        )

        # ----------------------------------------------------
        # 3. Smooth cutoff envelope
        # ----------------------------------------------------

        cutoff_envelope = self.cutoff_fn(
            distance
        )

        if self.use_cutoff_envelope:

            rbf_for_encoding = (
                rbf
                * cutoff_envelope.unsqueeze(
                    -1
                )
            )

        else:

            rbf_for_encoding = rbf

        # ----------------------------------------------------
        # 4. RBF -> latent geometry feature
        # ----------------------------------------------------

        H_geo = self.geometry_mlp(
            rbf_for_encoding
        )

        H_geo = self.norm(
            H_geo
        )

        H_geo = self.dropout(
            H_geo
        )

        # ----------------------------------------------------
        # 5. Return
        # ----------------------------------------------------

        if return_geometry:

            return {
                "H_geo": H_geo,
                "distance": distance,
                "displacement": geometry[
                    "displacement"
                ],
                "direction": geometry[
                    "direction"
                ],
                "rbf": rbf,
                "cutoff_envelope": (
                    cutoff_envelope
                ),
            }

        return H_geo


# ============================================================
# Standalone example
# ============================================================

def main() -> None:
    """Small geometry-encoder sanity check."""

    # --------------------------------------------------------
    # Example: simple 3-atom geometry
    # --------------------------------------------------------
    #
    # Atom 0:
    #     (0, 0, 0)
    #
    # Atom 1:
    #     (1, 0, 0)
    #
    # Atom 2:
    #     (1, 1, 0)
    #
    # Directed edges:
    #
    #     0 -> 1
    #     1 -> 0
    #     1 -> 2
    #     2 -> 1

    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )

    encoder = GeometryEncoder(
        embed_dim=128,
        num_rbf=32,
        cutoff=5.0,
        use_cutoff_envelope=True,
    )

    output = encoder(
        pos=pos,
        edge_index=edge_index,
        return_geometry=True,
    )

    print(
        "=" * 72
    )

    print(
        "Geometry Encoder"
    )

    print(
        "=" * 72
    )

    print(
        "pos shape:",
        tuple(
            pos.shape
        ),
    )

    print(
        "edge_index shape:",
        tuple(
            edge_index.shape
        ),
    )

    print(
        "\ndisplacement:"
    )

    print(
        output[
            "displacement"
        ]
    )

    print(
        "\ndistance:"
    )

    print(
        output[
            "distance"
        ]
    )

    print(
        "\ndirection:"
    )

    print(
        output[
            "direction"
        ]
    )

    print(
        "\nRBF shape:"
    )

    print(
        tuple(
            output[
                "rbf"
            ].shape
        )
    )

    print(
        "\nH_geo shape:"
    )

    print(
        tuple(
            output[
                "H_geo"
            ].shape
        )
    )

    print(
        "\nExpected:"
    )

    print(
        "distance shape = [N_edge]"
    )

    print(
        "direction shape = [N_edge, 3]"
    )

    print(
        "RBF shape = [N_edge, num_rbf]"
    )

    print(
        "H_geo shape = [N_edge, embed_dim]"
    )


if __name__ == "__main__":
    main()
