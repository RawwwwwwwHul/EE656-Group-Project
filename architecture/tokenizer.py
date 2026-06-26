"""
tokenizer.py
------------

Tokenizer module for ISTVT.

Implements the tokenization stage described in the paper.

Pipeline
--------
Input
    (B, T, 728, H, W)

↓

Flatten spatial dimensions

↓

Linear Projection
728 -> 512

↓

Prepend CLS Token

↓

Add Spatial Positional Embeddings

↓

Add Temporal Positional Embeddings

↓

Output
    (B, T, N+1, 512)

where

B = batch size
T = sequence length (6)
N = number of spatial patches (H×W)
"""

import torch
import torch.nn as nn


class Tokenizer(nn.Module):

    def __init__(
        self,
        in_channels=728,
        embed_dim=512,
        feature_size=(8, 8),
        seq_len=6,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len

        H, W = feature_size
        self.num_patches = H * W

        # --------------------------------------------------
        # Linear Projection
        # --------------------------------------------------

        self.projection = nn.Linear(
            in_channels,
            embed_dim
        )

        # --------------------------------------------------
        # CLS Token
        # --------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                1,
                embed_dim
            )
        )

        # --------------------------------------------------
        # Spatial Positional Embedding
        #
        # Shape:
        # (1,1,N+1,D)
        #
        # Shared across all frames
        # --------------------------------------------------

        self.spatial_embedding = nn.Parameter(
            torch.randn(
                1,
                1,
                self.num_patches + 1,
                embed_dim
            ) * 0.02
        )

        # --------------------------------------------------
        # Temporal Positional Embedding
        #
        # Shape:
        # (1,T,1,D)
        #
        # Shared across all patches
        # --------------------------------------------------

        self.temporal_embedding = nn.Parameter(
            torch.randn(
                1,
                seq_len,
                1,
                embed_dim
            ) * 0.02
        )

        self._initialize_weights()

    # ------------------------------------------------------

    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.spatial_embedding,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.temporal_embedding,
            std=0.02
        )

        nn.init.xavier_uniform_(
            self.projection.weight
        )

        nn.init.zeros_(
            self.projection.bias
        )

    # ------------------------------------------------------

    def forward(self, x):
        """
        Parameters
        ----------
        x

        Shape:
            (B,T,728,H,W)

        Returns
        -------
        Shape:
            (B,T,N+1,512)
        """

        B, T, C, H, W = x.shape

        # --------------------------------------------
        # Flatten feature maps into spatial tokens
        # --------------------------------------------

        x = x.permute(
            0,
            1,
            3,
            4,
            2
        )

        x = x.reshape(
            B,
            T,
            H * W,
            C
        )

        # --------------------------------------------
        # Linear Projection
        # --------------------------------------------

        x = self.projection(x)

        # --------------------------------------------
        # CLS Token
        # --------------------------------------------

        cls = self.cls_token.expand(
            B,
            T,
            -1,
            -1
        )

        x = torch.cat(
            [cls, x],
            dim=2
        )

        # --------------------------------------------
        # Spatial Position Encoding
        # --------------------------------------------

        x = x + self.spatial_embedding

        # --------------------------------------------
        # Temporal Position Encoding
        # --------------------------------------------

        x = x + self.temporal_embedding

        return x


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    tokenizer = Tokenizer()

    dummy = torch.randn(
        2,
        6,
        728,
        8,
        8
    )

    out = tokenizer(dummy)

    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)

    print()

    print("Embedding dimension :", tokenizer.embed_dim)
    print("Number of patches   :", tokenizer.num_patches)
    print("Sequence length     :", tokenizer.seq_len)