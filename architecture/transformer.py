import torch
import torch.nn as nn

from .attention import (
    TemporalAttention,
    SpatialAttention,
)

class MLP(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.net = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                embed_dim
            ),

            nn.Dropout(dropout)

        )

    def forward(self,x):

        return self.net(x)

class ISTVTBlock(nn.Module):
    """
    One ISTVT Transformer Block.

    Pipeline
    --------
    Input
        ↓
    LayerNorm
        ↓
    Temporal Attention
        ↓
    Residual

        ↓
    LayerNorm
        ↓
    Spatial Attention
        ↓
    Residual

        ↓
    LayerNorm
        ↓
    MLP
        ↓
    Residual

        ↓
    Output
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        # ----------------------------
        # LayerNorm before Temporal Attention
        # ----------------------------

        self.norm1 = nn.LayerNorm(embed_dim)

        self.temporal = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ----------------------------
        # LayerNorm before Spatial Attention
        # ----------------------------

        self.norm2 = nn.LayerNorm(embed_dim)

        self.spatial = SpatialAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ----------------------------
        # LayerNorm before MLP
        # ----------------------------

        self.norm3 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    def forward(self, x):

        # ----------------------------
        # Temporal Attention
        # ----------------------------

        x = x + self.temporal(
            self.norm1(x)
        )

        # ----------------------------
        # Spatial Attention
        # ----------------------------

        x = x + self.spatial(
            self.norm2(x)
        )

        # ----------------------------
        # Feed Forward Network
        # ----------------------------

        x = x + self.mlp(
            self.norm3(x)
        )

        return x

if __name__ == "__main__":

    model = ISTVTBlock()

    x = torch.randn(
        2,
        6,
        65,
        512
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)