import torch
import torch.nn as nn
from .backbone import XceptionEntryFlow
from .tokenizer import Tokenizer
from .transformer import ISTVTBlock

class ISTVT(nn.Module):
    """
    Interpretable Spatial-Temporal Video Transformer (ISTVT)

    Pipeline
    --------
    Input
        (B,6,3,128,128)

            ↓

    Xception Entry Flow

            ↓

        (B,6,728,8,8)

            ↓

        Tokenizer

            ↓

        (B,6,65,512)

            ↓

        Transformer Stack

            ↓

        Final LayerNorm

            ↓

        CLS Token

            ↓

    Classification Head

            ↓

        Fake / Real
    """

    def __init__(
        self,
        num_blocks=6,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()

        # -------------------------------------------------
        # Backbone
        # -------------------------------------------------

        self.backbone = XceptionEntryFlow(
            pretrained=True
        )

        # -------------------------------------------------
        # Tokenizer
        # -------------------------------------------------

        self.tokenizer = Tokenizer(
            in_channels=728,
            embed_dim=embed_dim,
            feature_size=(8, 8),
            seq_len=6,
        )

        # -------------------------------------------------
        # Transformer Blocks
        # -------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                ISTVTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # -------------------------------------------------
        # Final LayerNorm
        # -------------------------------------------------

        self.norm = nn.LayerNorm(embed_dim)

        # -------------------------------------------------
        # Classification Head
        # -------------------------------------------------

        self.head = nn.Sequential(

        nn.Linear(
            embed_dim * 6,
            512,
        ),

        nn.GELU(),

        nn.Dropout(dropout),

        nn.Linear(
            512,
            256,
        ),

        nn.GELU(),

        nn.Dropout(dropout),

        nn.Linear(
            256,
            num_classes,
        ) 

        )

    
    def forward(self, x):
   

    # ------------------------------------------
    # Xception Backbone
    # ------------------------------------------

        x = self.backbone(x)

    # (B,6,728,8,8)

    # ------------------------------------------
    # Tokenizer
    # ------------------------------------------

        x = self.tokenizer(x)

    # (B,6,65,512)

    # ------------------------------------------
    # Transformer Stack
    # ------------------------------------------

        for block in self.blocks:
            x = block(x)

    # ------------------------------------------
    # Final LayerNorm
    # ------------------------------------------

        x = self.norm(x)

    # ------------------------------------------
    # CLS Tokens
    # ------------------------------------------

        cls = x[:, :, 0, :]

    # (B,6,512)

    # ------------------------------------------
    # Flatten CLS Tokens
    # ------------------------------------------

        cls = cls.reshape(
            cls.size(0),
            -1
        )

    # (B,3072)

    # ------------------------------------------
    # Classification
    # ------------------------------------------

        logits = self.head(cls)

        return logits

if __name__ == "__main__":

    model = ISTVT()

    x = torch.randn(
        2,
        6,
        3,
        128,
        128
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)