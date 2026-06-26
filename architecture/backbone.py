"""
Xception Entry Flow Feature Extractor for ISTVT.

Uses timm's Xception implementation, keeping only the entry flow
(stem + block1 + block2 + block3) as described in the paper (Section III-A).

Input:  (N, 3, H, W)  -- a batch of RGB face crops
Output: (N, 728, H/16, W/16)  -- texture feature maps

Examples:
  300x300 input -> 728 x 19 x 19  (paper's exact setting)
  128x128 input -> 728 x  8 x  8  (our scaled-down setting)
"""

import torch
import torch.nn as nn
import timm


class XceptionEntryFlow(nn.Module):
    out_channels = 728  # fixed by Xception architecture

    def __init__(self, pretrained=True):
        super().__init__()
        full_model = timm.create_model("xception", pretrained=pretrained)

        # keep only entry flow, discard everything else
        self.conv1  = full_model.conv1
        self.bn1    = full_model.bn1
        self.act1   = full_model.act1
        self.conv2  = full_model.conv2
        self.bn2    = full_model.bn2
        self.act2   = full_model.act2
        self.block1 = full_model.block1   # 64  -> 128 ch
        self.block2 = full_model.block2   # 128 -> 256 ch
        self.block3 = full_model.block3   # 256 -> 728 ch

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        B, T, C, H, W = x.shape
    # Merge batch and time dimensions
        x = x.view(B * T, C, H, W)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        _, C, Hf, Wf = x.shape

    # Restore batch and time dimensions
        x = x.view(B, T, C, Hf, Wf)

        return x


if __name__ == "__main__":
    model = XceptionEntryFlow(pretrained=True)
    model.eval()

    # test at paper resolution
    x = torch.randn(2, 6, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}")

    # test at toy resolution
    x = torch.randn(2, 6, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"Project resolution    : {tuple(x.shape)} -> {tuple(y.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters        : {n_params:,}")
