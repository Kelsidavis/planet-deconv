"""Small U-Net for single-frame planetary restoration.

3-level encoder/decoder, ~600k params at base=32. Single-channel in/out.
Residual on the input — the network predicts a correction, not the full image,
which is much easier when the input is already close to the target.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GELU(),
    )


class TinyUNet(nn.Module):
    """Multi-frame-capable U-Net.

    Input is (B, in_ch, H, W) where in_ch is the burst length (odd, >=1).
    Output is (B, 1, H, W) — the residual is added to the *center* input
    frame, not all of them, so the model produces "center frame + correction
    informed by the temporal neighborhood".
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        if in_ch % 2 == 0:
            raise ValueError(f"in_ch must be odd (need a center frame), got {in_ch}")
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.in_ch = in_ch
        self.center = in_ch // 2
        self.enc1 = _conv_block(in_ch, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        self.bott = _conv_block(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _conv_block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _conv_block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _conv_block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, out_ch, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        # Residual on the *center* frame. No clamp during training —
        # saturating at [0,1] kills gradients on a poor init.
        center = x[:, self.center : self.center + 1]
        return center + self.head(d1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).clamp(0.0, 1.0)
