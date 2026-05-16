"""
model.py — RMAU-Net with Local Channel Attention
Architecture: U-Net backbone with ResSEBlock encoder/decoder
and MAB (ASPP + LocalChannelAttention + SpatialAttention) at bottleneck
and all decoder stages.

Modification from paper:
    Global ChannelAttention (AdaptiveAvgPool2d) replaced with
    LocalChannelAttention (window-based 8x8 pooling) to improve
    multi-lesion tumor detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResSEBlock(nn.Module):
    """Residual Squeeze-and-Excitation Block."""
    def __init__(self, in_channels, out_channels, reduction_ratio=6):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True))
        rd = max(1, out_channels // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(out_channels, rd), nn.ReLU(inplace=True),
            nn.Linear(rd, out_channels), nn.Sigmoid())
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        u = self.conv_block(x)
        s = self.se(u).view(u.size(0), u.size(1), 1, 1)
        return self.residual(x) + u * s


class LocalChannelAttention(nn.Module):
    """
    Window-based Local Channel Attention.
    Replaces global AdaptiveAvgPool2d(1) with 8x8 window pooling
    so each spatial region gets independent attention weights.
    Fixes multi-lesion detection failure in global attention.
    """
    def __init__(self, in_channels, reduction_ratio=16, window_size=8):
        super().__init__()
        self.ws  = window_size
        rd = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, rd),
            nn.ReLU(inplace=True),
            nn.Linear(rd, in_channels))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # Fallback to global attention for small feature maps
        if H < self.ws or W < self.ws:
            avg = F.adaptive_avg_pool2d(x, 1).view(B, C)
            mx  = F.adaptive_max_pool2d(x, 1).view(B, C)
            w   = self.sig(self.mlp(avg) + self.mlp(mx)).view(B, C, 1, 1)
            return x * w

        # Partition into non-overlapping windows
        patches = x.unfold(2, self.ws, self.ws).unfold(3, self.ws, self.ws)
        nh, nw  = H // self.ws, W // self.ws
        patches = patches.contiguous().view(B, C, nh * nw, self.ws, self.ws)

        # Pool within each window
        avg = patches.mean(dim=(3, 4))          # B x C x (nh*nw)
        mx  = patches.amax(dim=(3, 4))          # B x C x (nh*nw)

        # MLP per window
        att = self.sig(
            self.mlp(avg.permute(0, 2, 1)) +
            self.mlp(mx.permute(0, 2, 1))
        )                                        # B x (nh*nw) x C
        att = att.permute(0, 2, 1)              # B x C x (nh*nw)
        att = att.view(B, C, nh, nw)

        # Upsample back to original size
        att = att.repeat_interleave(self.ws, dim=2).repeat_interleave(self.ws, dim=3)
        return x * att


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM-style)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg   = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x,  dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with dilation rates r=1,2,4,8."""
    def __init__(self, in_channels):
        super().__init__()
        bc = max(1, in_channels // 4)
        self.b1   = self._b(in_channels, bc, 1)
        self.b2   = self._b(in_channels, bc, 2)
        self.b3   = self._b(in_channels, bc, 4)
        self.b4   = self._b(in_channels, bc, 8)
        self.fuse = nn.Sequential(
            nn.Conv2d(bc * 4, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels), nn.LeakyReLU(inplace=True))

    def _b(self, ic, oc, d):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=d, dilation=d, bias=False),
            nn.BatchNorm2d(oc), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


class MAB(nn.Module):
    """Multi-scale Attention Block: ASPP + LocalChannelAttention + SpatialAttention."""
    def __init__(self, in_channels):
        super().__init__()
        self.aspp = ASPP(in_channels)
        self.ca   = LocalChannelAttention(in_channels)  # local window attention
        self.sa   = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(self.aspp(x)))


class RMAUNet(nn.Module):
    """
    RMAU-Net: Residual Multi-Scale Attention U-Net
    Modified with LocalChannelAttention (window-based 8x8)
    instead of global channel attention from the original paper.

    Input:  (B, 1, 256, 256)
    Output: (B, 3, 256, 256) — 3-class segmentation
    """
    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()
        self.enc1 = ResSEBlock(in_channels, 32)
        self.enc2 = ResSEBlock(32,  64)
        self.enc3 = ResSEBlock(64,  128)
        self.enc4 = ResSEBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck     = ResSEBlock(256, 512)
        self.bottleneck_mab = MAB(512)

        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ResSEBlock(512, 256); self.dec4_mab = MAB(256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResSEBlock(256, 128); self.dec3_mab = MAB(128)

        self.up2  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec2 = ResSEBlock(128, 64);  self.dec2_mab = MAB(64)

        self.up1  = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.dec1 = ResSEBlock(64,  32);  self.dec1_mab = MAB(32)

        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b  = self.bottleneck_mab(self.bottleneck(self.pool(s4)))
        d4 = self.dec4_mab(self.dec4(torch.cat([self.up4(b),  s4], dim=1)))
        d3 = self.dec3_mab(self.dec3(torch.cat([self.up3(d4), s3], dim=1)))
        d2 = self.dec2_mab(self.dec2(torch.cat([self.up2(d3), s2], dim=1)))
        d1 = self.dec1_mab(self.dec1(torch.cat([self.up1(d2), s1], dim=1)))
        return self.head(d1)
