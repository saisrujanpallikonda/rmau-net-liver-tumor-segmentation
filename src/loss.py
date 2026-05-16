"""
loss.py — Hybrid Loss for liver/tumor segmentation
Exact paper formulation: Ltotal = 0.5 * FocalLoss + 1.0 * DiceLoss
Class weights: BG=0.1, Liver=1.0, Tumor=5.0 (to handle extreme class imbalance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Weighted Dice Loss.
    Formula from paper (Eq. 9):
        Ldice = 1 - (2 * sum(p*y)) / (sum(p^2) + sum(y^2))
    Class weights: BG=0.1, Liver=1.0, Tumor=5.0
    """
    def __init__(self, n_classes=3, smooth=1.0):
        super().__init__()
        self.nc = n_classes
        self.sm = smooth

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        t = F.one_hot(targets, self.nc).permute(0, 3, 1, 2).float()
        p = p.view(p.size(0), self.nc, -1)
        t = t.view(t.size(0), self.nc, -1)
        dice    = (2*(p*t).sum(2)+self.sm) / ((p**2).sum(2)+(t**2).sum(2)+self.sm)
        weights = torch.tensor([0.1, 1.0, 5.0], device=logits.device)
        return 1.0 - (dice * weights.view(1, -1)).mean()


class FocalLoss(nn.Module):
    """
    Class-wise Focal Loss.
    Formula from paper (Eq. 7-8):
        LF = alpha_t * (1 - pt)^gamma * CE
    Class-wise alpha: BG=0.1, Liver=1.0, Tumor=5.0
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, reduction='none')
        pt    = torch.exp(-ce)
        alpha = torch.tensor([0.1, 1.0, 5.0], device=logits.device)
        at    = alpha[targets]
        return (at * (1 - pt) ** self.gamma * ce).mean()


class WeightedCELoss(nn.Module):
    """Weighted Cross-Entropy Loss. BG=0.05, Liver=1.0, Tumor=5.0"""
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        weights = torch.tensor([0.05, 1.0, 5.0], device=logits.device)
        return F.cross_entropy(logits, targets, weight=weights)


class HybridLoss(nn.Module):
    """
    Hybrid Loss — exact paper formulation (Eq. 10):
        Ltotal = 0.5 * FocalLoss + 1.0 * DiceLoss
    Returns: (total_loss, focal_loss, dice_loss, dice_loss)
    """
    def __init__(self, n_classes=3):
        super().__init__()
        self.focal = FocalLoss()
        self.dice  = DiceLoss(n_classes)

    def forward(self, logits, targets):
        lf = self.focal(logits, targets)
        ld = self.dice(logits, targets)
        return 0.5 * lf + 1.0 * ld, lf, ld, ld
