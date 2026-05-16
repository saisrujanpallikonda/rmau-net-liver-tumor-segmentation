"""
metrics.py — Evaluation metrics for liver/tumor segmentation
Metrics: DSC (Dice Similarity Coefficient), VOE (Volume Overlap Error),
         RVD (Relative Volume Difference)
"""

import numpy as np
import torch


def compute_metrics(pred, true, n_classes=3):
    """
    Compute DSC, VOE, RVD for a single prediction/ground truth pair.

    Args:
        pred: torch.Tensor or np.ndarray (H, W) — predicted class labels
        true: torch.Tensor or np.ndarray (H, W) — ground truth class labels
        n_classes: int — number of classes (default 3: BG, Liver, Tumor)

    Returns:
        dict: {class_id: {"DSC": float, "VOE": float, "RVD": float}}
        Note: class 0 (background) is skipped
    """
    if torch.is_tensor(pred): pred = pred.detach().cpu().numpy()
    if torch.is_tensor(true): true = true.detach().cpu().numpy()

    m = {}
    for cls in range(1, n_classes):
        p = (pred == cls).astype(np.float32)
        t = (true == cls).astype(np.float32)

        intersection = np.sum(p * t)
        sum_p = np.sum(p)
        sum_t = np.sum(t)
        union = np.sum(np.logical_or(p, t))

        # Dice Similarity Coefficient
        dsc = (2 * intersection + 1e-6) / (sum_p + sum_t + 1e-6)

        # Volume Overlap Error
        voe = 0.0 if union < 1 else 1 - (intersection + 1e-6) / (union + 1e-6)

        # Relative Volume Difference
        rvd = 0.0 if sum_t < 1 else float(
            np.clip((sum_p - sum_t) / sum_t, -1.0, 10.0))

        m[cls] = {"DSC": dsc, "VOE": voe, "RVD": rvd}
    return m


def batch_metrics(logits, targets, n_classes=3):
    """
    Compute average metrics over a batch.

    Args:
        logits: torch.Tensor (B, C, H, W) — model output
        targets: torch.Tensor (B, H, W)   — ground truth
        n_classes: int

    Returns:
        dict: {class_id: {"DSC": float, "VOE": float, "RVD": float}}
    """
    preds = torch.argmax(logits, dim=1)
    am    = {cls: {"DSC": [], "VOE": [], "RVD": []}
             for cls in range(1, n_classes)}

    for i in range(preds.shape[0]):
        m = compute_metrics(preds[i], targets[i], n_classes)
        for cls in range(1, n_classes):
            for k in ["DSC", "VOE", "RVD"]:
                am[cls][k].append(m[cls][k])

    return {cls: {k: np.mean(v) if v else 0.0
                  for k, v in am[cls].items()}
            for cls in range(1, n_classes)}
