"""
train.py — Training and evaluation functions for RMAU-Net
Includes gradient clipping (max_norm=1.0) to prevent loss explosion.
"""

import time
import torch
import numpy as np
from metrics import batch_metrics


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Train model for one epoch.

    Args:
        model:     RMAUNet
        loader:    DataLoader (train)
        optimizer: torch.optim
        loss_fn:   HybridLoss
        device:    torch.device

    Returns:
        dict: {"loss": float, "liver_dsc": float, "tumor_dsc": float}
    """
    model.train()
    tl, ld, td = [], [], []

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss, _, _, _ = loss_fn(logits, masks)
        loss.backward()
        # Gradient clipping — prevents loss explosion with weighted Dice
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tl.append(loss.item())
        m = batch_metrics(logits.detach(), masks)
        ld.append(m[1]["DSC"])
        td.append(m[2]["DSC"])

    return {
        "loss":      np.mean(tl),
        "liver_dsc": np.mean(ld) if ld else 0.0,
        "tumor_dsc": np.mean(td) if td else 0.0,
    }


def evaluate(model, loader, loss_fn, device):
    """
    Evaluate model on validation set.

    Args:
        model:    RMAUNet
        loader:   DataLoader (val)
        loss_fn:  HybridLoss
        device:   torch.device

    Returns:
        dict: all metrics for liver and tumor (DSC, VOE, RVD) + val loss
    """
    model.eval()
    tl = []
    am = {cls: {"DSC": [], "VOE": [], "RVD": []} for cls in [1, 2]}

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss, _, _, _ = loss_fn(logits, masks)
            tl.append(loss.item())
            m = batch_metrics(logits, masks)
            for cls in [1, 2]:
                for k in ["DSC", "VOE", "RVD"]:
                    am[cls][k].append(m[cls][k])

    return {
        "loss":      np.mean(tl),
        "liver_dsc": np.mean(am[1]["DSC"]),
        "liver_voe": np.mean(am[1]["VOE"]),
        "liver_rvd": np.mean(am[1]["RVD"]),
        "tumor_dsc": np.mean(am[2]["DSC"]),
        "tumor_voe": np.mean(am[2]["VOE"]),
        "tumor_rvd": np.mean(am[2]["RVD"]),
    }


def train(model, train_loader, val_loader, optimizer, loss_fn,
          device, save_path, start_epoch=1, n_epochs=100,
          best_dsc=0.0, history=None):
    """
    Full training loop with checkpoint saving every epoch.

    Args:
        model, train_loader, val_loader, optimizer, loss_fn, device
        save_path:   str — path to save checkpoint (.pth)
        start_epoch: int — resume from this epoch
        n_epochs:    int — total epochs to run
        best_dsc:    float — best avg DSC so far (for tracking)
        history:     dict — existing history to append to

    Returns:
        history dict
    """
    if history is None:
        history = {
            "train_loss": [], "val_loss": [],
            "liver_dsc":  [], "tumor_dsc": [],
            "liver_voe":  [], "tumor_voe": [],
            "liver_rvd":  [], "tumor_rvd": [],
        }

    end_epoch = start_epoch + n_epochs - 1
    print(f"Training epochs {start_epoch} → {end_epoch}")
    print("=" * 65)

    for epoch in range(start_epoch, end_epoch + 1):
        t0      = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m   = evaluate(model, val_loader, loss_fn, device)
        avg_dsc = (val_m["liver_dsc"] + val_m["tumor_dsc"]) / 2

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["liver_dsc"].append(val_m["liver_dsc"])
        history["tumor_dsc"].append(val_m["tumor_dsc"])
        history["liver_voe"].append(val_m["liver_voe"])
        history["tumor_voe"].append(val_m["tumor_voe"])
        history["liver_rvd"].append(val_m["liver_rvd"])
        history["tumor_rvd"].append(val_m["tumor_rvd"])

        if avg_dsc > best_dsc:
            best_dsc  = avg_dsc
            save_flag = "💾 BEST"
        else:
            save_flag = ""

        # Save every epoch
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dsc":  best_dsc,
            "history":   history,
        }, save_path)

        print(f"Epoch [{epoch:03d}/{end_epoch}] | {time.time()-t0:.1f}s "
              f"| Train: {train_m['loss']:.4f} "
              f"| Val: {val_m['loss']:.4f} "
              f"| Liver DSC: {val_m['liver_dsc']:.4f} "
              f"| Tumor DSC: {val_m['tumor_dsc']:.4f} "
              f"| Avg: {avg_dsc:.4f} {save_flag}")

    return history
