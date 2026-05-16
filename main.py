"""
main.py — RMAU-Net Training Script
Residual Multi-Scale Attention U-Net for Liver & Tumor Segmentation
Dataset: LiTS (Liver Tumor Segmentation Challenge)

Usage:
    python main.py --img_dir /path/to/images --mask_dir /path/to/masks
    python main.py --img_dir /path/to/images --mask_dir /path/to/masks --resume checkpoint.pth
    python main.py --img_dir /path/to/images --mask_dir /path/to/masks --epochs 150
"""

import os
import argparse
import time
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from src.dataset import LiverDataset
from src.model   import RMAUNet
from src.loss    import HybridLoss
from src.metrics import batch_metrics


# ═══════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(description='RMAU-Net Training')
    parser.add_argument('--img_dir',    type=str, required=True,  help='Path to images directory')
    parser.add_argument('--mask_dir',   type=str, required=True,  help='Path to masks directory')
    parser.add_argument('--save_path',  type=str, default='checkpoints/rmau_net.pth', help='Checkpoint save path')
    parser.add_argument('--epochs',     type=int, default=150,    help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,     help='Batch size')
    parser.add_argument('--lr',         type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split',  type=float, default=0.2,  help='Validation split ratio')
    parser.add_argument('--seed',       type=int, default=42,     help='Random seed for split')
    parser.add_argument('--num_workers',type=int, default=4,      help='DataLoader workers (0 for Drive)')
    parser.add_argument('--resume',     type=str, default=None,   help='Path to checkpoint to resume from')
    parser.add_argument('--n_classes',  type=int, default=3,      help='Number of output classes')
    return parser.parse_args()


# ═══════════════════════════════════════════════
# TRAIN ONE EPOCH
# ═══════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    tl, ld, td = [], [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss, _, _, _ = loss_fn(logits, masks)
        loss.backward()
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


# ═══════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════
def evaluate(model, loader, loss_fn, device):
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


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Checkpoint dir ───────────────────────
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)

    # ── Datasets ─────────────────────────────
    print("\nLoading dataset...")
    train_base = LiverDataset(args.img_dir, args.mask_dir, augment=True)
    val_base   = LiverDataset(args.img_dir, args.mask_dir, augment=False)

    g          = torch.Generator().manual_seed(args.seed)
    perm       = torch.randperm(len(train_base), generator=g).tolist()
    train_size = int((1 - args.val_split) * len(train_base))
    train_idx  = perm[:train_size]
    val_idx    = perm[train_size:]

    train_ds = Subset(train_base, train_idx)
    val_ds   = Subset(val_base,   val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False)

    print(f"Train : {len(train_ds):,} samples")
    print(f"Val   : {len(val_ds):,} samples")

    # ── Model ────────────────────────────────
    model   = RMAUNet(in_channels=1, n_classes=args.n_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn   = HybridLoss(n_classes=args.n_classes).to(device)
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume ───────────────────────────────
    start_epoch = 1
    best_dsc    = 0.0
    history = {
        "train_loss": [], "val_loss": [],
        "liver_dsc":  [], "tumor_dsc": [],
        "liver_voe":  [], "tumor_voe": [],
        "liver_rvd":  [], "tumor_rvd": [],
    }

    resume_path = args.resume or args.save_path
    if args.resume and os.path.exists(resume_path):
        print(f"\nResuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_dsc    = ckpt.get("best_dsc", 0.0)
        history     = ckpt.get("history", history)
        print(f"Resumed from epoch {start_epoch-1} | Best DSC: {best_dsc:.4f}")

    # ── Training loop ─────────────────────────
    end_epoch = start_epoch + args.epochs - 1
    print(f"\nTraining epochs {start_epoch} → {end_epoch}")
    print("=" * 70)

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

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dsc":  best_dsc,
            "history":   history,
        }, args.save_path)

        print(f"Epoch [{epoch:03d}/{end_epoch}] | {time.time()-t0:.1f}s "
              f"| Train: {train_m['loss']:.4f} "
              f"| Val: {val_m['loss']:.4f} "
              f"| Liver DSC: {val_m['liver_dsc']:.4f} "
              f"| Tumor DSC: {val_m['tumor_dsc']:.4f} "
              f"| Avg: {avg_dsc:.4f} {save_flag}")

    # ── Final results ─────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS")
    print(f"{'='*55}")
    print(f"  {'Metric':<12} {'Liver':>10} {'Tumor':>10}")
    print(f"  {'-'*35}")
    print(f"  {'DSC':<12} {history['liver_dsc'][-1]:>10.4f} {history['tumor_dsc'][-1]:>10.4f}")
    print(f"  {'VOE':<12} {history['liver_voe'][-1]:>10.4f} {history['tumor_voe'][-1]:>10.4f}")
    print(f"  {'RVD':<12} {history['liver_rvd'][-1]:>10.4f} {history['tumor_rvd'][-1]:>10.4f}")
    print(f"{'='*55}")
    print(f"\nCheckpoint saved to: {args.save_path}")


if __name__ == "__main__":
    main()
