"""
predict.py — Inference script for RMAU-Net
Runs segmentation on a folder of CT images and saves predictions.

Usage:
    python predict.py --img_dir /path/to/images --checkpoint checkpoints/rmau_net.pth
    python predict.py --img_dir /path/to/images --checkpoint checkpoints/rmau_net.pth --save_dir results/
"""

import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model import RMAUNet


def get_args():
    parser = argparse.ArgumentParser(description='RMAU-Net Inference')
    parser.add_argument('--img_dir',    type=str, required=True, help='Path to CT images')
    parser.add_argument('--mask_dir',   type=str, default=None,  help='Path to ground truth masks (optional)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_dir',   type=str, default='results/predictions', help='Directory to save outputs')
    parser.add_argument('--n_samples',  type=int, default=8,     help='Number of samples to visualize')
    parser.add_argument('--n_classes',  type=int, default=3,     help='Number of classes')
    return parser.parse_args()


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # 1x1x256x256


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    new_mask = np.zeros_like(mask, dtype=np.int64)
    if 127 in np.unique(mask):
        new_mask[mask == 127] = 1
        new_mask[mask == 255] = 2
    else:
        new_mask[mask == 255] = 1
    return new_mask


def predict_single(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred


def visualize(img, gt, pred, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    ct = img.squeeze().numpy()

    # CT
    axes[0].imshow(ct, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title('CT Slice'); axes[0].axis('off')

    # Ground Truth
    axes[1].imshow(ct, cmap='gray', vmin=-1, vmax=1)
    gt_col = np.zeros((*gt.shape, 4))
    gt_col[gt==1] = [0.0, 1.0, 0.0, 0.6]
    gt_col[gt==2] = [1.0, 0.0, 0.0, 0.8]
    axes[1].imshow(gt_col)
    axes[1].set_title(f'GT  L={np.sum(gt==1):,} T={np.sum(gt==2):,}')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(ct, cmap='gray', vmin=-1, vmax=1)
    pr_col = np.zeros((*pred.shape, 4))
    pr_col[pred==1] = [0.0, 1.0, 0.0, 0.6]
    pr_col[pred==2] = [1.0, 0.0, 0.0, 0.8]
    axes[2].imshow(pr_col)
    axes[2].set_title(f'Pred L={np.sum(pred==1):,} T={np.sum(pred==2):,}')
    axes[2].axis('off')

    # Overlay
    axes[3].imshow(ct, cmap='gray', vmin=-1, vmax=1)
    ov = np.zeros((*gt.shape, 4))
    ov[(gt==1)&(pred==1)] = [0.0, 1.0, 0.0, 0.5]
    ov[(gt==1)&(pred!=1)] = [0.0, 0.0, 1.0, 0.6]
    ov[(gt!=1)&(pred==1)] = [1.0, 0.0, 0.0, 0.5]
    ov[(gt==2)&(pred==2)] = [1.0, 1.0, 0.0, 0.8]
    ov[(gt==2)&(pred!=2)] = [1.0, 0.5, 0.0, 0.8]
    axes[3].imshow(ov)
    axes[3].set_title('Overlay'); axes[3].axis('off')

    patches = [
        mpatches.Patch(color='green',  label='Liver TP'),
        mpatches.Patch(color='blue',   label='Liver FN'),
        mpatches.Patch(color='red',    label='Liver FP'),
        mpatches.Patch(color='yellow', label='Tumor TP'),
        mpatches.Patch(color='orange', label='Tumor FN'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = RMAUNet(in_channels=1, n_classes=args.n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch {ckpt['epoch']} | Best DSC: {ckpt['best_dsc']:.4f}")

    # Get image files
    img_files = sorted(os.listdir(args.img_dir))[:args.n_samples]
    print(f"\nRunning inference on {len(img_files)} images...")

    for fname in img_files:
        img_path  = os.path.join(args.img_dir, fname)
        img_tensor = load_image(img_path)
        pred       = predict_single(model, img_tensor, device)

        if args.mask_dir and os.path.exists(os.path.join(args.mask_dir, fname)):
            gt        = load_mask(os.path.join(args.mask_dir, fname))
            save_path = os.path.join(args.save_dir, fname.replace('.png', '_pred.png'))
            visualize(img_tensor.squeeze(0), gt, pred, save_path)
        else:
            # No GT — just save prediction mask
            pred_colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
            pred_colored[pred==1] = [0, 200, 0]
            pred_colored[pred==2] = [200, 0, 0]
            save_path = os.path.join(args.save_dir, fname.replace('.png', '_pred.png'))
            cv2.imwrite(save_path, pred_colored)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
