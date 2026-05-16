"""
dataset.py — LiverDataset for LiTS CT slice segmentation
Labels: 0=Background, 1=Liver, 2=Tumor
Mask encoding: 127=Liver, 255=Tumor (context-aware mapping)
"""

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class LiverDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.augment   = augment

        imgs  = set(os.listdir(image_dir))
        masks = set(os.listdir(mask_dir))
        self.images = sorted(list(imgs & masks))
        print(f"  Valid pairs: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img  = cv2.imread(os.path.join(self.image_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir,  name), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Error loading: {name}")

        # Resize to 256x256
        img  = cv2.resize(img,  (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Context-aware label mapping:
        # If 127 present → liver=127, tumor=255
        # If 127 absent  → liver=255 (liver-only slice, no tumor)
        new_mask = np.zeros_like(mask, dtype=np.int64)
        if 127 in np.unique(mask):
            new_mask[mask == 127] = 1   # liver
            new_mask[mask == 255] = 2   # tumor
        else:
            new_mask[mask == 255] = 1   # liver only
        mask = new_mask

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                img  = cv2.flip(img,  1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.3:
                img  = cv2.flip(img,  0)
                mask = cv2.flip(mask, 0)
            if random.random() < 0.3:
                angle = random.uniform(-30, 30)
                M = cv2.getRotationMatrix2D((128, 128), angle, 1.0)
                img  = cv2.warpAffine(img,  M, (256, 256),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, M, (256, 256),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT_101)

        # Normalize to [-1, 1]
        img  = img.astype(np.float32) / 127.5 - 1.0
        img  = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).long()
        return img, mask
