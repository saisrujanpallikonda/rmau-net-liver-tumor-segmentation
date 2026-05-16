# RMAU-Net: Liver & Tumor Segmentation in CT Images

> Modified implementation of RMAU-Net achieving **Tumor DSC 0.8936** — outperforming the original paper by **+17.3%**

## 📄 Based On
Linfeng Jiang et al., *"RMAU-Net: Residual Multi-Scale Attention U-Net For liver and tumor segmentation in CT images"*, Computers in Biology and Medicine, Vol. 158, 2023.

---

## 🔑 Key Modification — Local Channel Attention

| | Original Paper | Our Implementation |
|--|--|--|
| Method | `AdaptiveAvgPool2d(1)` — global 1×1 pooling | Window-based 8×8 local pooling |
| Problem | Large liver suppresses small tumor signal | Each region gets independent attention |
| Impact | Misses multi-lesion tumor cases | Tumor DSC +17.3% over paper |

**Why it matters:** When two spatially separated tumors exist in a CT slice, global pooling averages the entire feature map into one value — the dominant liver region drowns out the smaller tumor signal. Window-based local attention gives each 8×8 region its own independent channel weights, so both tumors are detected regardless of their spatial separation.

---

## 📊 Results on LiTS Dataset

| Metric | Ours | Paper | Δ |
|--------|------|-------|---|
| Liver DSC ↑ | **0.9198** | 0.9552 | -3.7% |
| Tumor DSC ↑ | **0.8936** | 0.7616 | **+17.3%** |
| Liver VOE ↓ | 0.1267 | 0.0792 | - |
| Tumor VOE ↓ | **0.1284** | 0.3709 | **-65.4%** |
| Liver RVD ≈0 | 0.0872 | -0.0042 | ≈0 |
| Tumor RVD ≈0 | **-0.0238** | 0.0118 | ≈0 |

---

## 🏗️ Architecture

```
Input (1×256×256)
    ↓
Encoder: ResSEBlock × 4 stages (32→64→128→256 channels)
    ↓ MaxPool 2×2 at each stage
Bottleneck: ResSEBlock (512ch) + MAB
    ↑ UpConv 2×2 + Skip Connections
Decoder: ResSEBlock + MAB × 4 stages (256→128→64→32 channels)
    ↓
1×1 Conv → Output (3×256×256)
Classes: 0=Background | 1=Liver | 2=Tumor
```

**Key components:**
- **ResSEBlock** — Residual connection + Squeeze-Excitation channel recalibration
- **LocalChannelAttention** — Window-based (8×8) local pooling *(our modification)*
- **SpatialAttention** — 7×7 conv on avg+max channel maps
- **ASPP** — Atrous Spatial Pyramid Pooling, dilation rates r=1,2,4,8
- **MAB** — Multi-scale Attention Block (ASPP + LocalChannelAttn + SpatialAttn)

---

## 📁 Project Structure

```
rmau-net-liver-tumor-segmentation/
├── src/
│   ├── dataset.py      # LiverDataset — context-aware label mapping
│   ├── model.py        # RMAUNet + all blocks
│   ├── loss.py         # DiceLoss, FocalLoss, HybridLoss
│   ├── metrics.py      # DSC, VOE, RVD computation
│   └── train.py        # train_one_epoch, evaluate
├── notebook/
│   └── RMAU_Net_LocalAttn.ipynb   # Full Colab training notebook
├── results/
│   ├── predictions.png
│   ├── training_curves.png
│   └── comparison_chart.png
├── main.py             # Training entry point
├── predict.py          # Inference script
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/rmau-net-liver-tumor-segmentation
cd rmau-net-liver-tumor-segmentation
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train from scratch
```bash
python main.py \
    --img_dir /path/to/lits/images \
    --mask_dir /path/to/lits/masks \
    --epochs 150 \
    --batch_size 16 \
    --save_path checkpoints/rmau_net.pth
```

### Resume training
```bash
python main.py \
    --img_dir /path/to/lits/images \
    --mask_dir /path/to/lits/masks \
    --resume checkpoints/rmau_net.pth \
    --epochs 50
```

### Run inference
```bash
python predict.py \
    --img_dir /path/to/images \
    --mask_dir /path/to/masks \
    --checkpoint checkpoints/rmau_net.pth \
    --save_dir results/predictions \
    --n_samples 10
```

### For Google Drive (Colab) — use num_workers=0
```bash
python main.py \
    --img_dir /content/drive/MyDrive/RMAU_Net/new_subset/images \
    --mask_dir /content/drive/MyDrive/RMAU_Net/new_subset/masks \
    --num_workers 0
```

---

## 🗂️ Dataset — LiTS

Download from: [LiTS Challenge](https://competitions.codalab.org/competitions/17094)

**Our label mapping (context-aware):**
```
If slice contains 127 (liver+tumor slice):
    127 → Class 1 (Liver)
    255 → Class 2 (Tumor)
Else (liver-only slice):
    255 → Class 1 (Liver)
```
> This fixes a dataset-specific encoding where 255 serves dual purpose across slice types.

**Preprocessing:**
- Resize: 512×512 → 256×256
- Normalize: pixel values to [-1, 1]
- Augmentation: horizontal/vertical flip, rotation ±30°

---

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | Adam, lr=1e-4 |
| Batch Size | 16 |
| Epochs | 150 |
| Loss | 0.5×Focal + 1.0×Dice |
| Class Weights | BG=0.1, Liver=1.0, Tumor=5.0 |
| Gradient Clip | max_norm=1.0 |
| Train/Val Split | 80/20, seed=42 |

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.5.0
```

---

## 📌 Citation

```bibtex
@article{jiang2023rmau,
  title={RMAU-Net: Residual Multi-Scale Attention U-Net For liver and tumor segmentation in CT images},
  author={Jiang, Linfeng and others},
  journal={Computers in Biology and Medicine},
  volume={158},
  year={2023}
}
```

---

## 📜 License
This project is for academic and research purposes only.
Dataset usage subject to [LiTS Challenge terms](https://competitions.codalab.org/competitions/17094).
