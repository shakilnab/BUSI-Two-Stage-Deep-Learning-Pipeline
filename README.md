# BUSI Breast Ultrasound Analysis — Two-Stage Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BUSI-green)](#dataset)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#license)

End-to-end deep learning system for **breast ultrasound image analysis** using the public [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset-images). The project first classifies an image as **benign / malignant / normal**, and then — only for abnormal cases — segments the tumour region. Three classification backbones and three pretrained segmentation backbones (plus a from-scratch baseline) are compared individually and as ensembles.

> Final test results: **Classification ensemble accuracy 86.32 %** and **UNet++ Dice 0.7933** on a held-out 15 % stratified test split.

---

## Table of Contents

- [Project Highlights](#project-highlights)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Highlights

| | |
|---|---|
| **Task** | Image classification + semantic segmentation |
| **Classes** | 3 (benign, malignant, normal) |
| **Classifier backbones** | EfficientNet-B4, Swin-Tiny, MobileViTv2-100 |
| **Segmenter backbones** | Attention U-Net, U-Net++, TransUNet (all with ResNet34 encoder pretrained on ImageNet) |
| **Baseline segmenter** | Attention U-Net trained from scratch (no pretrained encoder) |
| **Best classification** | Weighted-soft-voting **Ensemble** — Accuracy **0.8632** |
| **Best segmentation** | **U-Net++** — Dice **0.7933**, IoU **0.7035** |
| **Training strategy** | 3-phase fine-tuning (freeze → partial unfreeze → full), AdamW + Cosine, weighted CE / BCE+Dice combo |
| **Inference strategy** | Classifier decides whether segmentation is required (`normal → skip`) |
| **Statistical rigour** | Bootstrap 95 % CIs on all metrics (test set, n = 117) |

---

## Pipeline Architecture

```
                    ┌──────────────────────────────────────┐
                    │   Breast Ultrasound Image (256×256)  │
                    └────────────────┬─────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Classification     │  EfficientNet-B4
                          │  (3-class softmax)  │  Swin-Tiny
                          │                     │  MobileViTv2-100
                          └──────────┬──────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
            normal                benign            malignant
                │                    │                    │
                ▼                    └────────┬───────────┘
            (skip —                 ┌────────▼──────────┐
             no lesion)             │   Segmentation    │  Attention U-Net
                                    │   binary mask     │  U-Net++
                                    │                   │  TransUNet
                                    └────────┬──────────┘
                                             │
                                             ▼
                                  Pixel-wise tumour mask
```

**Key design choice:** because normal images contain no tumour, segmentation is only triggered when the classifier predicts an abnormal class. This avoids false-positive masks on healthy tissue.

---

## Dataset

We use the **BUSI — Breast Ultrasound Images** dataset (Al-Dhabyani *et al.*, 2020).

| Split | Count | benign | malignant | normal |
|---|---:|---:|---:|---:|
| Train (after offline augmentation) | 546 → expanded | 306 | 147 | 93 |
| Validation | 117 | 65 | 32 | 20 |
| Test (held-out) | 117 | 66 | 31 | 20 |
| **Original dataset** | **780** | **437** | **210** | **133** |

- **Original images:** 780 PNGs across 3 class folders.
- **Pre-processing:** all images resized to **256 × 256**; multiple mask files per image merged with **logical OR** into a single binary mask.
- **Data-quality checks:** 0 corrupt images found, mask coverage analysed, duplicate image set identified.
- **Augmentation:** offline augmentation (rotations, flips, brightness/contrast, etc.) on the training set to mitigate class imbalance and small-dataset overfitting.
- **Class balancing:** inverse-frequency class weights applied to cross-entropy.
- **Split:** stratified 70 / 15 / 15 with fixed seed (`42`).

> The BUSI dataset is **not included** in this repository. Download it from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset-images) and place it under `dataset/original/`.

---

## Methodology

### Stage 1 — Classification

Three ImageNet-pretrained backbones are wrapped with a 3-class classifier head:

| Model | Type | Params (approx.) |
|---|---|---:|
| **EfficientNet-B4** | CNN | ~19 M |
| **Swin Transformer Tiny** | Hierarchical transformer | ~28 M |
| **MobileViTv2-100** | Hybrid CNN + transformer | ~5 M |

**Training schedule (3 phases per model):**

| Phase | Epochs | Trainable | Learning rate |
|---|---:|---|---:|
| 1 — Freeze backbone | 10 | head only | 1 × 10⁻³ |
| 2 — Partial unfreeze | 10 | last blocks + head | 1 × 10⁻⁴ |
| 3 — Full fine-tune | 30 | all layers | 1 × 10⁻⁵ |

- **Loss:** weighted `CrossEntropyLoss` (inverse-frequency weights).
- **Optimiser:** AdamW (`weight_decay=0.01`), cosine annealing.
- **Augmentation (on-the-fly):** flips, rotations, color jitter.
- **Early stopping:** patience 10 on validation accuracy.

**Ensemble:** weighted soft-voting — class probabilities from each of the three models are averaged and `argmax`-ed.

### Stage 2 — Segmentation

Four binary-mask models are compared (input 256 × 256 × 3, output 256 × 256 × 1):

| Model | Backbone | Pretrained? |
|---|---|---|
| Attention U-Net | ResNet34 | ✅ ImageNet |
| U-Net++ (Nested U-Net) | ResNet34 | ✅ ImageNet |
| TransUNet | ResNet34 + ViT | ✅ ImageNet |
| Attention U-Net *(baseline)* | ResNet34 | ❌ random init |

**Training schedule (pretrained models, 3 phases):**

| Phase | Epochs | LR |
|---|---:|---:|
| 1 — Freeze encoder | 5 | 1 × 10⁻³ |
| 2 — Partial unfreeze | 5 | 1 × 10⁻⁴ |
| 3 — Full fine-tune | 40 | 1 × 10⁻⁵ |

- **Loss:** `0.5 · BCE + 0.5 · Dice`.
- **Optimiser:** AdamW (`weight_decay=0.01`), cosine annealing.
- **Mask post-processing:** probability map → threshold 0.5 → morphological cleanup.
- **Ensemble:** probability-map averaging across the three pretrained models.
- **From-scratch baseline:** single-phase 50-epoch training, no pretrained weights.

### Stage 3 — Pipeline Integration

```
predict(image)
 ├─ classifier.predict(image) → label
 └─ if label in {benign, malignant}:
        segmentation.predict(image) → mask
   else:
        return (label, None)
```

End-to-end demo on test samples is provided in Step 18 of the notebook.

---

## Results

All numbers below are on the **held-out 15 % stratified test set (n = 117)**.

### Classification — Test-set metrics

| Metric | EfficientNet-B4 | Swin-T | MobileViT-S | **Ensemble** |
|---|---:|---:|---:|---:|
| Accuracy | 0.8034 | 0.8547 | 0.8205 | **0.8632** |
| AUC-ROC (Macro) | 0.9391 | **0.9438** | 0.9260 | 0.9425 |
| Macro F1 | 0.8097 | 0.8481 | 0.8102 | **0.8612** |
| MCC | 0.6829 | 0.7503 | 0.7020 | **0.7729** |
| Sensitivity (benign) | 0.7576 | 0.8788 | 0.8182 | 0.8485 |
| Sensitivity (malignant) | 0.7742 | 0.8065 | 0.7742 | 0.8065 |
| Sensitivity (normal) | 1.0000 | 0.8500 | 0.9000 | 1.0000 |
| Specificity (malignant) | 0.8721 | 0.9186 | 0.9070 | **0.9419** |

> **Best classifier:** Weighted ensemble — Accuracy **0.8632**, Macro F1 **0.8612**.

### Segmentation — Test-set metrics

| Metric | Att-UNet | U-Net++ | TransUNet | Ensemble | Att-UNet *(Scratch)* |
|---|---:|---:|---:|---:|---:|
| Dice | 0.7602 | **0.7933** | 0.7788 | 0.7770 | 0.7459 |
| IoU | 0.6696 | **0.7035** | 0.6893 | 0.6911 | 0.6571 |
| Pixel Accuracy | 0.9595 | **0.9644** | 0.9627 | 0.9631 | 0.9537 |
| Sensitivity | 0.7859 | **0.8128** | 0.7718 | 0.7837 | 0.7726 |
| Specificity | 0.9816 | 0.9838 | **0.9871** | 0.9852 | 0.9799 |
| Hausdorff-95 (px) | 28.64 | **25.00** | 26.02 | 26.82 | 26.60 |

> **Best segmenter (pretrained):** **U-Net++** — Dice **0.7933**, IoU **0.7035**, Hausdorff-95 **25.00 px**.
> **From-scratch baseline** (no pretrained encoder) still reaches **0.7459 Dice**, confirming the architecture itself is sound; the ~5 pt Dice gain comes from ImageNet pretraining.

### 95 % Bootstrap Confidence Intervals (test set)

```
Ensemble Classifier:
  Accuracy                0.8638 (0.7949 – 0.9231)
  AUC-ROC (Macro)         0.9424 (0.9014 – 0.9766)
  Macro F1                0.8590 (0.7888 – 0.9212)
  MCC                     0.7726 (0.6726 – 0.8710)

UNet++:
  Dice                    0.7933 (0.7415 – 0.8394)
  IoU                     0.7035 (0.6567 – 0.7493)
  Hausdorff-95           24.9965 (12.5325 – 43.1397)

Ensemble Segmentation:
  Dice                    0.7770 (0.7193 – 0.8268)
  Hausdorff-95           26.8179 (14.3808 – 44.4130)
```

---

## Repository Structure

```
BUSI-2_Stage_Pipeline/
├── README.md                       # ← this file
├── BUSI_FDL_(Final_V).ipynb        # full end-to-end notebook (Steps 0–19)
│
├── Presentation/                   # final presentation deck
│   ├── BUSI_Presentation.pptx
│   └── BUSI_Presentation.pdf
│
├── Explanation/                    # per-step and per-model explanations
│   ├── busi_notebook_step_by_step_summary.md
│   ├── CLF Models/
│   │   ├── classification_full_explanation.md
│   │   ├── efficientnet_b4_explanation.md
│   │   ├── swin_transformer_tiny_explanation.md
│   │   └── mobilevitv2_100_explanation.md
│   └── SEG Models/
│       ├── segmentation_full_explanation.md
│       ├── attention_unet_from_scratch_explanation.md
│       ├── attention_unet_pretrained_explanation.md
│       ├── unetplusplus_pretrained_explanation.md
│       └── transunet_pretrained_explanation.md
│
│
└── Model test/checkpoints/         # 🔒 NOT uploaded — see .gitignore
    ├── EfficientNet-B4_best.pth
    ├── Swin-T_best.pth
    ├── MobileViT-S_best.pth
    ├── Attention-UNet_best.pth
    ├── UNetplusplus_best.pth
    ├── TransUNet_best.pth
    ├── Attention-UNet_scratch_best.pth
    └── *_history.json + class_weights.json
```

> ⚠️ Model checkpoints (`*.pth`, ~750 MB total) are **not tracked in git**. Re-train using the notebook or request the weights from the authors.

---

## Setup & Installation

The project was developed on **Google Colab** with a single GPU, but it runs locally too.

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (≥ 8 GB VRAM recommended; 16 GB used during training)
- ~10 GB free disk space for dataset + checkpoints

### Install

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3. Install PyTorch (pick the right CUDA build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest
pip install timm albumentations opencv-python scikit-learn \
            matplotlib seaborn pandas numpy tqdm scipy jupyter
```

### Dataset

Download **BUSI** from Kaggle and place it as:

```
dataset/
└── original/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png
    │   └── ...
    ├── malignant/
    └── normal/
```

The notebook expects this layout by default (configurable via `BASE_DIR`).

---

## How to Run

Open the end-to-end notebook:

```bash
jupyter notebook "BUSI_FDL_(Final_V).ipynb"
```

The notebook is organised into **Steps 0–19**:

| Step | Purpose |
|---|---|
| 0 | Colab setup: Drive mount, GPU check, library install |
| 1 | Library imports |
| 2 | Configuration & hyperparameters |
| 3 | EDA + OR-mask merge |
| 4 | Stratified 70/15/15 split |
| 5 | Offline augmentation |
| 6 | Class-weight calculation |
| 7 | Dataset classes & DataLoaders |
| 8 | Classification model classes (EffNet-B4, Swin-T, MobileViT-S) |
| 9 | 3-phase training function |
| 10 | Train the three classifiers |
| 11 | Classification ensemble |
| 12 | Test-set evaluation + bootstrap CIs |
| 13 | Segmentation model classes (Att-UNet, UNet++, TransUNet) |
| 14 | Segmentation training function |
| 15 | Train segmentation models + from-scratch baseline |
| 16 | Segmentation ensemble |
| 17 | Test-set segmentation evaluation + CIs |
| 18 | Full pipeline integration demo |
| 19 | Final summary tables & comparison charts |

**To reproduce from scratch:** run cells in order. Each step saves its outputs (CSV splits, JSON histories, PNG charts) to disk.

**To use a pre-trained model for inference:** load the checkpoint with `torch.load(path)` and call `model.eval()` before passing a tensor through the classification or segmentation pipeline defined in Step 18.

---

## Tech Stack

- **Deep learning:** PyTorch 2.x, TorchVision, [timm](https://github.com/huggingface/pytorch-image-models)
- **Data & augmentation:** albumentations, OpenCV, PIL
- **Metrics:** scikit-learn, scipy (`directed_hausdorff`), custom Dice/IoU implementations
- **Visualisation:** matplotlib, seaborn
- **Notebook / experiment tracking:** Jupyter / Google Colab

---

## Limitations

- **Small dataset (780 images).** Even with augmentation, models are at risk of overfitting; the relatively wide bootstrap CIs reflect this.
- **Pipeline error propagation.** A false-normal prediction skips segmentation; this is a known weakness of two-stage designs. Severity is mitigated by the strong normal-class sensitivity (1.0 for the ensemble).
- **Duplicate images** in the original BUSI dataset are preserved in our split (consistent with BUSI's release); this can inflate apparent performance.
- **Single seed.** Reported numbers come from one seed (`42`). Cross-validation (5-fold) is listed as future work.
- **BUSI is collected at one site** with one scanner protocol — generalisation to other ultrasound devices is unverified.
- **No clinical validation.** This is an academic project, not a medical device.

---

## References

1. Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). *Dataset of breast ultrasound images: BUSI.* Data in Brief, 28, 104863.
2. Oktay, O. *et al.* (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL.
3. Zhou, Z. *et al.* (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* DLMIA.
4. Chen, J. *et al.* (2021). *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.* arXiv:2102.04306.
5. Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
6. Liu, Z. *et al.* (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV.
7. Mehta, S. & Rastegari, M. (2022). *MobileViTv2.*
8. World Health Organization — Breast cancer fact sheet (global burden statistics).

---

## Acknowledgements

This project was completed as part of the **Final Deep Learning (FDL)** course. Thanks to our supervisor and teammates for their feedback throughout the experiments.

---

## License

This repository is released for **academic and educational use only**. The BUSI dataset has its own licence — see the original source on Kaggle. Trained weights are not redistributed in this repo.

---

*Last updated: June 2026.*
