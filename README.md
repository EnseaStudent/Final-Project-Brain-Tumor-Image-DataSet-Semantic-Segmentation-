# Brain Tumor Semantic Segmentation (CE6146 Final Project)

This repository contains our final project for **CE6146 – Introduction to Deep Learning** (National Central University, 2025). The objective is to train deep learning models that **detect and segment brain tumors in MRI images**.

## Team

- Anthonin Guillaume (114522902)
- Lynna Moussaoui (114522906)
- Lina Bachir Belarbi (114522903)
- Saïd Graja (114522901)

## Course context

The course expects:
- a **presentation** (problem, methods, experiments, results)
- a **GitHub repository** with code, documentation, and visualizations to ensure reproducibility

## Dataset

We use the **Brain Tumor Image Dataset – Semantic Segmentation** (Kaggle).

Key points:
- MRI brain images with polygon annotations
- COCO-style annotation files: `_annotations.coco.json`
- The project uses `train/`, `valid/`, and `test/` splits
- Tumor regions are often much smaller than the background, which creates **strong class imbalance** at the pixel level

## Method

Our workflow has three main steps:

1. **Baseline (U-Net from scratch)**
   - U-Net built with `segmentation-models-pytorch`
   - Encoder: ResNet34 **without** pre-trained weights
   - Loss: **BCE + Dice**
   - Metrics: Dice (and optional IoU)

2. **Transfer learning (pre-trained encoder)**
   - Same U-Net architecture
   - Encoder: ResNet34 **pre-trained on ImageNet**
   - Fine-tuning with a smaller learning rate
   - Loss: **BCE + Dice**

3. **Loss function comparison (imbalanced segmentation)**
   - Keep the same pre-trained U-Net (ResNet34 encoder)
   - Compare:
     - **BCE + Dice**
     - **Focal Loss**
     - **Tversky Loss**

## Implementation details

### Libraries

- PyTorch
- `segmentation-models-pytorch`
- `albumentations` (augmentations)
- `timm` (backbones)
- `kagglehub` (dataset download)

### Preprocessing and augmentations

- Resize to `256 × 256`
- Train-time augmentation:
  - horizontal flips
  - random 90° rotations
  - small shift/scale/rotate
- Normalization uses mean/std `(0.5, 0.5, 0.5)`

### Dataset and masks

A custom dataset class reads the COCO annotations and builds a **binary mask**:
- tumor pixels = `1`
- background pixels = `0`

Polygons are rendered into masks using PIL.

### Training

Typical settings used in the notebook:

- Epochs: `15`
- Batch size: `8`
- Optimizer: Adam
- Learning rate:
  - baseline: `1e-3`
  - transfer learning / loss comparison: `1e-4`

### Checkpoints

The best model (highest validation Dice) is saved during training:

- `unet_baseline_best.pth`
- `unet_resnet34_pretrained_best.pth`
- `unet_resnet34_focal_best.pth`
- `unet_resnet34_tversky_best.pth`

### Visualizations

The notebook includes:
- qualitative examples (input / ground truth / prediction)
- Dice and loss curves (baseline vs transfer learning)
- validation Dice curves for different losses
- bar plot of best validation Dice per configuration

## Results (summary)

From our recorded runs:

| Configuration | Best validation Dice |
|---|---:|
| Baseline U-Net (ResNet34, no pretraining) + BCE + Dice | ~0.7647 |
| Transfer learning U-Net (ResNet34 pretrained) + BCE + Dice | ~0.7881 |
| Transfer learning U-Net (ResNet34 pretrained) + Focal Loss | ~0.7308 |
| Transfer learning U-Net (ResNet34 pretrained) + Tversky Loss | ~0.7724 |

**Best overall:** U-Net with a **pre-trained ResNet34 encoder** trained with **BCE + Dice**.

## How to run

### Option A — Google Colab (recommended)

1. Install dependencies:

```bash
pip install -q segmentation-models-pytorch==0.5.0 timm==1.0.22 albumentations==1.4 kagglehub
```

2. Download the dataset with `kagglehub` in the notebook:

```python
import kagglehub
path = kagglehub.dataset_download("pkdarabi/brain-tumor-image-dataset-semantic-segmentation")
```

3. Run the notebook sections in order:
- dataset exploration
- baseline training
- transfer learning training
- focal / tversky experiments
- plots and qualitative results

### Option B — Local run

You can run locally if you have a working PyTorch environment (GPU recommended). The easiest path is to convert the notebook to a script and execute it step-by-step, because the project was originally written for Colab.

## Repository notes

- The project was initially developed in **Colab** and later exported.
- You may find Git/GDrive helper cells in the code because the workflow included pushing the notebook to GitHub from Colab.

