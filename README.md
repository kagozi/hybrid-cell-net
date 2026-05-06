# HybridCellNet: Merging Classical Vision Features and Deep Learning for Interpretable Blood Cell Analysis
---

## Overview

HybridCellNet is a hybrid feature-fusion framework for automated blood cell classification. It combines:

- **Classical descriptors** — Histogram of Oriented Gradients (HOG) and Gray-Level Co-occurrence Matrix (GLCM) texture features
- **Deep features** — MobileNetV3-Small pretrained on ImageNet, fine-tuned on cell crops
- **Explicit concatenation** of both into a 2,356-d fused representation before a dense classification head
- **Grad-CAM** visual explanations on the deep branch for clinical interpretability

The system classifies individual blood cell crops into three categories: **RBC**, **WBC**, and **Platelet**.

### Results (BCCD test set — 945 crops)

| Model | Accuracy | Macro F1 |
|---|---|---|
| HOG + GLCM → SVM (baseline) | 97.35% | 0.9273 |
| MobileNetV3-Small only (baseline) | 99.79% | 0.9926 |
| **HybridCellNet (ours)** | **99.89%** | **0.9952** |

Per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RBC | 1.00 | 1.00 | 1.00 | 805 |
| WBC | 0.99 | 1.00 | 0.99 | 71 |
| Platelet | 1.00 | 0.99 | 0.99 | 69 |

---

## Architecture

```
Input cell crop (224×224×3)
        │
   Preprocessing
   (CLAHE · Gaussian blur · Otsu · Morphology)
        │
   Watershed segmentation
   (Marker-controlled cell isolation)
        │
   ┌────┴────────────────────┐
   │                         │
Handcrafted branch       Deep branch
HOG (1,764-d)            MobileNetV3-Small
GLCM (16-d)              AdaptiveAvgPool → 576-d
BatchNorm1d              BatchNorm1d
   │                         │
   └────────────┬────────────┘
                │
        Concatenation (2,356-d)
                │
        Dense(512) → ReLU → Dropout(0.4)
        Dense(128) → ReLU → Dropout(0.4)
        Softmax → 3 classes
                │
        RBC / WBC / Platelet
```

---

## Dataset

The project uses the **BCCD (Blood Cell Count and Detection)** dataset.

**Download:** [https://www.kaggle.com/datasets/orvile/bccd-blood-cell-count-and-detection-dataset](https://www.kaggle.com/datasets/orvile/bccd-blood-cell-count-and-detection-dataset)

The dataset contains 364 annotated blood smear microscope images with bounding-box labels in three classes (RBC, WBC, Platelet), pre-split into `train/`, `val/`, and `test/` directories.

### Download via Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Place your kaggle.json API token at ~/.kaggle/kaggle.json
# Get it from: https://www.kaggle.com/settings → API → Create New Token

kaggle datasets download -d orvile/bccd-blood-cell-count-and-detection-dataset
unzip bccd-blood-cell-count-and-detection-dataset.zip -d data/
```

Expected folder structure after extraction:

```
data/
├── train/
│   ├── img/        ← .jpeg images
│   └── ann/        ← .json annotation files
├── val/
│   ├── img/
│   └── ann/
└── test/
    ├── img/
    └── ann/
```

---

## Running on Kaggle (recommended)

The notebook is designed to run on Kaggle with GPU acceleration.

1. Go to [kaggle.com](https://www.kaggle.com) and create a free account
2. Click **+ New Notebook**
3. Upload `hybridcellnet.ipynb`
4. Under **Data** (right panel) → **Add Data** → search for  
   `orvile/bccd-blood-cell-count-and-detection-dataset` and add it
5. Under **Settings** → **Accelerator** → select **GPU T4 x2** or **P100**
6. Click **Run All**

The dataset path in the notebook is already set to:
```python
BCCD_PATH = '/kaggle/input/datasets/orvile/bccd-blood-cell-count-and-detection-dataset'
```

All outputs are saved to `/kaggle/working/HybridCellNet/`.

---

## Running Locally

### 1. Clone the repository

```bash
git https://github.com/kagozi/hybrid-cell-net.git
cd hybridcellnet
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Follow the [Kaggle API instructions](#download-via-kaggle-api) above, then update the dataset path in the notebook:

```python
# Cell 2 — change this line
BCCD_PATH = 'data/'               # point to your local extraction folder
```

### 5. Launch the notebook

```bash
jupyter notebook hybridcellnet.ipynb
```

Run cells sequentially from Cell 1 through Cell 13.

---

## Notebook Structure

| Cell | Description |
|---|---|
| 1 | Installations, imports, device setup, random seed |
| 2 | Dataset paths, structure verification, class distribution |
| 3 | Preprocessing pipeline (CLAHE, Gaussian blur, Otsu, morphology) |
| 4 | Marker-controlled watershed segmentation |
| 5 | Handcrafted feature extraction (HOG + GLCM) |
| 6 | Cell-crop dataset class (`BCCDCropDataset`) |
| 7 | Weighted sampler and DataLoaders |
| 8 | HybridCellNet model definition |
| 9–10 | Training loop and training curves |
| 11 | Final evaluation on test set (accuracy, F1, confusion matrix) |
| 12 | Grad-CAM visual explanations |
| 13 | Ablation study (SVM baseline, deep-only baseline, HybridCellNet) |

---

## Output Files

After running all cells, the following are saved to `/kaggle/working/HybridCellNet/` (or your local working directory):

```
HybridCellNet/
├── best_hybridcellnet.pth          ← best model checkpoint
├── 00_sample_images.png
├── 01_preprocessing.png
├── 02_segmentation.png
├── 03_hog_demo.png
├── 04_training_curves.png
├── 05_confusion_matrix.png
├── 06_gradcam.png
├── 07_ablation_study.png
├── 08_confusion_matrices_all.png
├── test_results.csv
└── ablation_study.csv
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | AdamW (backbone lr=2e-5, head lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine Annealing (T_max=30) |
| Loss | Cross-entropy with inverse class-frequency weights |
| Epochs | 30 |
| Batch size | 16 |
| Imbalance handling | WeightedRandomSampler + class-weighted loss |
| Best model criterion | Macro-F1 → accuracy → val loss (tiebreakers) |
| Deep branch frozen | First 9/13 MobileNetV3 blocks |
| Dropout | 0.4 on both dense layers |

---

## Requirements

See `requirements.txt`. Core dependencies:

- Python 3.9+
- PyTorch 2.0+
- torchvision
- scikit-image
- scikit-learn
- OpenCV
- matplotlib / seaborn
---

## License

This project is released for academic and educational use.  
The BCCD dataset is provided by its original authors via Kaggle and is subject to its own license terms.
