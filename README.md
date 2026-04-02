# Bone Fracture Detection (Object Detection + Binary Baselines)

MSE 446/623 course project (University of Waterloo). We train and evaluate object detection models to **localize** and **classify** fracture-related regions in arm/hand X-rays, and we add simple **binary CNN baselines** (“type present vs not”) for proposal-alignment and additional analysis.

## Project goal (guideline §4.1)
- **Goal**: Build reproducible ML pipelines to detect bone fractures in X-ray images (localization + class label), compare model variants, and analyze limitations.

## Problem studied (guideline §4.2)
- **Task**: Multi-class **object detection** (YOLO-format bounding boxes) over 6 fracture-related classes.
- **Motivation**: Assistive decision-support style workflow (highlight regions + category) rather than a single image-level label.

## Dataset
- **Source**: Kaggle dataset `pkdarabi/bone-fracture-detection-computer-vision-project` (YOLO-format boxes).
- **On disk**: `BoneFractureYolo8/{train,valid,test}/{images,labels}`.
- **Classes** (cleaned): `elbow positive`, `fingers positive`, `forearm fracture`, `humerus fracture`, `shoulder fracture`, `wrist positive`.

### Data cleaning (performed in `notebooks/EDA.ipynb`)
- **Merged duplicate label**: `"humerus"` (old ID 4) → `"humerus fracture"` (ID 3).
- **Reindexed classes**: shifted remaining IDs after removal.
- **Rebalanced**:
  - removed 150 `fingers positive` samples (overrepresented)
  - augmented 75 `wrist positive` samples (underrepresented)

## Repository structure (what to run)
- **`notebooks/EDA.ipynb`**: download + EDA + cleaning + rebalancing.
- **`notebooks/train_model.ipynb`**: baseline YOLOv8 training + validation metrics table.
- **`notebooks/train_model2.ipynb`**: controlled runs (longer training / controlled settings; artifacts under `results/training_artifacts/`).
- **`scripts/train_model_binary.py`**: Simple-CNN binary baselines + ROC/AUC + per-class runs.
- **`Jenna_Data/`**: additional team experiments (YOLO variants, YOLOv11, R-CNN prep notebooks, summary spreadsheet).
- **`results/`**: saved metrics/plots from runs (`binary_runs/`, `binary_runs_per_class/`, `training_artifacts/`).

## Setup (reproducible environment)
This repo is notebook-heavy but all experiments can be reproduced from a fresh environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## How to reproduce results

### 1) Run EDA + cleaning
Open and run `notebooks/EDA.ipynb`. This produces the cleaned/rebalanced label set and updates the dataset configuration used for training.

### 2) Train YOLO baseline
Open and run `notebooks/train_model.ipynb`.

**Baseline YOLOv8n (validation)**
- mAP50: **0.318**
- mAP50-95: **0.119**
- Precision: **0.410**, Recall: **0.348**

Per-class breakdown is printed in the notebook and summarized below (validation):

| Class             | P     | R     | mAP50 | mAP50-95 |
|------------------|-------|-------|-------|----------|
| elbow positive   | 0.127 | 0.103 | 0.034 | 0.012    |
| fingers positive | 0.314 | 0.271 | 0.221 | 0.066    |
| forearm fracture | 0.687 | 0.535 | 0.558 | 0.214    |
| shoulder fracture| 0.669 | 0.583 | 0.557 | 0.219    |
| wrist positive   | 0.254 | 0.250 | 0.220 | 0.084    |

### 3) Train Simple-CNN binary baselines (proposal-aligned)
The proposal originally mentioned “fracture vs no-fracture” classification. This dataset primarily contains **fracture-positive images with boxes**, and does not include a curated set of **normal (healthy) X-rays**. Instead, we run a set of binary classifiers:

> **“Is class k present in this image?” vs “not present”** (one run per class).

Run all classes (writes one subfolder per class under the output directory):

```bash
python scripts/train_model_binary.py \
  --data-root BoneFractureYolo8 \
  --mode has_class \
  --all-classes \
  --data-yaml BoneFractureYolo8/data.yaml \
  --epochs 25 \
  --out-dir results/binary_runs_per_class
```

**Binary CNN (type-vs-rest) ROC-AUC (validation)** (from `results/binary_runs_per_class/all_classes_summary.json`):
- elbow positive: **0.766**
- fingers positive: **0.892**
- forearm fracture: **0.676**
- shoulder fracture: **0.836**
- wrist positive: **0.904**
- humerus fracture: **undefined** (0 positives in validation split in this export)

Each class folder contains:
- `binary_cnn_metrics.csv` (per-epoch metrics)
- `binary_cnn_summary.json` (headline metrics)
- `binary_roc.json` and `roc_curve.png` (ROC/AUC + Youden threshold marker)

## Notes on evaluation and thresholds
- **YOLO detection metrics** (mAP/AP) evaluate **both** localization and correct class simultaneously (IoU match + class match).
- **Binary CNN ROC/AUC** is threshold-free; we also compute **Youden’s J** to suggest an operating threshold that maximizes \(TPR - FPR\) on validation.

## Team experiments (Jenna_Data)
`Jenna_Data/` contains:
- YOLO training notebooks (including YOLOv11 attempt)
- R-CNN / COCO conversion preparation notebook
- summary spreadsheet of YOLO runs

Large artifacts (dataset copies, zips, weights, runs) are ignored by `.gitignore` to keep the repository lightweight and reproducible.

## Kaggle baseline comparison (optional claim)
We can include a comparison to a popular Kaggle notebook baseline **only if** we cite the notebook link and ensure metrics are comparable (same split + metric definition). This repo currently does not store the Kaggle notebook URL/metrics, so add them here before making a “better than Kaggle” claim.