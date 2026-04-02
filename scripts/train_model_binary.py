#!/usr/bin/env python3
"""
Train a small CNN for binary image classification on the bone-fracture YOLO dataset.

Why this script exists
----------------------
The project proposal mentioned a simple CNN for binary fracture vs. no-fracture
classification alongside YOLO localization. This dataset is built for object
detection: almost every image has at least one fracture box, so a literal
"fracture vs. normal" task needs a source of normal X-rays (see --normal-dir).
This set is mostly fracture-positive images with boxes; there are typically no
standalone "healthy" scans unless you add them (--normal-dir) or many empty
label files (rare).

This script supports three labeling modes:

  any_box     y=1 if the label file has >=1 object, else y=0 (often only y=1 here).
  has_class   y=1 if any box has class id --class-id, else y=0 (e.g. elbow vs rest).
  normal_dir  y=1 for all images under train/valid that have any box;
              y=0 for images in --normal-dir (no labels required).

Metrics per epoch (validation): accuracy, precision, recall, F1 (binary).
After training, the best checkpoint is re-evaluated on the validation set once to
compute ROC curve and AUC (fast: one forward pass). Outputs: ``binary_roc.json``,
optional ``roc_curve.png`` (needs matplotlib).

Example
-------
  # Elbow-present vs. not (both classes usually exist in this dataset)
  python train_model_binary.py --data-root . --mode has_class --class-id 0 \\
      --epochs 30 --out-dir binary_runs

  # All six fracture types vs. not (one model per class; subfolders under out-dir)
  python train_model_binary.py --data-root BoneFractureYolo8 --all-classes \\
      --data-yaml BoneFractureYolo8/data.yaml --epochs 25 --out-dir binary_runs

  # Fracture vs. normal if you add a folder of normal X-rays (jpg/png)
  python train_model_binary.py --data-root . --mode normal_dir \\
      --normal-dir /path/to/normal_xrays --epochs 30

Dependencies: torch, torchvision, Pillow, numpy, tqdm; scikit-learn recommended
for ROC/AUC (``pip install scikit-learn``). matplotlib optional for the ROC plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    roc_auc_score = None  # type: ignore
    roc_curve = None  # type: ignore

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slugify_class_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return (s[:48] if s else "class")


def load_nc_names_from_yaml(yaml_path: Path) -> Tuple[int, List[str]]:
    if yaml is None:
        raise SystemExit(
            "Reading --data-yaml requires PyYAML: pip install pyyaml"
        )
    if not yaml_path.is_file():
        raise SystemExit(f"data-yaml not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    nc = int(cfg["nc"])
    names = list(cfg["names"])
    if len(names) != nc:
        raise SystemExit(f"data-yaml nc={nc} but len(names)={len(names)}")
    return nc, names


def youden_best_from_roc(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> Optional[dict]:
    """Maximize TPR - FPR (Youden's J). sklearn: len(thresholds)==len(fpr)-1."""
    if thresholds.size == 0 or len(fpr) < 2:
        return None
    n = min(len(thresholds), len(fpr) - 1, len(tpr) - 1)
    if n < 1:
        return None
    j = tpr[1 : n + 1] - fpr[1 : n + 1]
    k = int(np.argmax(j))
    return {
        "threshold_youden": float(thresholds[k]),
        "youden_j": float(j[k]),
        "tpr_at_youden": float(tpr[k + 1]),
        "fpr_at_youden": float(fpr[k + 1]),
    }


def binary_metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pred = (y_score >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def read_yolo_classes(label_path: Path) -> List[int]:
    if not label_path.is_file():
        return []
    classes: List[int] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                classes.append(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def list_images(images_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not images_dir.is_dir():
        return out
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def label_path_for_image(images_dir: Path, labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


@dataclass
class Sample:
    path: Path
    y: int  # 0 or 1


class BinaryFractureDataset(Dataset):
    """Image paths with binary labels."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        if augment:
            self.tf = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(float(s.y), dtype=torch.float32)
        return x, y


class SmallCNN(nn.Module):
    """Lightweight baseline CNN for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def collect_samples_any_box(
    images_dir: Path, labels_dir: Path
) -> Tuple[List[Sample], dict]:
    stats = {"n_pos": 0, "n_neg": 0, "skipped_no_label": 0}
    samples: List[Sample] = []
    for img in list_images(images_dir):
        lp = label_path_for_image(images_dir, labels_dir, img)
        cls_list = read_yolo_classes(lp)
        if not lp.is_file():
            stats["skipped_no_label"] += 1
            continue
        y = 1 if len(cls_list) > 0 else 0
        samples.append(Sample(img, y))
        stats["n_pos" if y else "n_neg"] += 1
    return samples, stats


def collect_samples_has_class(
    images_dir: Path, labels_dir: Path, class_id: int
) -> Tuple[List[Sample], dict]:
    stats = {"n_pos": 0, "n_neg": 0, "skipped_no_label": 0}
    samples: List[Sample] = []
    for img in list_images(images_dir):
        lp = label_path_for_image(images_dir, labels_dir, img)
        if not lp.is_file():
            stats["skipped_no_label"] += 1
            continue
        cls_list = read_yolo_classes(lp)
        y = 1 if class_id in cls_list else 0
        samples.append(Sample(img, y))
        stats["n_pos" if y else "n_neg"] += 1
    return samples, stats


def collect_samples_normal_dir(
    images_dir: Path,
    labels_dir: Path,
    normal_dir: Path,
    split: str,
) -> Tuple[List[Sample], dict]:
    """
    Positive: any image in split images/ with >=1 box in YOLO label.
    Negative: all images under normal_dir (same split name optional — we use all normals for train/val carefully).

    For train: normals are only used in training set; for val we duplicate strategy:
    we assign normals to both splits proportionally is messy — simpler: use all
    fracture images from valid for val positives, and a held-out portion of normals
    for val negatives. Here we pass split-specific normal lists from caller.
    """
    stats = {"n_pos": 0, "n_neg": 0, "skipped_no_label": 0}
    samples: List[Sample] = []
    for img in list_images(images_dir):
        lp = label_path_for_image(images_dir, labels_dir, img)
        if not lp.is_file():
            stats["skipped_no_label"] += 1
            continue
        cls_list = read_yolo_classes(lp)
        if len(cls_list) == 0:
            continue
        samples.append(Sample(img, 1))
        stats["n_pos"] += 1

    for img in list_images(normal_dir):
        samples.append(Sample(img, 0))
        stats["n_neg"] += 1

    return samples, stats


def split_normals_for_train_val(
    normal_paths: List[Path], val_fraction: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(normal_paths)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_fraction)) if len(paths) > 1 else 0
    if n_val >= len(paths):
        n_val = len(paths) // 5
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    if not train_paths and val_paths:
        train_paths, val_paths = val_paths[:-1], val_paths[-1:]
    return train_paths, val_paths


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    losses: List[float] = []
    criterion = nn.BCEWithLogitsLoss()
    all_logits: List[float] = []
    all_y: List[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(float(loss.item()))
        all_logits.extend(logits.detach().cpu().numpy().tolist())
        all_y.extend(y.detach().cpu().numpy().tolist())

    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    pred = (probs >= threshold).astype(np.int64)
    y_true = np.array(all_y).astype(np.int64)

    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": len(y_true),
    }


@torch.no_grad()
def val_probs_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns y_true (int 0/1) and positive-class probabilities."""
    model.eval()
    all_logits: List[float] = []
    all_y: List[float] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.extend(logits.detach().cpu().numpy().tolist())
        all_y.extend(y.detach().cpu().numpy().tolist())
    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits, dtype=np.float64)))
    y_true = np.array(all_y).astype(np.int64)
    return y_true, probs


def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir: Path,
    skip_plot: bool,
) -> Optional[dict]:
    """
    One validation pass worth of scores is already computed; this only runs
    sklearn + optional matplotlib (milliseconds to seconds).
    """
    if roc_curve is None or roc_auc_score is None:
        print(
            "ROC/AUC skipped: install scikit-learn (pip install scikit-learn)."
        )
        return None

    if y_true.size == 0:
        return None
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        print(
            f"ROC/AUC skipped: validation set needs both classes "
            f"(pos={n_pos}, neg={n_neg})."
        )
        skipped = {
            "skipped": True,
            "reason": "single_class_in_validation",
            "n_positive": n_pos,
            "n_negative": n_neg,
        }
        (out_dir / "binary_roc.json").write_text(
            json.dumps(skipped, indent=2), encoding="utf-8"
        )
        return skipped

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(roc_auc_score(y_true, y_score))
    # sklearn auc() matches roc_auc_score for binary; keep one number
    out: dict = {
        "roc_auc": roc_auc,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }

    yd = youden_best_from_roc(fpr, tpr, thresholds)
    if yd:
        out.update(yd)
        ydm = binary_metrics_at_threshold(
            y_true, y_score, yd["threshold_youden"]
        )
        out["youden_val_accuracy"] = ydm["accuracy"]
        out["youden_val_precision"] = ydm["precision"]
        out["youden_val_recall"] = ydm["recall"]
        out["youden_val_f1"] = ydm["f1"]

    roc_json = out_dir / "binary_roc.json"
    roc_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote ROC data (AUC={roc_auc:.4f}) to {roc_json}")
    if yd:
        print(
            f"  Youden J = {yd['youden_j']:.4f} at threshold = {yd['threshold_youden']:.4f} "
            f"(val precision={out['youden_val_precision']:.3f}, recall={out['youden_val_recall']:.3f})"
        )

    if not skip_plot and plt is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        if yd:
            ax.scatter(
                [yd["fpr_at_youden"]],
                [yd["tpr_at_youden"]],
                s=60,
                zorder=5,
                c="C1",
                label=(
                    f"Youden (t={yd['threshold_youden']:.3f}, "
                    f"J={yd['youden_j']:.3f})"
                ),
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Validation ROC curve (best weights)")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        fig.tight_layout()
        png_path = out_dir / "roc_curve.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {png_path}")
        out["plot_path"] = str(png_path)
    elif not skip_plot and plt is None:
        print("ROC plot skipped: install matplotlib (pip install matplotlib).")

    return out


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    csv_path: Path,
    weights_path: Path,
) -> List[dict]:
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[dict] = []
    best_f1 = -1.0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
            ],
        )
        writer.writeheader()

        epoch_iter: Iterable[int] = range(1, epochs + 1)
        if tqdm is not None:
            epoch_iter = tqdm(epoch_iter, desc="epochs")

        for epoch in epoch_iter:
            model.train()
            batch_losses: List[float] = []
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            val_metrics = evaluate(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
            history.append({**row, **{k: val_metrics[k] for k in ("tp", "tn", "fp", "fn")}})
            writer.writerow(row)
            cf.flush()

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), weights_path)

    return history


def run_one_binary_run(
    args: argparse.Namespace,
    root: Path,
    out_dir: Path,
    label_desc: str,
    stats: dict,
    train_samples: List[Sample],
    val_samples: List[Sample],
    class_id: Optional[int] = None,
    class_name: Optional[str] = None,
) -> dict:
    """Train one binary CNN, ROC/Youden, write artifacts under out_dir."""
    if len(train_samples) < 8:
        msg = f"Too few training samples ({len(train_samples)})"
        print(f"SKIP: {msg}")
        return {
            "skipped": True,
            "reason": msg,
            "data_root": str(root),
            "out_dir": str(out_dir),
            "class_id": class_id,
            "class_name": class_name,
            "label_description": label_desc,
            "stats": stats,
        }
    if len(val_samples) < 2:
        msg = f"Too few validation samples ({len(val_samples)})"
        print(f"SKIP: {msg}")
        return {
            "skipped": True,
            "reason": msg,
            "data_root": str(root),
            "out_dir": str(out_dir),
            "class_id": class_id,
            "class_name": class_name,
            "label_description": label_desc,
            "stats": stats,
        }

    pos_tr = sum(1 for s in train_samples if s.y == 1)
    neg_tr = len(train_samples) - pos_tr
    if neg_tr == 0 or pos_tr == 0:
        print(
            "WARNING: Training set is single-class for this mode. "
            "Metrics may be trivial."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Out: {out_dir}")
    print(f"Task: {label_desc}")
    print(f"Train samples: {len(train_samples)} (pos={pos_tr}, neg={neg_tr})")
    print(
        f"Val samples: {len(val_samples)} "
        f"(pos={sum(1 for s in val_samples if s.y == 1)})"
    )
    print(f"Stats: {json.dumps(stats, default=str)}")

    train_ds = BinaryFractureDataset(
        train_samples, image_size=args.image_size, augment=True
    )
    val_ds = BinaryFractureDataset(
        val_samples, image_size=args.image_size, augment=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = SmallCNN().to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "binary_cnn_metrics.csv"
    weights_path = out_dir / "binary_cnn_best.pt"

    history = train_loop(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        csv_path,
        weights_path,
    )

    roc_info: Optional[dict] = None
    if not args.skip_roc:
        try:
            state = torch.load(
                weights_path, map_location=device, weights_only=True
            )
        except TypeError:
            state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        y_true, y_score = val_probs_and_labels(model, val_loader, device)
        roc_info = compute_roc_auc(
            y_true, y_score, out_dir, skip_plot=args.no_roc_plot
        )

    best_row = max(history, key=lambda r: r["val_f1"])
    summary: dict = {
        "data_root": str(root),
        "mode": args.mode,
        "class_id": class_id,
        "class_name": class_name,
        "normal_dir": str(args.normal_dir) if args.normal_dir else None,
        "label_description": label_desc,
        "epochs_ran": args.epochs,
        "best_epoch_f1": best_row["epoch"],
        "best_val_f1": best_row["val_f1"],
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_precision": best_row["val_precision"],
        "best_val_recall": best_row["val_recall"],
        "metrics_csv": str(csv_path),
        "weights": str(weights_path),
        "stats": stats,
        "roc_auc_validation": (
            roc_info.get("roc_auc") if roc_info and "roc_auc" in roc_info else None
        ),
        "threshold_youden": (
            roc_info.get("threshold_youden")
            if roc_info and "threshold_youden" in roc_info
            else None
        ),
        "youden_j": (
            roc_info.get("youden_j") if roc_info and "youden_j" in roc_info else None
        ),
        "youden_val_precision": (
            roc_info.get("youden_val_precision")
            if roc_info and "youden_val_precision" in roc_info
            else None
        ),
        "youden_val_recall": (
            roc_info.get("youden_val_recall")
            if roc_info and "youden_val_recall" in roc_info
            else None
        ),
        "youden_val_f1": (
            roc_info.get("youden_val_f1")
            if roc_info and "youden_val_f1" in roc_info
            else None
        ),
        "roc_json": str(out_dir / "binary_roc.json")
        if (out_dir / "binary_roc.json").is_file()
        else None,
    }
    (out_dir / "binary_cnn_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary CNN on fracture YOLO dataset")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing train/, valid/, test/ with images/ and labels/",
    )
    p.add_argument(
        "--mode",
        choices=("any_box", "has_class", "normal_dir"),
        default="has_class",
        help="How to define binary labels (see docstring).",
    )
    p.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="For has_class: YOLO class index treated as positive (0=elbow in cleaned 6-class yaml).",
    )
    p.add_argument(
        "--all-classes",
        action="store_true",
        help="Train one binary model per YOLO class (type vs not); uses has_class. "
        "Writes subfolders under --out-dir and all_classes_summary.json.",
    )
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Dataset YAML with nc and names (for --all-classes). "
        "Default: <repo>/BoneFractureYolo8/data.yaml next to this script.",
    )
    p.add_argument(
        "--normal-dir",
        type=Path,
        default=None,
        help="For normal_dir: directory of normal (non-fracture) images.",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "binary_runs",
        help="Directory for metrics.csv, summary.json, best weights.",
    )
    p.add_argument(
        "--val-normal-fraction",
        type=float,
        default=0.2,
        help="Fraction of normal images reserved for validation (normal_dir mode).",
    )
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--skip-roc",
        action="store_true",
        help="Do not run post-training ROC/AUC on the validation set.",
    )
    p.add_argument(
        "--no-roc-plot",
        action="store_true",
        help="Compute ROC/AUC and JSON but do not save roc_curve.png.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_classes and args.mode != "has_class":
        raise SystemExit("--all-classes only supports --mode has_class")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = args.data_root.resolve()
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    val_img = root / "valid" / "images"
    val_lbl = root / "valid" / "labels"

    if args.all_classes:
        script_dir = Path(__file__).resolve().parent
        yaml_path = (
            args.data_yaml.resolve()
            if args.data_yaml is not None
            else (script_dir / "BoneFractureYolo8" / "data.yaml").resolve()
        )
        nc, names = load_nc_names_from_yaml(yaml_path)
        base_out = args.out_dir.resolve()
        base_out.mkdir(parents=True, exist_ok=True)
        all_rows: List[dict] = []

        class_pairs = list(enumerate(names))
        outer = class_pairs
        if tqdm is not None:
            outer = tqdm(class_pairs, desc="per-class models")

        for cid, cname in outer:
            sub = base_out / f"class_{cid:02d}_{slugify_class_name(cname)}"
            train_samples, st_tr = collect_samples_has_class(
                train_img, train_lbl, cid
            )
            val_samples, st_va = collect_samples_has_class(
                val_img, val_lbl, cid
            )
            label_desc = f"has class_id={cid} ({cname}) vs not"
            stats = {"train": st_tr, "val": st_va, "class_name": cname}
            summary = run_one_binary_run(
                args,
                root,
                sub,
                label_desc,
                stats,
                train_samples,
                val_samples,
                class_id=cid,
                class_name=cname,
            )
            all_rows.append(summary)

        agg_path = base_out / "all_classes_summary.json"
        agg_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print(f"Wrote aggregate summary: {agg_path}")
        return

    if args.mode == "normal_dir":
        if args.normal_dir is None:
            raise SystemExit("--normal-dir is required when --mode normal_dir")
        normal_dir = args.normal_dir.resolve()
        if not normal_dir.is_dir():
            raise SystemExit(f"normal-dir not found: {normal_dir}")

        train_pos, st_tr = collect_samples_any_box(train_img, train_lbl)
        train_pos = [s for s in train_pos if s.y == 1]
        val_pos, st_va = collect_samples_any_box(val_img, val_lbl)
        val_pos = [s for s in val_pos if s.y == 1]

        normals = list_images(normal_dir)
        if len(normals) < 4:
            raise SystemExit("Need at least a few normal images for normal_dir mode.")
        tr_n_paths, va_n_paths = split_normals_for_train_val(
            normals, args.val_normal_fraction, args.seed
        )
        train_samples = train_pos + [Sample(p, 0) for p in tr_n_paths]
        val_samples = val_pos + [Sample(p, 0) for p in va_n_paths]
        label_desc = "fracture (any box) vs normal images from --normal-dir"
        stats = {
            "train_pos": len(train_pos),
            "train_neg": len(tr_n_paths),
            "val_pos": len(val_pos),
            "val_neg": len(va_n_paths),
        }
    elif args.mode == "has_class":
        train_samples, st_tr = collect_samples_has_class(
            train_img, train_lbl, args.class_id
        )
        val_samples, st_va = collect_samples_has_class(
            val_img, val_lbl, args.class_id
        )
        label_desc = f"has class_id={args.class_id} vs not"
        stats = {"train": st_tr, "val": st_va}
    else:
        train_samples, st_tr = collect_samples_any_box(train_img, train_lbl)
        val_samples, st_va = collect_samples_any_box(val_img, val_lbl)
        label_desc = "any YOLO box vs empty label file"
        stats = {"train": st_tr, "val": st_va}

    if len(train_samples) < 8:
        raise SystemExit(
            f"Too few training samples ({len(train_samples)}). "
            f"Is --data-root correct? Expected {root}/train/images with images. "
            f"Download/clean data per EDA.ipynb first."
        )
    if len(val_samples) < 2:
        raise SystemExit(f"Too few validation samples ({len(val_samples)}).")

    out_dir = args.out_dir.resolve()
    cid = args.class_id if args.mode == "has_class" else None
    run_one_binary_run(
        args,
        root,
        out_dir,
        label_desc,
        stats,
        train_samples,
        val_samples,
        class_id=cid,
        class_name=None,
    )


if __name__ == "__main__":
    main()
