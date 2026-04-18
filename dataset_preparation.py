#!/usr/bin/env python3
"""
dataset_preparation.py — Merge, remap, split, label, and augment datasets.

Scans ``data/raw/`` for all image+label pairs across available datasets,
remaps class IDs to the unified ``UNIFIED_CLASSES`` mapping, generates
synthetic RL/Qd regression labels, splits 80/10/10 train/val/test, and
optionally runs Albumentations augmentation.

Usage:
    python3 dataset_preparation.py                  # full run
    python3 dataset_preparation.py --dry-run         # scan-only report
    python3 dataset_preparation.py --no-augment      # skip augmentation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from config import (
    AUGMENTED_DIR,
    PROCESSED_DIR,
    PROCESSED_IMAGES,
    PROCESSED_LABELS,
    PROJECT_ROOT,
    RAW_DIR,
    RL_QD_LABELS_CSV,
    DATASET_YAML,
    UNIFIED_CLASSES,
    NUM_CLASSES,
    TrainConfig,
)
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Dataset-specific class remapping tables
# ---------------------------------------------------------------------------

# archive/ dataset: 13 classes (road markings – Polish/English)
_REMAP_ARCHIVE: Dict[int, int] = {
    0: 5,    # BUS LANE → stop_line (closest semantic match)
    1: 1,    # Yellow Markings → yellow_lane_marking
    2: 0,    # Line 1 → white_lane_marking
    3: 0,    # Line 2 → white_lane_marking
    4: 4,    # Crossing → pedestrian_crossing
    5: 2,    # Romb → arrow_marking (diamond shape)
    6: 5,    # SLOW → stop_line (text marking)
    7: 2,    # Left Arrow → arrow_marking
    8: 2,    # Forward Arrow → arrow_marking
    9: 2,    # Forward Arrow-Left → arrow_marking
    10: 2,   # Forward Arrow-Right → arrow_marking
    11: 2,   # Right Arrow → arrow_marking
    12: 4,   # Bicycle → pedestrian_crossing (shared lane marking)
}

# Traffic signs dataset (if present): typical GTSDB-style IDs
_REMAP_TRAFFIC_SIGNS: Dict[int, int] = {
    0: 6,    # warning → traffic_sign_warning
    1: 7,    # mandatory → traffic_sign_mandatory
    2: 8,    # informatory → traffic_sign_informatory
    3: 7,    # prohibitory → traffic_sign_mandatory
    4: 6,    # danger → traffic_sign_warning
}


def _polygon_to_bbox(coords: List[float]) -> Tuple[float, float, float, float]:
    """Convert YOLO polygon coordinates to bounding box (cx, cy, w, h)."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return (cx, cy, w, h)


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def scan_archive_dataset() -> List[Tuple[Path, Path]]:
    """Scan the archive/ directory for image+label pairs."""
    pairs: List[Tuple[Path, Path]] = []
    archive_dir = PROJECT_ROOT / "archive"
    if not archive_dir.exists():
        return pairs

    for split in ("train", "valid", "test"):
        img_dir = archive_dir / split / "images"
        lbl_dir = archive_dir / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs


def scan_raw_datasets() -> Dict[str, List[Tuple[Path, Path]]]:
    """Scan data/raw/ sub-directories for additional datasets."""
    result: Dict[str, List[Tuple[Path, Path]]] = {}
    for dataset_dir in sorted(RAW_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        pairs: List[Tuple[Path, Path]] = []
        # Try common layout patterns
        for sub in ("train", "valid", "test", ""):
            img_dir = dataset_dir / sub / "images" if sub else dataset_dir / "images"
            lbl_dir = dataset_dir / sub / "labels" if sub else dataset_dir / "labels"
            if img_dir.exists() and lbl_dir.exists():
                for img_path in img_dir.glob("*.*"):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        lbl_path = lbl_dir / (img_path.stem + ".txt")
                        if lbl_path.exists():
                            pairs.append((img_path, lbl_path))
        if pairs:
            result[dataset_dir.name] = pairs
    return result


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def is_quality_ok(img_path: Path, min_size: int = 64) -> bool:
    """Reject too-small or corrupted images.

    Parameters
    ----------
    img_path : Path
        Image file path.
    min_size : int
        Minimum width and height in pixels.

    Returns
    -------
    bool
        ``True`` if the image passes quality checks.
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        if h < min_size or w < min_size:
            return False
        # Reject if too dark (mean brightness < 15)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 15:
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

def remap_labels(
    lbl_path: Path,
    remap_table: Dict[int, int],
    convert_polygon: bool = True,
) -> List[str]:
    """Read a YOLO label file and remap class IDs.

    Parameters
    ----------
    lbl_path : Path
        Original YOLO label file.
    remap_table : Dict[int, int]
        ``{original_class_id: unified_class_id}``.
    convert_polygon : bool
        If ``True``, convert polygon annotations to bounding boxes.

    Returns
    -------
    List[str]
        Remapped label lines in YOLO bbox format ``cls cx cy w h``.
    """
    lines: List[str] = []
    with open(lbl_path) as fh:
        for raw in fh:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(parts[0])
            new_cls = remap_table.get(old_cls)
            if new_cls is None:
                continue  # skip unmapped classes

            coords = list(map(float, parts[1:]))
            if len(coords) == 4:
                # Already cx, cy, w, h
                cx, cy, w, h = coords
            elif convert_polygon and len(coords) >= 4:
                # Polygon → bbox
                cx, cy, w, h = _polygon_to_bbox(coords)
            else:
                continue

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))

            lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ---------------------------------------------------------------------------
# Synthetic RL/Qd label generation
# ---------------------------------------------------------------------------

def generate_synthetic_rl(
    class_name: str,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """Generate a synthetic RL/Qd label for training.

    Parameters
    ----------
    class_name : str
        Unified class name.
    rng : np.random.Generator | None
        Optional numpy RNG for reproducibility.

    Returns
    -------
    Dict[str, float]
        Keys: ``rl_label``, ``qd_label``, ``simulated_distance``,
        ``simulated_humidity``, ``simulated_tilt``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base RL depends on surface type
    if "yellow" in class_name:
        base_rl = rng.normal(loc=280, scale=100)
    elif "sign" in class_name or "gantry" in class_name:
        base_rl = rng.normal(loc=120, scale=60)
    elif "delineator" in class_name or "stud" in class_name:
        base_rl = rng.normal(loc=350, scale=130)
    else:  # white markings, arrows, crossings
        base_rl = rng.normal(loc=350, scale=120)

    degradation = rng.uniform(0.4, 1.0)
    sim_humidity = rng.uniform(30, 95)
    sim_tilt = rng.uniform(0, 8)
    sim_distance = rng.uniform(200, 4000)

    rl_label = max(10.0, base_rl * degradation)
    if sim_humidity > 70:
        rl_label *= 0.85
    if sim_tilt > 5:
        rl_label *= 0.95

    qd_label = float(np.clip(rng.normal(0.5, 0.15), 0.05, 0.95))

    return {
        "rl_label": round(rl_label, 2),
        "qd_label": round(qd_label, 4),
        "simulated_distance": round(sim_distance, 1),
        "simulated_humidity": round(sim_humidity, 1),
        "simulated_tilt": round(sim_tilt, 2),
    }


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def get_augmentation_pipeline():
    """Return the Albumentations augmentation pipeline.

    Returns
    -------
    albumentations.Compose
        Configured pipeline with bbox support.
    """
    import albumentations as A

    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.RandomRain(rain_type="heavy", p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.25),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.CLAHE(p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    pairs: List[Tuple[Path, Path, List[str]]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[list, list, list]:
    """Split image+label pairs by dominant class.

    Parameters
    ----------
    pairs : list
        Each element is ``(img_path, lbl_path, remapped_lines)``.
    ratios : tuple
        ``(train, val, test)`` fractions.
    seed : int
        Random seed.

    Returns
    -------
    Tuple[list, list, list]
        ``(train, val, test)`` subsets.
    """
    by_class: Dict[int, list] = defaultdict(list)
    for item in pairs:
        lines = item[2]
        if lines:
            dominant = int(lines[0].split()[0])
        else:
            dominant = 0
        by_class[dominant].append(item)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls_items in by_class.values():
        rng.shuffle(cls_items)
        n = len(cls_items)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        train.extend(cls_items[:n_train])
        val.extend(cls_items[n_train : n_train + n_val])
        test.extend(cls_items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, no_augment: bool = False) -> None:
    """Execute the full dataset preparation pipeline.

    Parameters
    ----------
    dry_run : bool
        If ``True``, scan and report only — no files written.
    no_augment : bool
        Skip augmentation step.
    """
    logger.info("=== Dataset Preparation Pipeline ===")

    # 1. Scan datasets
    archive_pairs = scan_archive_dataset()
    raw_datasets = scan_raw_datasets()

    total_raw = len(archive_pairs) + sum(len(v) for v in raw_datasets.values())
    logger.info(f"Found {len(archive_pairs)} archive pairs")
    for ds_name, ds_pairs in raw_datasets.items():
        logger.info(f"Found {len(ds_pairs)} pairs in data/raw/{ds_name}")
    logger.info(f"Total raw pairs: {total_raw}")

    if dry_run:
        n_est = int(total_raw * 0.95)  # ~5% quality rejection
        print(f"Found {total_raw} images across datasets. Estimated output: {n_est} processed pairs.")
        return

    # 2. Quality filter + remap
    processed: List[Tuple[Path, Path, List[str]]] = []

    logger.info("Processing archive dataset...")
    for img_p, lbl_p in tqdm(archive_pairs, desc="Archive"):
        if not is_quality_ok(img_p):
            continue
        lines = remap_labels(lbl_p, _REMAP_ARCHIVE, convert_polygon=True)
        if lines:
            processed.append((img_p, lbl_p, lines))

    for ds_name, ds_pairs in raw_datasets.items():
        remap = _REMAP_TRAFFIC_SIGNS if "sign" in ds_name.lower() else _REMAP_ARCHIVE
        logger.info(f"Processing {ds_name}...")
        for img_p, lbl_p in tqdm(ds_pairs, desc=ds_name):
            if not is_quality_ok(img_p):
                continue
            lines = remap_labels(lbl_p, remap, convert_polygon=True)
            if lines:
                processed.append((img_p, lbl_p, lines))

    logger.info(f"After quality filter: {len(processed)} pairs")

    # 3. Stratified split
    train, val, test = stratified_split(processed)
    logger.info(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    # 4. Copy files to processed/
    rl_qd_rows: List[Dict] = []
    rng = np.random.default_rng(42)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        img_out = PROCESSED_IMAGES / split_name
        lbl_out = PROCESSED_LABELS / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, _, remapped_lines in tqdm(split_data, desc=f"Write {split_name}"):
            # Unique filename via hash
            uid = hashlib.md5(str(img_path).encode()).hexdigest()[:8]
            new_name = f"{img_path.stem}_{uid}"
            dst_img = img_out / f"{new_name}.jpg"
            dst_lbl = lbl_out / f"{new_name}.txt"

            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, "w") as fh:
                fh.write("\n".join(remapped_lines) + "\n")

            # Generate synthetic RL/Qd for each object
            for line in remapped_lines:
                parts = line.split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                class_name = UNIFIED_CLASSES.get(cls_id, "unknown")
                synth = generate_synthetic_rl(class_name, rng)
                rl_qd_rows.append({
                    "image_path": str(dst_img.relative_to(PROJECT_ROOT)),
                    "bbox_x": cx,
                    "bbox_y": cy,
                    "bbox_w": bw,
                    "bbox_h": bh,
                    "object_type": class_name,
                    **synth,
                })

    # 5. Write RL/Qd CSV
    RL_QD_LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "object_type", "rl_label", "qd_label",
        "simulated_distance", "simulated_humidity", "simulated_tilt",
    ]
    with open(RL_QD_LABELS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rl_qd_rows)
    logger.info(f"Wrote {len(rl_qd_rows)} RL/Qd labels → {RL_QD_LABELS_CSV}")

    # 6. Write dataset.yaml
    yaml_content = {
        "path": str(PROCESSED_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": {k: v for k, v in UNIFIED_CLASSES.items()},
    }
    with open(DATASET_YAML, "w") as fh:
        yaml.dump(yaml_content, fh, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote dataset YAML → {DATASET_YAML}")

    # 7. Augmentation (train set only)
    if not no_augment:
        try:
            aug = get_augmentation_pipeline()
            aug_img_dir = AUGMENTED_DIR / "images"
            aug_lbl_dir = AUGMENTED_DIR / "labels"
            aug_img_dir.mkdir(parents=True, exist_ok=True)
            aug_lbl_dir.mkdir(parents=True, exist_ok=True)

            train_img_dir = PROCESSED_IMAGES / "train"
            train_lbl_dir = PROCESSED_LABELS / "train"
            aug_count = 0

            for img_file in tqdm(sorted(train_img_dir.glob("*.jpg")), desc="Augmenting"):
                lbl_file = train_lbl_dir / (img_file.stem + ".txt")
                if not lbl_file.exists():
                    continue

                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Parse labels
                bboxes, class_labels = [], []
                with open(lbl_file) as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_labels.append(int(parts[0]))
                            bboxes.append([float(x) for x in parts[1:5]])

                if not bboxes:
                    continue

                for aug_i in range(TrainConfig().augmentation_factor):
                    try:
                        result = aug(
                            image=img_rgb,
                            bboxes=bboxes,
                            class_labels=class_labels,
                        )
                        aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                        aug_name = f"{img_file.stem}_aug{aug_i}"
                        cv2.imwrite(str(aug_img_dir / f"{aug_name}.jpg"), aug_img)

                        with open(aug_lbl_dir / f"{aug_name}.txt", "w") as fh:
                            for cls, bb in zip(result["class_labels"], result["bboxes"]):
                                fh.write(f"{cls} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
                        aug_count += 1
                    except Exception:
                        continue  # skip if augmentation fails for this sample

            logger.info(f"Augmented {aug_count} images → {AUGMENTED_DIR}")
        except ImportError:
            logger.warning("Albumentations not installed — skipping augmentation")
    else:
        logger.info("Augmentation skipped (--no-augment)")

    # 8. Dataset statistics
    _print_stats(rl_qd_rows)


def _print_stats(rows: List[Dict]) -> None:
    """Print dataset composition statistics."""
    from collections import Counter
    types = Counter(r["object_type"] for r in rows)
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total labelled objects: {len(rows)}")
    for cls_name, count in types.most_common():
        logger.info(f"  {cls_name}: {count}")
    rl_vals = [r["rl_label"] for r in rows]
    logger.info(f"RL range: [{min(rl_vals):.1f}, {max(rl_vals):.1f}] mcd/m²/lx")
    logger.info(f"RL mean: {np.mean(rl_vals):.1f} ± {np.std(rl_vals):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HighwayRetroAI Dataset Preparation")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, don't write")
    parser.add_argument("--no-augment", action="store_true", help="Skip augmentation")
    args = parser.parse_args()
    main(dry_run=args.dry_run, no_augment=args.no_augment)
