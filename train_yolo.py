#!/usr/bin/env python3
"""
train_yolo.py — Fine-tune YOLOv8 on the unified road-marking + signs dataset.

Usage:
    python3 train_yolo.py --epochs 50 --batch 16 --img-size 640
    python3 train_yolo.py --fast           # quick 5-epoch test
    python3 train_yolo.py --resume         # resume from last checkpoint
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import (
    DATASET_YAML,
    MODEL_DIR,
    YOLO_MODEL_PATH,
    PROJECT_ROOT,
    TrainConfig,
)
from src.utils.logger import logger


def train(
    epochs: int = 50,
    batch: int = 16,
    img_size: int = 640,
    base_model: str = "yolov8m.pt",
    resume: bool = False,
) -> Path:
    """Fine-tune YOLOv8 on the processed dataset.

    Parameters
    ----------
    epochs : int
        Training epochs.
    batch : int
        Batch size.
    img_size : int
        Input image size (square).
    base_model : str
        Pretrained YOLO checkpoint to start from.
    resume : bool
        Resume from the last interrupted run.

    Returns
    -------
    Path
        Path to the best saved model weights.
    """
    from ultralytics import YOLO

    yaml_path = DATASET_YAML
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {yaml_path}. Run dataset_preparation.py first."
        )

    logger.info(f"Loading base model: {base_model}")
    model = YOLO(base_model)

    logger.info(
        f"Starting training: epochs={epochs}, batch={batch}, "
        f"img={img_size}, data={yaml_path}"
    )

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=str(PROJECT_ROOT / "runs" / "yolo"),
        name="combined",
        exist_ok=True,
        resume=resume,
        patience=15,
        save=True,
        save_period=5,
        verbose=True,
        device="0" if _cuda_available() else "cpu",
        workers=4,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # Copy best weights to standard location
    best_pt = PROJECT_ROOT / "runs" / "yolo" / "combined" / "weights" / "best.pt"
    if best_pt.exists():
        YOLO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, YOLO_MODEL_PATH)
        logger.info(f"Best model saved → {YOLO_MODEL_PATH}")
    else:
        logger.warning(f"best.pt not found at {best_pt}")

    # Log final metrics
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info(f"Final mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"Final mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

    return YOLO_MODEL_PATH


def _cuda_available() -> bool:
    """Check CUDA availability."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on road elements")
    cfg = TrainConfig()
    parser.add_argument("--epochs", type=int, default=cfg.yolo_epochs)
    parser.add_argument("--batch", type=int, default=cfg.yolo_batch)
    parser.add_argument("--img-size", type=int, default=cfg.yolo_img_size)
    parser.add_argument("--base-model", type=str, default=cfg.yolo_base_model)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Quick 5-epoch test run")
    args = parser.parse_args()

    if args.fast:
        args.epochs = 5
        args.batch = 8

    train(
        epochs=args.epochs,
        batch=args.batch,
        img_size=args.img_size,
        base_model=args.base_model,
        resume=args.resume,
    )
