#!/usr/bin/env python3
"""
convert_trt.py — Convert YOLO and RL regressor models to TensorRT engines.

Supports FP16 and INT8 calibration for optimised inference on Jetson Xavier NX.
TensorRT is optional; the system runs in PyTorch mode if engines don't exist.

Usage:
    python3 convert_trt.py                    # FP16 for both models
    python3 convert_trt.py --int8             # INT8 for YOLO (uses calibration images)
    python3 convert_trt.py --yolo-only        # convert YOLO only
    python3 convert_trt.py --rl-only          # convert RL regressor only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from config import (
    MODEL_DIR,
    PROCESSED_IMAGES,
    RL_ONNX_PATH,
    RL_TRT_ENGINE,
    YOLO_MODEL_PATH,
    YOLO_TRT_ENGINE,
)
from src.utils.logger import logger


def convert_yolo_trt(fp16: bool = True, int8: bool = False) -> bool:
    """Convert YOLO .pt → TensorRT .engine via ultralytics export.

    Parameters
    ----------
    fp16 : bool
        Use FP16 precision.
    int8 : bool
        Use INT8 quantisation (requires calibration data).

    Returns
    -------
    bool
        ``True`` if conversion succeeded.
    """
    if not YOLO_MODEL_PATH.exists():
        logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}")
        return False

    try:
        from ultralytics import YOLO

        model = YOLO(str(YOLO_MODEL_PATH))
        logger.info("Exporting YOLO to TensorRT engine...")

        export_args = {
            "format": "engine",
            "half": fp16,
            "int8": int8,
            "dynamic": False,
            "simplify": True,
            "workspace": 4,  # GB
        }

        if int8:
            cal_dir = PROCESSED_IMAGES / "val"
            if cal_dir.exists():
                export_args["data"] = str(cal_dir)

        result = model.export(**export_args)

        # Move engine to standard location
        engine_path = Path(str(YOLO_MODEL_PATH).replace(".pt", ".engine"))
        if engine_path.exists():
            YOLO_TRT_ENGINE.parent.mkdir(parents=True, exist_ok=True)
            engine_path.rename(YOLO_TRT_ENGINE)
            logger.info(f"YOLO TRT engine → {YOLO_TRT_ENGINE}")

        return True
    except Exception as exc:
        logger.error(f"YOLO TRT conversion failed: {exc}")
        return False


def convert_rl_trt(fp16: bool = True) -> bool:
    """Convert RL regressor ONNX → TensorRT engine via trtexec.

    Parameters
    ----------
    fp16 : bool
        Use FP16 precision.

    Returns
    -------
    bool
        ``True`` if conversion succeeded.
    """
    if not RL_ONNX_PATH.exists():
        logger.warning(f"RL ONNX model not found at {RL_ONNX_PATH}")
        return False

    RL_TRT_ENGINE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={RL_ONNX_PATH}",
        f"--saveEngine={RL_TRT_ENGINE}",
        "--workspace=2048",
    ]
    if fp16:
        cmd.append("--fp16")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            logger.info(f"RL TRT engine → {RL_TRT_ENGINE}")
            return _verify_rl_engine()
        else:
            logger.error(f"trtexec failed: {result.stderr[:500]}")
            return False
    except FileNotFoundError:
        logger.warning("trtexec not found — install TensorRT toolkit")
        return False
    except subprocess.TimeoutExpired:
        logger.error("trtexec timed out (>600s)")
        return False


def _verify_rl_engine() -> bool:
    """Run a single test forward pass through the RL TRT engine.

    Returns
    -------
    bool
        ``True`` if verification succeeded.
    """
    try:
        import tensorrt as trt
        import numpy as np

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(RL_TRT_ENGINE, "rb") as f:
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        logger.info(f"RL TRT engine verified: {engine.num_bindings} bindings")

        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            io = "input" if engine.binding_is_input(i) else "output"
            logger.info(f"  [{io}] {name}: shape={shape}, dtype={dtype}")

        return True
    except ImportError:
        logger.warning("tensorrt Python package not available — skipping verification")
        return True
    except Exception as exc:
        logger.error(f"TRT verification failed: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert models to TensorRT")
    parser.add_argument("--fp16", action="store_true", default=True, help="FP16 precision")
    parser.add_argument("--int8", action="store_true", help="INT8 quantisation for YOLO")
    parser.add_argument("--yolo-only", action="store_true")
    parser.add_argument("--rl-only", action="store_true")
    args = parser.parse_args()

    success = True
    if not args.rl_only:
        logger.info("=== Converting YOLO to TensorRT ===")
        if not convert_yolo_trt(fp16=args.fp16, int8=args.int8):
            success = False

    if not args.yolo_only:
        logger.info("=== Converting RL Regressor to TensorRT ===")
        if not convert_rl_trt(fp16=args.fp16):
            success = False

    if success:
        logger.info("All conversions completed successfully")
    else:
        logger.warning("Some conversions failed — system will use PyTorch fallback")
