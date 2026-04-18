"""
rl_calculator.py — Retroreflectivity (RL) and Qd prediction + IoT corrections.

Loads the trained EfficientNet-based regression model and applies geometry,
tilt, and wet-surface corrections from IoT sensor data.

Usage:
    from src.retroreflectivity.rl_calculator import RLCalculator
    calc = RLCalculator("models/rl_regressor/efficientnet_rl.pt")
    result = calc.predict(roi_tensor, {"distance_cm": 300, ...})
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from config import (
    RL_MODEL_PATH,
    RL_TRT_ENGINE,
    USE_TRT,
    WET_CORRECTION_DEFAULT,
    WET_CORRECTION_TABLE,
    VEHICLE_HEIGHT_CM,
    TrainConfig,
)
from src.retroreflectivity.geometry import calculate_angles, geometry_correction_factor
from src.utils.logger import logger


class _RLRegressorModel(nn.Module):
    """EfficientNet-B0 + scalar fusion head for RL/Qd regression.

    Architecture (matches train_rl_model.py):
        Image → EfficientNet-B0 → 1280-dim
        Scalar [5] →
        Concat → FC(512) → ReLU → Dropout → FC(2) → [RL, Qd]
    """

    def __init__(self, scalar_dim: int = 5, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet

            self.backbone = EfficientNet.from_name("efficientnet-b0")
            backbone_out = 1280
        except ImportError:
            # Fallback: simple CNN
            logger.warning("efficientnet_pytorch not found — using fallback CNN backbone")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            backbone_out = 64

        self.head = nn.Sequential(
            nn.Linear(backbone_out + scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [RL, Qd]
        )

    def forward(self, image: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        image : torch.Tensor
            ``(B, 3, 224, 224)`` normalised image batch.
        scalars : torch.Tensor
            ``(B, 5)`` scalar features.

        Returns
        -------
        torch.Tensor
            ``(B, 2)`` predictions ``[RL, Qd]``.
        """
        if hasattr(self.backbone, "extract_features"):
            # EfficientNet path
            feats = self.backbone.extract_features(image)
            feats = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        else:
            feats = self.backbone(image)
        fused = torch.cat([feats, scalars], dim=1)
        return self.head(fused)


class RLCalculator:
    """Predict RL/Qd from a cropped ROI and apply IoT corrections.

    Parameters
    ----------
    model_path : str | Path | None
        Path to saved ``.pt`` weights.  If ``None``, default from config.
    use_trt : bool
        Load a TensorRT engine instead (future).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_trt: bool = USE_TRT,
    ) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[_RLRegressorModel] = None

        path = Path(model_path) if model_path else RL_MODEL_PATH
        if path.exists():
            self._model = _RLRegressorModel()
            state = torch.load(str(path), map_location=self._device, weights_only=False)
            self._model.load_state_dict(state)
            self._model.to(self._device).eval()
            logger.info(f"RL regressor loaded from {path}")
        else:
            logger.warning(f"RL model not found at {path} — predictions will be simulated")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(
        self,
        roi_tensor: torch.Tensor,
        scalar_inputs: Dict[str, float],
    ) -> Dict[str, float]:
        """Predict RL and Qd, then apply sensor-based corrections.

        Parameters
        ----------
        roi_tensor : torch.Tensor
            ``(1, 3, 224, 224)`` normalised ROI tensor.
        scalar_inputs : Dict[str, float]
            Keys: ``distance_cm``, ``tilt_deg``, ``temperature_c``,
            ``humidity_pct``, ``is_night`` (0 or 1).

        Returns
        -------
        Dict[str, float]
            ``{"rl_raw": float, "rl_corrected": float, "qd": float}``
        """
        if self._model is not None:
            scalars = torch.tensor(
                [
                    scalar_inputs.get("distance_cm", 300.0),
                    scalar_inputs.get("tilt_deg", 0.0),
                    scalar_inputs.get("temperature_c", 25.0),
                    scalar_inputs.get("humidity_pct", 50.0),
                    scalar_inputs.get("is_night", 0.0),
                ],
                dtype=torch.float32,
            ).unsqueeze(0).to(self._device)

            roi_tensor = roi_tensor.to(self._device)
            with torch.no_grad():
                preds = self._model(roi_tensor, scalars)
            rl_raw = max(0.0, preds[0, 0].item())
            qd = max(0.0, min(1.0, preds[0, 1].item()))
        else:
            # Simulated prediction (when model is missing)
            rl_raw = float(np.random.normal(300, 80))
            rl_raw = max(10.0, rl_raw)
            qd = float(np.clip(np.random.normal(0.5, 0.15), 0.05, 0.95))

        rl_corrected = self.apply_corrections(rl_raw, scalar_inputs)
        return {"rl_raw": round(rl_raw, 2), "rl_corrected": round(rl_corrected, 2), "qd": round(qd, 4)}

    def apply_corrections(self, rl_raw: float, sensor_data: Dict[str, float]) -> float:
        """Apply geometry, tilt, and wet-surface corrections to raw RL.

        Parameters
        ----------
        rl_raw : float
            Raw RL prediction (mcd/m²/lx).
        sensor_data : Dict[str, float]
            Sensor snapshot with ``distance_cm``, ``tilt_deg``, ``humidity_pct``.

        Returns
        -------
        float
            Corrected RL value.
        """
        rl = rl_raw

        # 1. Geometry correction
        distance = sensor_data.get("distance_cm", 300.0)
        if distance > 0:
            angles = calculate_angles(distance, VEHICLE_HEIGHT_CM)
            geo_factor = geometry_correction_factor(angles["observation_angle_deg"])
            rl *= geo_factor

        # 2. Tilt correction
        tilt = abs(sensor_data.get("tilt_deg", 0.0))
        if tilt > 0:
            rl *= math.cos(math.radians(tilt))

        # 3. Wet-surface (humidity) correction
        humidity = sensor_data.get("humidity_pct", 50.0)
        wet_factor = WET_CORRECTION_DEFAULT
        for threshold, factor in WET_CORRECTION_TABLE:
            if humidity > threshold:
                wet_factor = factor
                break
        rl *= wet_factor

        return max(0.0, rl)
