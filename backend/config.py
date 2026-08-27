"""Configuration and runtime settings for DepthWizard engine."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class EngineSettings(BaseModel):
    """Engine runtime settings and path configuration."""

    # Project directories
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    artifacts_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models_weights")

    # Depth model defaults
    default_model_name: str = "depth_anything_v2_vits"
    model_encoder: str = "vits"  # 'vits', 'vitb', 'vitl', 'vitg'
    default_device: str = "auto"  # 'auto', 'cuda', 'cpu', 'mps'
    input_size: int = 518  # Optimal input size for DepthAnything V2

    # Processing limits & defaults
    max_image_dimension: Optional[int] = None  # None = unconstrained maximum resolution (hardware capacity)
    default_dsm_resolution: float = 0.5  # meters per pixel
    default_nodata_value: float = -9999.0
    point_cloud_subsample_limit: int = 250_000  # Max points for JSON web preview

    # Geospatial defaults
    default_fallback_crs: str = "EPSG:4326"

    # API Settings
    api_title: str = "DepthWizard Depth & Metric Geometry Engine API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Metric Calibration & Provisional Fallback
    enable_provisional_fallback: bool = True
    provisional_metric_scale: float = 10.0

    model_config = {
        "arbitrary_types_allowed": True
    }


# Global settings singleton instance
settings = EngineSettings()
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)

PROVISIONAL_METRIC_SCALE: float = settings.provisional_metric_scale
