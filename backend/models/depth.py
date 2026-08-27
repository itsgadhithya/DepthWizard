"""Data models for relative and metric depth representations."""

from typing import Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DepthMetadata(BaseModel):
    """Metadata regarding depth estimation processing, backbone, and summary stats."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "depth_anything_v2"
    encoder: str = "vits"
    input_width: int = 0
    input_height: int = 0
    device: str = "cpu"
    inference_time_ms: float = 0.0
    is_metric: bool = False
    units: str = "dimensionless"
    min_depth_m: Optional[float] = None
    max_depth_m: Optional[float] = None
    mean_depth_m: Optional[float] = None


class RelativeDepthMap(BaseModel):
    """Raw relative/model-space depth output from DepthAnything V2.

    CRITICAL INVARIANT: Relative depth is dimensionless and must NEVER be treated as meters.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    array: np.ndarray = Field(..., description="Raw float32 2D array of relative depth")
    width: int
    height: int
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    model_name: str
    model_config_name: str
    device: str
    inference_time_ms: float
    representation: str = "relative_depth"  # "relative_depth" or "relative_disparity"
    is_metric: bool = False
    units: str = "dimensionless"
    visual_bytes: Optional[bytes] = None
    metadata: Optional[DepthMetadata] = None

    @property
    def depth_map(self) -> np.ndarray:
        return self.array


class MetricDepthMap(BaseModel):
    """Calibrated metric depth map in physical meters."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    array: np.ndarray = Field(..., description="Calibrated float32 2D array of metric depth in meters (Z > 0)")
    width: int
    height: int
    min_depth_m: float
    max_depth_m: float
    mean_depth_m: float
    std_depth_m: float
    calibration_method: str
    scale_factor: float
    shift_offset: float = 0.0
    confidence_score: float  # 0.0 to 1.0
    is_metric: bool = True
    is_provisional: bool = False
    units: str = "meters"
    valid_mask: Optional[np.ndarray] = None
    visual_bytes: Optional[bytes] = None
    metadata: Optional[DepthMetadata] = None

    @property
    def depth_meters(self) -> np.ndarray:
        return self.array

