"""Data models for pipeline states, summaries, and output artifact registry."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from backend.models.metadata import ImageMetadata
from backend.models.camera import CameraModel
from backend.models.calibration import CalibrationResult
from backend.models.validation import ValidationReport


class PipelineState(str, Enum):
    """The achieved execution state of the single-image processing pipeline."""
    STATE_A = "STATE_A"  # Relative depth map only (no camera or metric scale)
    STATE_B = "STATE_B"  # Relative depth + 3D geometry in camera frame (unscaled)
    STATE_C = "STATE_C"  # Calibrated metric depth + metric 3D point cloud (meters)
    STATE_D = "STATE_D"  # Metric 3D geometry + Georeferenced DSM (GeoTIFF)


class ArtifactInfo(BaseModel):
    """Reference to a saved computational or visualization artifact."""
    name: str
    filename: str
    artifact_type: str  # e.g., "raw_depth_npy", "visual_png", "point_cloud_ply", "dsm_tif"
    file_path: str
    download_url: str
    size_bytes: int = 0
    is_computational: bool = False
    is_visualization: bool = False


class ProcessingSummary(BaseModel):
    """High-level summary of the processing pipeline results and degradation status."""
    request_id: str
    state: PipelineState
    relative_depth_available: bool = False
    camera_model_available: bool = False
    camera_position_available: bool = False
    camera_orientation_available: bool = False
    complete_camera_pose_available: bool = False
    metric_depth_available: bool = False
    georeferencing_available: bool = False
    dsm_available: bool = False
    validation_available: bool = False
    model_name: str = ""
    device_used: str = ""
    total_time_ms: float = 0.0
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    messages: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Optional[ImageMetadata] = None
    camera: Optional[CameraModel] = None
    calibration: Optional[CalibrationResult] = None
    validation: Optional[ValidationReport] = None
    artifacts: Dict[str, ArtifactInfo] = Field(default_factory=dict)
