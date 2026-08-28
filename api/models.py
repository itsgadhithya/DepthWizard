"""
Pydantic Schemas for DepthWizard API Response Contract
Strictly matches the standardized DepthWizard JSON structure.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class TimingsBreakdown(BaseModel):
    ingestion_ms: float = 0.0
    metadata_extraction_ms: float = 0.0
    depth_inference_ms: float = 0.0
    camera_modeling_ms: float = 0.0
    reconstruction_3d_ms: float = Field(default=0.0, serialization_alias="3d_reconstruction_ms")
    metric_calibration_ms: float = 0.0
    geospatial_and_dsm_ms: float = 0.0


class ArtifactInfo(BaseModel):
    filename: str
    artifact_type: str
    download_url: str
    size_bytes: Optional[int] = None
    media_type: Optional[str] = None
    description: Optional[str] = None


class CameraParameters(BaseModel):
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    fov_deg: Optional[float] = None
    estimated: bool = True
    sensor_width_mm: Optional[float] = None
    focal_length_mm: Optional[float] = None


class GeospatialMetadata(BaseModel):
    crs: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    transform: Optional[List[float]] = None
    pixel_spacing_m: Optional[float] = None
    nodata: Optional[Union[float, int]] = None


class ImageMetadata(BaseModel):
    image_width: int
    image_height: int
    channels: int
    format: str
    exif: Dict[str, Any] = Field(default_factory=dict)
    geospatial: Optional[GeospatialMetadata] = None


class CalibrationInfo(BaseModel):
    method: str = "none"
    scale_factor: Optional[float] = None
    reference_type: Optional[str] = None
    reference_value: Optional[float] = None
    calibrated: bool = False


class ValidationMetrics(BaseModel):
    mae: Optional[float] = None
    rmse: Optional[float] = None
    bias: Optional[float] = None
    reference_source: Optional[str] = None


class DepthWizardResponse(BaseModel):
    request_id: str
    state: str  # e.g., "STATE_B", "STATE_C", "STATE_D", "FAILED"

    relative_depth_available: bool = True
    camera_model_available: bool = True
    metric_depth_available: bool = False
    georeferencing_available: bool = False
    dsm_available: bool = False
    validation_available: bool = False

    model_name: str = "DepthAnythingV2"
    device_used: str = "cpu"

    total_time_ms: float = 0.0
    timings_ms: TimingsBreakdown = Field(default_factory=TimingsBreakdown)

    messages: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    metadata: ImageMetadata
    camera: CameraParameters = Field(default_factory=CameraParameters)
    calibration: CalibrationInfo = Field(default_factory=CalibrationInfo)
    validation: Optional[ValidationMetrics] = None

    artifacts: Dict[str, ArtifactInfo] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "healthy"
    api_version: str = "v1"
    depth_model: str = "DepthAnythingV2"
    device: str = "cpu"
    supported_formats: List[str] = ["JPEG", "JPG", "PNG", "TIFF", "TIF", "GeoTIFF"]
