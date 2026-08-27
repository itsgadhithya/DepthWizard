"""Data models for metric calibration inputs and results."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CalibrationMethod(str, Enum):
    """Available metric calibration strategies."""
    NONE = "none"
    ALTITUDE_GROUND = "altitude_ground"
    GCP = "gcp"
    KNOWN_DISTANCE = "known_distance"
    KNOWN_OBJECT = "known_object"
    REFERENCE_DEM = "reference_dem"
    MANUAL_SCALE = "manual_scale"
    PROVISIONAL_FIXED_SCALE = "provisional_fixed_scale"


class GCPPoint(BaseModel):
    """Ground Control Point associating pixel coordinates with 3D physical coordinates or depth."""
    pixel_u: float
    pixel_v: float
    world_x: Optional[float] = None  # Projected Easting or Longitude
    world_y: Optional[float] = None  # Projected Northing or Latitude
    elevation_z: Optional[float] = None  # Elevation above datum in meters
    depth_z: Optional[float] = None  # Direct optical depth in meters
    point_id: Optional[str] = None


class DistanceMeasurement(BaseModel):
    """Known real-world distance reference between two image points."""
    point1_pixel: List[float] = Field(..., min_length=2, max_length=2, description="[u1, v1]")
    point2_pixel: List[float] = Field(..., min_length=2, max_length=2, description="[u2, v2]")
    distance_meters: float
    label: Optional[str] = None


class CalibrationReference(BaseModel):
    """User-supplied or metadata-derived reference measurements for metric calibration."""
    method: CalibrationMethod = CalibrationMethod.NONE
    camera_altitude_m: Optional[float] = None  # Camera altitude above datum
    ground_elevation_m: Optional[float] = None  # Ground surface elevation above datum
    flight_height_agl_m: Optional[float] = None  # Direct Above Ground Level (AGL) height
    gcps: Optional[List[GCPPoint]] = None
    distance_references: Optional[List[DistanceMeasurement]] = None
    reference_dem_path: Optional[str] = None
    reference_dem_array: Optional[Any] = None  # (H, W) numpy array of reference elevations in meters
    reference_dem_nodata: Optional[float] = None
    is_orthorectified: bool = False  # True if input image is an orthorectified optical raster
    has_verified_correspondence: bool = False  # True if image-to-DEM spatial correspondence is verified
    image_crs: Optional[str] = None
    image_bounds: Optional[List[float]] = None
    reference_dem_crs: Optional[str] = None
    reference_dem_bounds: Optional[List[float]] = None
    manual_scale_factor: Optional[float] = None
    notes: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True
    }


class CalibrationResult(BaseModel):
    """Result of metric calibration estimation."""
    success: bool
    method: CalibrationMethod
    scale_factor: Optional[float] = None
    shift_offset: Optional[float] = 0.0
    confidence: float = 0.0  # 0.0 (uncalibrated) to 1.0 (highly confident)
    residual_rmse: Optional[float] = None
    reference_count: int = 0
    is_provisional: bool = False
    reason: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
