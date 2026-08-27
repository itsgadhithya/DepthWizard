"""Typed camera model representing explicit intrinsics, extrinsics, and estimation flags."""

from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from backend.models.metadata import FieldProvenance


class CameraIntrinsics(BaseModel):
    """Pinhole camera intrinsic parameters and lens distortion."""
    fx: float  # Focal length in pixels (X axis)
    fy: float  # Focal length in pixels (Y axis)
    cx: float  # Principal point X (pixels)
    cy: float  # Principal point Y (pixels)
    k1: float = 0.0  # Radial distortion k1
    k2: float = 0.0  # Radial distortion k2
    p1: float = 0.0  # Tangential distortion p1
    p2: float = 0.0  # Tangential distortion p2
    k3: float = 0.0  # Radial distortion k3
    width: int
    height: int
    is_estimated: bool = True
    estimation_method: str = "heuristic_fov"  # "calibrated", "exif_sensor", "exif_35mm", "heuristic_fov"
    confidence: float = 0.5

    def get_k_matrix(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def get_inv_k_matrix(self) -> np.ndarray:
        """Return 3x3 inverse intrinsic matrix K^-1."""
        return np.linalg.inv(self.get_k_matrix())


class CameraExtrinsics(BaseModel):
    """Camera spatial position and 3D orientation in world/projected coordinates."""
    # Geographic coordinates (WGS84)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None  # Altitude above datum (meters)

    # Projected metric coordinates (e.g. UTM Easting, Northing, Elevation in meters)
    position_x: Optional[float] = None  # Projected Easting (meters)
    position_y: Optional[float] = None  # Projected Northing (meters)
    position_z: Optional[float] = None  # Elevation (meters above datum)
    projected_crs: Optional[str] = None  # e.g., "EPSG:32643"

    # Orientation (Gimbal / Camera attitude)
    yaw_deg: Optional[float] = None  # Heading (degrees clockwise from North)
    pitch_deg: Optional[float] = None  # Pitch (-90 = Nadir pointing straight down, 0 = Horizontal)
    roll_deg: Optional[float] = None  # Roll (degrees)

    # Availability & Estimation flags
    is_position_available: bool = False
    is_orientation_available: bool = False
    is_complete_pose_available: bool = False
    is_estimated: bool = True
    coordinate_frame: str = "WGS84_ENU"


class CameraModel(BaseModel):
    """Complete camera model combining intrinsics and extrinsics."""
    intrinsics: CameraIntrinsics
    extrinsics: Optional[CameraExtrinsics] = None
    has_position: bool = False
    has_orientation: bool = False
    has_complete_pose: bool = False
    provenance: Dict[str, FieldProvenance] = Field(default_factory=dict)
