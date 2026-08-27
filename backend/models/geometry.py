"""Data models for 3D geometry and point clouds."""

from enum import Enum
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CoordinateFrame(str, Enum):
    """Reference coordinate frame for 3D points."""
    CAMERA = "camera"  # Optical camera frame: +X right, +Y down, +Z forward
    CAMERA_OPTICAL = "camera"
    LOCAL_ENU = "local_enu"  # Local Tangential Plane: +X East, +Y North, +Z Up
    PROJECTED_CRS = "projected_crs"  # Map projected coordinates (e.g. UTM Easting, Northing, Elevation)
    GEOGRAPHIC = "geographic"  # Longitude, Latitude, Ellipsoidal Height


class Units(str, Enum):
    """Physical units of coordinates."""
    METERS = "meters"
    RELATIVE = "relative_units"
    NORMALIZED = "normalized"


class PointCloud3D(BaseModel):
    """Dense 3D point cloud representation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: np.ndarray = Field(..., description="(N, 3) float32 array of coordinates [X, Y, Z]")
    colors: Optional[np.ndarray] = Field(default=None, description="Optional (N, 3) uint8 array of RGB color values")
    normals: Optional[np.ndarray] = Field(default=None, description="Optional (N, 3) float32 surface normals")
    coordinate_frame: CoordinateFrame = CoordinateFrame.CAMERA
    units: Units = Units.RELATIVE
    is_metric: bool = False
    crs: Optional[str] = None
    point_count: int = 0
    bounds_min: List[float] = Field(default_factory=list)
    bounds_max: List[float] = Field(default_factory=list)

    def compute_bounds(self) -> None:
        """Compute spatial bounding box [min, max]."""
        if self.points is not None and len(self.points) > 0:
            self.bounds_min = [float(x) for x in np.nanmin(self.points, axis=0)]
            self.bounds_max = [float(x) for x in np.nanmax(self.points, axis=0)]
            self.point_count = len(self.points)
        else:
            self.bounds_min = [0.0, 0.0, 0.0]
            self.bounds_max = [0.0, 0.0, 0.0]
            self.point_count = 0
