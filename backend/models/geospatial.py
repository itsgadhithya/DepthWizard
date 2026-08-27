"""Data models for geospatial coordinate systems, transforms, and bounds."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CRSInfo(BaseModel):
    """Coordinate Reference System details."""
    crs_string: str  # e.g., "EPSG:32643", "EPSG:4326"
    is_projected: bool = True
    is_geographic: bool = False
    datum_name: Optional[str] = None
    units: str = "meters"
    utm_zone: Optional[int] = None
    utm_hemisphere: Optional[str] = None  # "N" or "S"


class GeoBounds(BaseModel):
    """Geospatial bounding box in specified CRS."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_z: Optional[float] = None
    max_z: Optional[float] = None
    crs: str = "EPSG:4326"


class GeoTransform(BaseModel):
    """Affine spatial transform for raster data [a, b, c, d, e, f]."""
    affine: List[float] = Field(..., min_length=6, max_length=6)
    resolution_x: float
    resolution_y: float
    origin_x: float
    origin_y: float
