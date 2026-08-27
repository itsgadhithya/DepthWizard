"""Data models for Digital Surface Model (DSM) rasters and geospatial metadata."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from backend.models.geospatial import GeoBounds


class DSMMetadata(BaseModel):
    """Provenance and scientific metadata describing how the DSM was generated."""
    generation_method: str = Field(default="max_surface_elevation", description="Surface extraction algorithm ('max_surface_elevation', 'mean_surface', 'idw')")
    void_filling_applied: bool = Field(default=True, description="Whether void filling/interpolation was performed")
    source_points_count: int = Field(default=0, description="Total 3D input points used for rasterization")
    valid_cells_count: int = Field(default=0, description="Count of grid cells containing valid elevation data")
    nodata_cells_count: int = Field(default=0, description="Count of grid cells marked as nodata")
    valid_coverage_percent: float = Field(default=0.0, description="Percentage of valid data cells over the bounding grid")
    min_elevation_m: float = Field(default=0.0, description="Minimum elevation value in meters")
    max_elevation_m: float = Field(default=0.0, description="Maximum elevation value in meters")
    mean_elevation_m: float = Field(default=0.0, description="Mean elevation value in meters")
    std_elevation_m: float = Field(default=0.0, description="Standard deviation of elevation values in meters")
    horizontal_crs: str = Field(..., description="Projected coordinate system code (e.g. 'EPSG:32643')")
    vertical_datum: str = Field(default="WGS84 MSL / Local Reference Datum", description="Vertical reference datum")
    elevation_units: str = Field(default="meters", description="Vertical physical units")
    resolution_m: float = Field(default=0.5, description="Grid resolution in meters per pixel")
    software: str = Field(default="DepthWizard Depth & Metric Geometry Engine", description="Generating software name")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO8601 creation timestamp")


class DSMResult(BaseModel):
    """Rasterized Digital Surface Model representing surface elevation above a vertical datum."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: np.ndarray = Field(..., description="2D float32 raster grid of surface elevations in meters")
    width: int
    height: int
    crs: str  # e.g., "EPSG:32643"
    transform: List[float]  # 6-element GDAL affine transform [c, a, b, f, d, e]
    bounds: GeoBounds
    resolution_m: float
    min_elevation_m: float
    max_elevation_m: float
    mean_elevation_m: float
    std_elevation_m: float = 0.0
    nodata_value: float = -9999.0
    valid_pixel_count: int = 0
    valid_coverage_percent: float = 0.0
    units: str = "meters"
    metadata: Optional[DSMMetadata] = None
