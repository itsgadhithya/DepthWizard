"""Typed metadata models with explicit provenance and field status tracking."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MetadataFieldStatus(str, Enum):
    """Status indicating how a metadata field was derived."""
    PRESENT = "present"
    ABSENT = "absent"
    ESTIMATED = "estimated"
    INFERRED = "inferred"


class FieldProvenance(BaseModel):
    """Provenance and origin tracking for individual metadata parameters."""
    field_name: str
    status: MetadataFieldStatus
    source: str = "none"  # e.g., "exif", "geotiff", "sensor_database", "heuristic", "user_input"
    confidence: float = 1.0  # 0.0 to 1.0
    notes: Optional[str] = None


class GPSMetadata(BaseModel):
    """Geographic position information extracted from image EXIF or tags."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None  # Altitude in meters
    altitude_ref: Optional[int] = None  # 0 = Above sea level, 1 = Below sea level
    dop: Optional[float] = None  # Dilution of Precision if available


class ExifMetadata(BaseModel):
    """Camera and capture parameters extracted from standard EXIF tags."""
    make: Optional[str] = None
    model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    focal_length_35mm_equiv: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    iso: Optional[int] = None
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    orientation: int = 1
    timestamp: Optional[str] = None


class GeoTIFFMetadata(BaseModel):
    """Geospatial raster metadata extracted from GeoTIFF tags."""
    crs: Optional[str] = None  # e.g., "EPSG:32643" or WKT
    transform: Optional[List[float]] = None  # 6-element affine matrix [a, b, c, d, e, f]
    bounds: Optional[List[float]] = None  # [minx, miny, maxx, maxy]
    resolution: Optional[List[float]] = None  # [res_x, res_y]
    nodata: Optional[float] = None
    driver: Optional[str] = None
    band_count: int = 1
    dtype: Optional[str] = None
    is_dem: bool = False  # True if detected as elevation model (single-band float/int)
    min_elevation_m: Optional[float] = None
    max_elevation_m: Optional[float] = None
    mean_elevation_m: Optional[float] = None
    std_elevation_m: Optional[float] = None


class ImageMetadata(BaseModel):
    """Complete image metadata representation with explicit field provenance."""
    filename: str
    format: str  # "JPEG", "PNG", "TIFF", "GeoTIFF"
    width: int
    height: int
    channels: int = 3
    bit_depth: int = 8
    is_dem: bool = False  # True if image is classified as digital elevation model
    has_exif: bool = False
    has_geotiff: bool = False
    has_gps: bool = False
    gps: Optional[GPSMetadata] = None
    exif: Optional[ExifMetadata] = None
    geotiff: Optional[GeoTIFFMetadata] = None
    provenance: Dict[str, FieldProvenance] = Field(default_factory=dict, exclude=True)

    def record_field(
        self,
        name: str,
        status: MetadataFieldStatus,
        source: str = "none",
        confidence: float = 1.0,
        notes: Optional[str] = None,
    ) -> None:
        """Record provenance for a field."""
        self.provenance[name] = FieldProvenance(
            field_name=name,
            status=status,
            source=source,
            confidence=confidence,
            notes=notes,
        )
