"""Test fixtures and synthetic data generators for DepthWizard test suite."""

import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image, ExifTags
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.calibration import GCPPoint, DistanceMeasurement, CalibrationReference, CalibrationMethod


@pytest.fixture
def test_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def synthetic_rgb_image() -> np.ndarray:
    """Create a synthetic (128, 128, 3) RGB uint8 image with geometric patterns."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :64] = [200, 50, 50]    # Red square (e.g. building roof)
    arr[:64, 64:] = [50, 200, 50]    # Green square (trees)
    arr[64:, :64] = [50, 50, 200]    # Blue square (water)
    arr[64:, 64:] = [220, 220, 100]  # Yellow square (road/field)
    return arr


@pytest.fixture
def synthetic_jpeg_bytes(synthetic_rgb_image) -> bytes:
    """Encode synthetic RGB image to plain JPEG bytes (no EXIF)."""
    pil_img = Image.fromarray(synthetic_rgb_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_jpeg_with_exif(synthetic_rgb_image) -> bytes:
    """Encode synthetic RGB image with simulated drone EXIF and GPS tags."""
    pil_img = Image.fromarray(synthetic_rgb_image)
    exif = pil_img.getexif()

    # Make & Model
    exif[ExifTags.Base.Make] = "DJI"
    exif[ExifTags.Base.Model] = "FC6310"  # Phantom 4 Pro (1-inch sensor)
    exif[ExifTags.Base.FocalLength] = 8.8  # mm
    exif[ExifTags.Base.FocalLengthIn35mmFilm] = 24  # 35mm equivalent

    # GPS info IFD
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    # Latitude: 37 deg 46 min 29.88 sec N -> 37.774967
    gps_ifd[ExifTags.GPS.GPSLatitude] = (37.0, 46.0, 29.88)
    gps_ifd[ExifTags.GPS.GPSLatitudeRef] = "N"
    # Longitude: 122 deg 25 min 9.84 sec W -> -122.4194
    gps_ifd[ExifTags.GPS.GPSLongitude] = (122.0, 25.0, 9.84)
    gps_ifd[ExifTags.GPS.GPSLongitudeRef] = "W"
    # Altitude: 150.0 meters MSL
    gps_ifd[ExifTags.GPS.GPSAltitude] = 150.0
    gps_ifd[ExifTags.GPS.GPSAltitudeRef] = 0

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


@pytest.fixture
def synthetic_geotiff_bytes() -> bytes:
    """Create a synthetic 1-band GeoTIFF in memory with EPSG:32643 CRS."""
    w, h = 64, 64
    elevation_data = np.linspace(50.0, 75.0, w * h, dtype=np.float32).reshape(h, w)

    # Origin at UTM 500000 E, 4000000 N, resolution 1.0m
    transform = from_origin(500000.0, 4000000.0, 1.0, 1.0)
    crs = CRS.from_epsg(32643)

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation_data, 1)

    return buf.getvalue()


@pytest.fixture
def synthetic_rgb_geotiff_bytes() -> bytes:
    """Create a synthetic 3-band RGB GeoTIFF in memory with EPSG:32633 CRS (e.g. Potsdam orthophoto)."""
    w, h = 64, 64
    r_data = np.full((h, w), 200, dtype=np.uint8)
    g_data = np.full((h, w), 100, dtype=np.uint8)
    b_data = np.full((h, w), 50, dtype=np.uint8)

    # Origin at UTM Zone 33N, resolution 0.05m/pixel (5cm GSD)
    transform = from_origin(379000.0, 5804000.0 + (h * 0.05), 0.05, 0.05)
    crs = CRS.from_epsg(32633)

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(r_data, 1)
        dst.write(g_data, 2)
        dst.write(b_data, 3)

    return buf.getvalue()

