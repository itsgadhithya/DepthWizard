"""Unit tests for image reader and metadata extractor with provenance tracking."""

import pytest
import numpy as np

from backend.ingestion.reader import ImageReader
from backend.ingestion.metadata_extractor import MetadataExtractor
from backend.models.metadata import MetadataFieldStatus


def test_image_reader_plain_jpeg(synthetic_jpeg_bytes):
    """Test reading plain JPEG into uint8 RGB numpy array."""
    img_np, format_name, info = ImageReader.read_image(synthetic_jpeg_bytes)
    assert isinstance(img_np, np.ndarray)
    assert img_np.dtype == np.uint8
    assert img_np.shape == (128, 128, 3)
    assert format_name == "JPEG"
    assert info["width"] == 128
    assert info["height"] == 128
    assert not info["is_geotiff"]


def test_image_reader_geotiff(synthetic_geotiff_bytes):
    """Test reading GeoTIFF into normalized RGB array and detecting GeoTIFF format."""
    img_np, format_name, info = ImageReader.read_image(synthetic_geotiff_bytes)
    assert isinstance(img_np, np.ndarray)
    assert img_np.shape == (64, 64, 3)
    assert format_name == "GeoTIFF"
    assert info["is_geotiff"]


def test_image_reader_invalid_bytes():
    """Test error handling for corrupted image bytes."""
    with pytest.raises(Exception):
        ImageReader.read_image(b"invalid image bytes")


def test_metadata_extractor_plain_image(synthetic_jpeg_bytes):
    """Verify that plain images without EXIF have explicit ABSENT status and no fabricated values."""
    meta = MetadataExtractor.extract_metadata(synthetic_jpeg_bytes, filename="plain.jpg")
    assert meta.filename == "plain.jpg"
    assert meta.width == 128
    assert meta.height == 128
    assert not meta.has_exif
    assert not meta.has_gps
    assert not meta.has_geotiff
    assert meta.exif is None
    assert meta.gps is None

    # Check field provenance
    assert meta.provenance["dimensions"].status == MetadataFieldStatus.PRESENT
    assert meta.provenance["exif"].status == MetadataFieldStatus.ABSENT


def test_metadata_extractor_with_exif_and_gps(synthetic_jpeg_with_exif):
    """Verify that EXIF and GPS tags are accurately parsed without fabrication."""
    meta = MetadataExtractor.extract_metadata(synthetic_jpeg_with_exif, filename="drone.jpg")
    assert meta.has_exif
    assert meta.has_gps
    assert meta.exif is not None
    assert meta.exif.make == "DJI"
    assert meta.exif.model == "FC6310"
    assert meta.exif.focal_length_mm == 8.8
    assert meta.exif.focal_length_35mm_equiv == 24.0

    assert meta.gps is not None
    assert pytest.approx(meta.gps.latitude, rel=1e-4) == 37.774967
    assert pytest.approx(meta.gps.longitude, rel=1e-4) == -122.4194
    assert meta.gps.altitude == 150.0

    # Provenance check
    assert meta.provenance["camera_make"].status == MetadataFieldStatus.PRESENT
    assert meta.provenance["gps_lat_lon"].status == MetadataFieldStatus.PRESENT
    assert meta.provenance["gps_altitude"].status == MetadataFieldStatus.PRESENT


def test_metadata_extractor_geotiff(synthetic_geotiff_bytes):
    """Verify GeoTIFF CRS, transform, and bounding box extraction."""
    meta = MetadataExtractor.extract_metadata(synthetic_geotiff_bytes, filename="elevation.tif")
    assert meta.has_geotiff
    assert meta.geotiff is not None
    assert "32643" in meta.geotiff.crs
    assert meta.geotiff.transform is not None
    assert len(meta.geotiff.transform) == 6
    assert meta.provenance["geotiff_crs"].status == MetadataFieldStatus.PRESENT
