"""Unit tests for camera modeling and intrinsic calculation."""

import pytest
import numpy as np

from backend.camera.sensor_db import lookup_sensor_dimensions
from backend.camera.model import CameraModelBuilder
from backend.models.metadata import ImageMetadata, ExifMetadata, GPSMetadata


def test_sensor_db_lookup():
    """Verify known camera sensors are found and unknown models return None."""
    dims = lookup_sensor_dimensions("DJI", "FC6310")
    assert dims is not None
    assert dims == (13.2, 8.8)

    dims_sony = lookup_sensor_dimensions("Sony", "ILCE-7RM4")
    assert dims_sony is not None
    assert dims_sony[0] > 35.0  # Full frame

    dims_none = lookup_sensor_dimensions("UnknownBrand", "ModelXYZ999")
    assert dims_none is None


def test_camera_model_user_override():
    """Verify explicit user overrides result in calibrated status and confidence 1.0."""
    meta = ImageMetadata(filename="test.jpg", format="JPEG", width=1920, height=1080)
    camera = CameraModelBuilder.build_camera_model(meta, overrides={"fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0})

    assert not camera.intrinsics.is_estimated
    assert camera.intrinsics.fx == 1500.0
    assert camera.intrinsics.fy == 1500.0
    assert camera.intrinsics.confidence == 1.0
    assert camera.intrinsics.estimation_method == "calibrated_user_input"

    # Verify K matrix
    k = camera.intrinsics.get_k_matrix()
    assert k.shape == (3, 3)
    assert k[0, 0] == 1500.0
    assert k[0, 2] == 960.0
    assert k[1, 1] == 1500.0
    assert k[1, 2] == 540.0


def test_camera_model_exif_sensor_lookup():
    """Verify EXIF focal length + Sensor DB match calculates pixel focal length."""
    exif = ExifMetadata(make="DJI", model="FC6310", focal_length_mm=8.8)
    meta = ImageMetadata(filename="drone.jpg", format="JPEG", width=1320, height=880, has_exif=True, exif=exif)

    camera = CameraModelBuilder.build_camera_model(meta)
    # sensor width is 13.2mm, image width is 1320px -> 100 px/mm * 8.8mm = 880px
    assert not camera.intrinsics.is_estimated
    assert pytest.approx(camera.intrinsics.fx, rel=1e-3) == 880.0
    assert camera.intrinsics.estimation_method == "exif_sensor_lookup"
    assert camera.intrinsics.confidence >= 0.85


def test_camera_model_heuristic_fallback():
    """Verify missing metadata gracefully falls back to estimated FOV with explicit flag."""
    meta = ImageMetadata(filename="plain.jpg", format="JPEG", width=1000, height=800)
    camera = CameraModelBuilder.build_camera_model(meta)

    assert camera.intrinsics.is_estimated
    assert camera.intrinsics.confidence < 0.5
    assert camera.intrinsics.estimation_method == "heuristic_fov"
    assert camera.intrinsics.cx == 500.0
    assert camera.intrinsics.cy == 400.0


def test_camera_model_gps_only_no_synthetic_orientation():
    """Verify that GPS-only metadata provides position but NEVER fabricates synthetic yaw/pitch/roll."""
    gps = GPSMetadata(latitude=37.7749, longitude=-122.4194, altitude=150.0)
    meta = ImageMetadata(
        filename="drone_gps.jpg",
        format="JPEG",
        width=1000,
        height=800,
        has_gps=True,
        gps=gps,
    )
    camera = CameraModelBuilder.build_camera_model(meta)

    # Position must be available
    assert camera.has_position is True
    assert camera.extrinsics is not None
    assert camera.extrinsics.is_position_available is True
    assert camera.extrinsics.latitude == 37.7749
    assert camera.extrinsics.longitude == -122.4194
    assert camera.extrinsics.altitude_m == 150.0
    assert camera.extrinsics.position_x is not None
    assert camera.extrinsics.position_y is not None
    assert camera.extrinsics.projected_crs is not None  # Converted to projected UTM meters

    # Orientation must NOT be fabricated
    assert camera.has_orientation is False
    assert camera.has_complete_pose is False
    assert camera.extrinsics.is_orientation_available is False
    assert camera.extrinsics.is_complete_pose_available is False
    assert camera.extrinsics.yaw_deg is None
    assert camera.extrinsics.pitch_deg is None
    assert camera.extrinsics.roll_deg is None

    # Provenance check
    assert camera.provenance["camera_position"].status.value == "present"
    assert camera.provenance["camera_orientation"].status.value == "absent"


def test_camera_model_explicit_pose_override():
    """Verify that explicit orientation overrides set complete camera pose."""
    gps = GPSMetadata(latitude=37.7749, longitude=-122.4194, altitude=150.0)
    meta = ImageMetadata(
        filename="drone_gps.jpg",
        format="JPEG",
        width=1000,
        height=800,
        has_gps=True,
        gps=gps,
    )
    camera = CameraModelBuilder.build_camera_model(
        meta,
        overrides={"yaw_deg": 45.0, "pitch_deg": -60.0, "roll_deg": 2.0},
    )

    assert camera.has_position is True
    assert camera.has_orientation is True
    assert camera.has_complete_pose is True
    assert camera.extrinsics.yaw_deg == 45.0
    assert camera.extrinsics.pitch_deg == -60.0
    assert camera.extrinsics.roll_deg == 2.0
    assert camera.provenance["camera_orientation"].status.value == "present"

