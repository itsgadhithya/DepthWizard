"""Unit tests for metric calibration strategies and graceful degradation."""

import pytest
import numpy as np

from backend.models.depth import RelativeDepthMap
from backend.models.camera import CameraIntrinsics
from backend.models.calibration import (
    CalibrationMethod,
    CalibrationReference,
    GCPPoint,
    DistanceMeasurement,
)
from backend.metric.calibrator import MetricCalibrator


@pytest.fixture
def sample_relative_depth() -> RelativeDepthMap:
    """Fixture creating a controlled relative depth map with median value 2.0."""
    arr = np.full((50, 50), 2.0, dtype=np.float32)
    # Add some variation [1.5 to 2.5]
    arr[:25, :] = 1.5
    arr[25:, :] = 2.5
    return RelativeDepthMap(
        array=arr,
        width=50,
        height=50,
        min_val=1.5,
        max_val=2.5,
        mean_val=2.0,
        std_val=0.5,
        model_name="depth_anything_v2_vits",
        model_config_name="vits",
        device="cpu",
        inference_time_ms=10.0,
    )


@pytest.fixture
def sample_intrinsics() -> CameraIntrinsics:
    """Fixture for sample camera intrinsics."""
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=25.0, cy=25.0, width=50, height=50)


def test_calibration_provisional_fallback_default(sample_relative_depth, sample_intrinsics):
    """Verify that absent metric references produce provisional fixed-scale (10x) with is_provisional=True."""
    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=None,
        intrinsics=sample_intrinsics,
        allow_provisional_fallback=True,
    )
    assert metric_depth is not None
    assert metric_depth.is_metric is True
    assert metric_depth.is_provisional is True
    assert calib_res.success is True
    assert calib_res.method == CalibrationMethod.PROVISIONAL_FIXED_SCALE
    assert calib_res.scale_factor == 10.0
    assert calib_res.shift_offset == 0.0
    assert calib_res.confidence == 0.20
    assert calib_res.reference_count == 0
    assert calib_res.is_provisional is True
    assert "provisional metric scale" in calib_res.reason.lower()


def test_calibration_provisional_fallback_disabled(sample_relative_depth, sample_intrinsics):
    """Verify that when provisional fallback is explicitly disabled, absent references return None."""
    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=None,
        intrinsics=sample_intrinsics,
        allow_provisional_fallback=False,
    )
    assert metric_depth is None
    assert not calib_res.success
    assert calib_res.method == CalibrationMethod.NONE
    assert calib_res.confidence == 0.0
    assert calib_res.is_provisional is False


def test_calibration_relative_depth_none_refusal(sample_intrinsics):
    """Verify that if relative depth is None, provisional metric depth is NOT created."""
    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=None,
        reference=None,
        intrinsics=sample_intrinsics,
        allow_provisional_fallback=True,
    )
    assert metric_depth is None
    assert calib_res.success is False
    assert calib_res.is_provisional is False


def test_altitude_ground_calibration(sample_relative_depth, sample_intrinsics):
    """Verify altitude-based calibration correctly scales depth to AGL meters."""
    ref = CalibrationReference(
        method=CalibrationMethod.ALTITUDE_GROUND,
        camera_altitude_m=120.0,
        ground_elevation_m=20.0,  # AGL = 100.0m
    )
    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=ref,
        intrinsics=sample_intrinsics,
    )
    assert calib_res.success
    assert metric_depth is not None
    assert metric_depth.is_metric
    assert metric_depth.units == "meters"

    # With median relative depth ~2.0 and AGL 100.0m, scale is ~50.0
    assert pytest.approx(calib_res.scale_factor, rel=0.1) == 50.0
    assert metric_depth.min_depth_m >= 50.0
    assert metric_depth.max_depth_m <= 150.0


def test_altitude_ground_missing_ground_elevation_refusal(sample_relative_depth, sample_intrinsics):
    """Verify that GPS altitude alone without ground elevation or direct AGL height refuses calibration."""
    ref = CalibrationReference(
        method=CalibrationMethod.ALTITUDE_GROUND,
        camera_altitude_m=120.0,
        ground_elevation_m=None,  # Missing ground elevation
        flight_height_agl_m=None,
    )
    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=ref,
        intrinsics=sample_intrinsics,
        allow_provisional_fallback=False,
    )
    assert not calib_res.success
    assert metric_depth is None
    assert "ground elevation is missing" in calib_res.reason.lower()



def test_gcp_calibration(sample_relative_depth, sample_intrinsics):
    """Verify least-squares GCP calibration."""
    # Point at (10, 10) has rel depth 1.5, true depth 45.0m -> target scale 30.0
    # Point at (30, 30) has rel depth 2.5, true depth 75.0m -> target scale 30.0
    gcps = [
        GCPPoint(pixel_u=10, pixel_v=10, depth_z=45.0),
        GCPPoint(pixel_u=30, pixel_v=30, depth_z=75.0),
    ]
    ref = CalibrationReference(method=CalibrationMethod.GCP, gcps=gcps)

    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=ref,
        intrinsics=sample_intrinsics,
    )
    assert calib_res.success
    assert pytest.approx(calib_res.scale_factor, rel=1e-3) == 30.0
    assert calib_res.residual_rmse is not None
    assert calib_res.residual_rmse < 0.1


def test_known_distance_calibration(sample_relative_depth, sample_intrinsics):
    """Verify calibration from known physical distance between two image pixels."""
    # Point 1 at (20, 25) with rel depth 2.5
    # Point 2 at (30, 25) with rel depth 2.5
    # fx = 100, delta_u = 10 -> delta_X_rel = (10 * 2.5) / 100 = 0.25 relative units
    # Target distance = 5.0 meters -> scale = 5.0 / 0.25 = 20.0
    dist_ref = DistanceMeasurement(point1_pixel=[20.0, 25.0], point2_pixel=[30.0, 25.0], distance_meters=5.0)
    ref = CalibrationReference(method=CalibrationMethod.KNOWN_DISTANCE, distance_references=[dist_ref])

    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=ref,
        intrinsics=sample_intrinsics,
    )
    assert calib_res.success
    assert pytest.approx(calib_res.scale_factor, rel=1e-3) == 20.0


def test_manual_scale_calibration(sample_relative_depth, sample_intrinsics):
    """Verify that manual scale factor strictly scales relative depth without renormalizing."""
    ref = CalibrationReference(
        method=CalibrationMethod.MANUAL_SCALE,
        manual_scale_factor=10.0,
    )
    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=ref,
        intrinsics=sample_intrinsics,
    )
    assert calib_res.success is True
    assert calib_res.method == CalibrationMethod.MANUAL_SCALE
    assert calib_res.scale_factor == 10.0
    assert calib_res.shift_offset == 0.0
    assert metric_depth is not None
    assert metric_depth.is_metric is True
    assert metric_depth.units == "meters"

    # Numerical verification: metric_depth == 10.0 * relative_depth
    np.testing.assert_allclose(metric_depth.array, 10.0 * sample_relative_depth.array, rtol=1e-5, atol=1e-5)
    assert pytest.approx(metric_depth.min_depth_m, rel=1e-3) == 10.0 * sample_relative_depth.min_val
    assert pytest.approx(metric_depth.max_depth_m, rel=1e-3) == 10.0 * sample_relative_depth.max_val



def test_reference_dem_calibration(sample_relative_depth, sample_intrinsics):
    """Verify metric calibration using a reference digital elevation model (DEM) with verified correspondence."""
    # Create synthetic reference DEM array (50x50)
    # At top half (rel depth 1.5): elevation = 60.0m -> with cam_alt 120.0m, target depth = 60.0m -> scale 40.0
    # At bottom half (rel depth 2.5): elevation = 20.0m -> with cam_alt 120.0m, target depth = 100.0m -> scale 40.0
    dem_arr = np.zeros((50, 50), dtype=np.float32)
    dem_arr[:25, :] = 60.0
    dem_arr[25:, :] = 20.0

    ref = CalibrationReference(
        method=CalibrationMethod.REFERENCE_DEM,
        reference_dem_array=dem_arr,
        camera_altitude_m=120.0,
        has_verified_correspondence=True,
    )

    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=ref,
        intrinsics=sample_intrinsics,
    )

    assert calib_res.success
    assert calib_res.method == CalibrationMethod.REFERENCE_DEM
    assert metric_depth is not None
    assert metric_depth.is_metric
    assert metric_depth.units == "meters"
    assert pytest.approx(calib_res.scale_factor, rel=1e-2) == 40.0
    assert calib_res.confidence >= 0.70
    assert calib_res.details["valid_dem_cell_count"] == 2500
    assert calib_res.details["correspondence_verified"] is True


def test_reference_dem_unverified_correspondence_refusal(sample_relative_depth, sample_intrinsics):
    """Verify that ReferenceDEMStrategy refuses calibration when spatial correspondence is unverified."""
    dem_arr = np.full((50, 50), 50.0, dtype=np.float32)
    ref = CalibrationReference(
        method=CalibrationMethod.REFERENCE_DEM,
        reference_dem_array=dem_arr,
        camera_altitude_m=120.0,
        has_verified_correspondence=False,
        is_orthorectified=False,
    )

    metric_depth, calib_res = MetricCalibrator.calibrate(
        relative_depth=sample_relative_depth,
        reference=ref,
        intrinsics=sample_intrinsics,
        allow_provisional_fallback=False,
    )

    assert not calib_res.success
    assert metric_depth is None
    assert "correspondence" in calib_res.reason.lower()


