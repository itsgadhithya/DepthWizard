"""End-to-end integration tests for SingleImagePipeline across States A, B, C, and D."""

import pytest
import numpy as np
from pathlib import Path

from backend.pipeline import SingleImagePipeline
from backend.models.results import PipelineState, ProcessingSummary
from backend.models.calibration import CalibrationReference, CalibrationMethod, GCPPoint


def test_pipeline_plain_image_provisional_fallback(synthetic_jpeg_bytes):
    """Provisional Fallback: Plain image with no metric references produces provisional metric depth (10x) and Local Metric DSM in State C."""
    summary = SingleImagePipeline.process(
        image_input=synthetic_jpeg_bytes,
        filename="plain_image.jpg",
        calibration_ref=None,
    )

    assert isinstance(summary, ProcessingSummary)
    assert summary.relative_depth_available is True
    assert summary.camera_model_available is True
    assert summary.metric_depth_available is True  # Provisional metric scale applied
    assert summary.georeferencing_available is False  # Plain image has NO georeferencing
    assert summary.dsm_available is True  # Local Metric DSM generated
    assert summary.dsm_type == "local_metric"
    assert summary.state == PipelineState.STATE_C

    # Calibration details
    assert summary.calibration is not None
    assert summary.calibration.success is True
    assert summary.calibration.method == CalibrationMethod.PROVISIONAL_FIXED_SCALE
    assert summary.calibration.scale_factor == 10.0
    assert summary.calibration.shift_offset == 0.0
    assert summary.calibration.confidence == 0.20
    assert summary.calibration.reference_count == 0
    assert summary.calibration.is_provisional is True

    # Warnings check
    assert any("Provisional metric scale applied" in w for w in summary.warnings)

    # Artifacts check
    assert "raw_relative_depth_npy" in summary.artifacts
    assert "relative_depth_visual_png" in summary.artifacts
    assert "point_cloud_ply" in summary.artifacts
    assert "metric_depth_npy" in summary.artifacts
    assert "dsm_geotiff" in summary.artifacts
    assert "dsm_model_glb" in summary.artifacts
    assert "dsm_npy" in summary.artifacts
    assert "dsm_visual_png" in summary.artifacts


def test_pipeline_state_c_metric_calibration(synthetic_jpeg_bytes):
    """State C: Plain image + explicit GCP calibration produces metric depth, 3D geometry in meters, and Local Metric DSM."""
    gcps = [
        GCPPoint(pixel_u=20, pixel_v=20, depth_z=30.0),
        GCPPoint(pixel_u=60, pixel_v=60, depth_z=30.0),
    ]
    calib_ref = CalibrationReference(method=CalibrationMethod.GCP, gcps=gcps)

    summary = SingleImagePipeline.process(
        image_input=synthetic_jpeg_bytes,
        filename="gcp_calibrated.jpg",
        calibration_ref=calib_ref,
    )

    assert summary.relative_depth_available is True
    assert summary.metric_depth_available is True
    assert summary.dsm_available is True
    assert summary.dsm_type == "local_metric"
    assert summary.state == PipelineState.STATE_C
    assert "metric_depth_npy" in summary.artifacts
    assert "dsm_geotiff" in summary.artifacts
    assert "dsm_model_glb" in summary.artifacts
    assert summary.calibration is not None
    assert summary.calibration.success is True
    assert summary.calibration.method == CalibrationMethod.GCP
    assert summary.calibration.is_provisional is False


def test_pipeline_state_d_georeferenced_dsm(synthetic_jpeg_with_exif):
    """State D: Image with GPS EXIF and Altitude reference produces georeferenced metric DSM."""
    calib_ref = CalibrationReference(
        method=CalibrationMethod.ALTITUDE_GROUND,
        camera_altitude_m=150.0,
        ground_elevation_m=20.0,  # AGL = 130m
    )

    summary = SingleImagePipeline.process(
        image_input=synthetic_jpeg_with_exif,
        filename="drone_dsm.jpg",
        calibration_ref=calib_ref,
        dsm_resolution_m=1.0,
    )

    assert summary.relative_depth_available
    assert summary.metric_depth_available
    assert summary.georeferencing_available
    assert summary.dsm_available
    assert summary.dsm_type == "georeferenced_metric"
    assert summary.state == PipelineState.STATE_D

    # Artifacts check
    assert "raw_relative_depth_npy" in summary.artifacts
    assert "metric_depth_npy" in summary.artifacts
    assert "point_cloud_ply" in summary.artifacts
    assert "dsm_geotiff" in summary.artifacts
    assert "dsm_model_glb" in summary.artifacts
    assert "dsm_npy" in summary.artifacts
    assert "dsm_visual_png" in summary.artifacts
    assert "dsm_hillshade_png" in summary.artifacts
    assert "dsm_color_relief_png" in summary.artifacts

    # File existence check
    dsm_info = summary.artifacts["dsm_geotiff"]
    assert Path(dsm_info.file_path).exists()
    glb_info = summary.artifacts["dsm_model_glb"]
    assert Path(glb_info.file_path).exists()
    assert Path(glb_info.file_path).stat().st_size >= 12


def test_pipeline_georeferenced_rgb_geotiff_capabilities(synthetic_rgb_geotiff_bytes):
    """Regression test: Georeferenced RGB GeoTIFF (e.g. Potsdam 5cm GSD) reports georeferencing_available=True

    and produces provisional metric depth and DSM in State C/D when no metric calibration reference is supplied.
    """
    summary = SingleImagePipeline.process(
        image_input=synthetic_rgb_geotiff_bytes,
        filename="top_potsdam_2_10_RGB.tif",
        calibration_ref=None,
    )

    # Capability assertions
    assert summary.relative_depth_available is True
    assert summary.camera_model_available is True
    assert summary.metric_depth_available is True  # Provisional scale applied
    assert summary.dsm_available is True
    assert summary.dsm_type in ["georeferenced_metric", "local_metric"]
    assert summary.calibration is not None
    assert summary.calibration.is_provisional is True
    assert summary.calibration.scale_factor == 10.0
    assert summary.georeferencing_available is True  # MUST be true for georeferenced GeoTIFF

    # Metadata assertions
    assert summary.metadata is not None
    assert summary.metadata.has_geotiff is True
    assert summary.metadata.is_dem is False
    assert summary.metadata.geotiff is not None
    assert "32633" in summary.metadata.geotiff.crs
    assert summary.metadata.geotiff.transform is not None
    assert len(summary.metadata.geotiff.transform) == 6
    assert summary.metadata.geotiff.bounds is not None
    assert summary.metadata.geotiff.resolution is not None
    assert pytest.approx(summary.metadata.geotiff.resolution[0], rel=1e-3) == 0.05
    assert pytest.approx(summary.metadata.geotiff.resolution[1], rel=1e-3) == 0.05


def test_pipeline_single_band_dem_passthrough(synthetic_geotiff_bytes):
    """Regression test: Single-band DEM GeoTIFF must be classified as DEM, bypass optical inference,

    and report georeferencing_available=True and produce valid DSM.
    """
    summary = SingleImagePipeline.process(
        image_input=synthetic_geotiff_bytes,
        filename="terrain_dem.tif",
        calibration_ref=None,
    )

    assert summary.metadata is not None
    assert summary.metadata.has_geotiff is True
    assert summary.metadata.is_dem is True
    assert summary.georeferencing_available is True
    assert summary.metric_depth_available is True
    assert summary.dsm_available is True
    assert summary.calibration is not None
    assert summary.calibration.is_provisional is True
    assert any("elevation raster (DEM)" in w for w in summary.warnings)


def test_pipeline_gps_only_image_capabilities(synthetic_jpeg_with_exif):
    """Regression test: An image with GPS EXIF but no orientation or metric calibration produces:
    - georeferencing_available = True
    - camera_position_available = True
    - camera_orientation_available = False
    - complete_camera_pose_available = False
    - metric_depth_available = True (provisional)
    - dsm_available = True (local metric DSM)
    - dsm_type = 'local_metric' (since orientation is uncalibrated)
    - is_provisional = True
    """
    summary = SingleImagePipeline.process(
        image_input=synthetic_jpeg_with_exif,
        filename="gps_drone_photo.jpg",
        calibration_ref=None,
    )

    assert summary.relative_depth_available is True
    assert summary.camera_model_available is True
    assert summary.georeferencing_available is True
    assert summary.camera_position_available is True
    assert summary.camera_orientation_available is False
    assert summary.complete_camera_pose_available is False
    assert summary.metric_depth_available is True
    assert summary.dsm_available is True
    assert summary.dsm_type in ["georeferenced_metric", "local_metric"]
    assert summary.calibration is not None
    assert summary.calibration.is_provisional is True
    assert summary.calibration.scale_factor == 10.0


def test_pipeline_manual_scale_end_to_end(synthetic_jpeg_bytes):
    """Regression test: Ordinary RGB image with injected manual scale reference (10x)

    produces calibrated metric depth (metric_depth == 10.0 * relative_depth), Local Metric DSM, and preserves
    the raw relative-depth array unchanged.
    """
    calib_ref = CalibrationReference(
        method=CalibrationMethod.MANUAL_SCALE,
        manual_scale_factor=10.0,
    )

    summary = SingleImagePipeline.process(
        image_input=synthetic_jpeg_bytes,
        filename="controlled_manual_scale_test.jpg",
        calibration_ref=calib_ref,
    )

    # 1. Pipeline capabilities and state
    assert summary.relative_depth_available is True
    assert summary.camera_model_available is True
    assert summary.metric_depth_available is True
    assert summary.state == PipelineState.STATE_C
    assert summary.georeferencing_available is False  # Plain image has no geospatial georeferencing
    assert summary.dsm_available is True  # Local Metric DSM generated!
    assert summary.dsm_type == "local_metric"

    # 2. Calibration summary details
    assert summary.calibration is not None
    assert summary.calibration.success is True
    assert summary.calibration.method == CalibrationMethod.MANUAL_SCALE
    assert summary.calibration.scale_factor == 10.0
    assert summary.calibration.shift_offset == 0.0

    # 3. Computational artifacts verification
    raw_rel_path = summary.artifacts["raw_relative_depth_npy"].file_path
    metric_path = summary.artifacts["metric_depth_npy"].file_path
    dsm_path = summary.artifacts["dsm_npy"].file_path

    raw_rel_depth = np.load(raw_rel_path)
    metric_depth = np.load(metric_path)
    dsm_grid = np.load(dsm_path)

    # 4. Numerical verification: metric_depth == 10.0 * raw_rel_depth exactly
    np.testing.assert_allclose(metric_depth, 10.0 * raw_rel_depth, rtol=1e-5, atol=1e-5)

    # 5. Min/Max metric bounds match
    assert pytest.approx(float(np.min(metric_depth)), rel=1e-3) == 10.0 * float(np.min(raw_rel_depth))
    assert pytest.approx(float(np.max(metric_depth)), rel=1e-3) == 10.0 * float(np.max(raw_rel_depth))
    assert np.any(np.isfinite(dsm_grid))



