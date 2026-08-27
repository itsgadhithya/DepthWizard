"""Unit tests for CRS resolution and geospatial coordinate transformation."""

import pytest
import numpy as np

from backend.geospatial.crs import CRSHelper
from backend.geospatial.transformer import CoordinateTransformer
from backend.models.geometry import PointCloud3D, CoordinateFrame, Units
from backend.models.camera import CameraModel, CameraIntrinsics, CameraExtrinsics


def test_utm_crs_auto_detection():
    """Verify UTM zone auto-calculation for various global locations."""
    # San Francisco, USA (W 122.4194, N 37.7749) -> UTM Zone 10N -> EPSG:32610
    crs_sf = CRSHelper.get_utm_crs_for_latlon(-122.4194, 37.7749)
    assert crs_sf == "EPSG:32610"

    # Paris, France (E 2.3522, N 48.8566) -> UTM Zone 31N -> EPSG:32631
    crs_paris = CRSHelper.get_utm_crs_for_latlon(2.3522, 48.8566)
    assert crs_paris == "EPSG:32631"

    # Sydney, Australia (E 151.2093, S 33.8688) -> UTM Zone 56S -> EPSG:32756
    crs_syd = CRSHelper.get_utm_crs_for_latlon(151.2093, -33.8688)
    assert crs_syd == "EPSG:32756"


def test_coordinate_transformer():
    """Verify transformation from camera frame to Projected CRS."""
    pts = np.array([
        [0.0, 0.0, 50.0],    # Point directly under nadir camera at 50m distance
        [10.0, 0.0, 50.0],   # 10m East
        [0.0, 10.0, 50.0],   # 10m South in optical frame -> 10m North in ENU
    ], dtype=np.float32)

    pc = PointCloud3D(
        points=pts,
        colors=np.zeros_like(pts, dtype=np.uint8),
        coordinate_frame=CoordinateFrame.CAMERA,
        units=Units.METERS,
        is_metric=True,
    )
    pc.compute_bounds()

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=250.0, cy=250.0, width=500, height=500)
    # Camera at altitude 100m MSL, pitch -90 (nadir)
    extrinsics = CameraExtrinsics(
        position_x=-122.4194,
        position_y=37.7749,
        position_z=100.0,
        pitch_deg=-90.0,
        yaw_deg=0.0,
    )
    camera = CameraModel(intrinsics=intrinsics, extrinsics=extrinsics)

    geo_pc = CoordinateTransformer.transform_camera_to_projected_crs(pc, camera, target_crs="EPSG:32610")

    assert geo_pc.coordinate_frame == CoordinateFrame.PROJECTED_CRS
    assert geo_pc.crs == "EPSG:32610"
    assert geo_pc.is_metric
    assert len(geo_pc.points) == 3

    # Elevation of point 1 at distance 50m from 100m camera -> 100 - 50 = 50m MSL
    assert pytest.approx(geo_pc.points[0, 2], rel=1e-2) == 50.0
