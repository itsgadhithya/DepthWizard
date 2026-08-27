"""Comprehensive validation tests for DSM generation, geospatial transformations, and GeoTIFF export."""

import math
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, xy, rowcol

from backend.models.geometry import PointCloud3D, CoordinateFrame, Units
from backend.models.dsm import DSMResult, DSMMetadata
from backend.dsm.rasterizer import DSMRasterizer
from backend.dsm.geotiff_exporter import GeoTIFFExporter
from backend.dsm.synthetic import SyntheticDatasetGenerator
from backend.geometry.backprojector import DepthBackProjector
from backend.geospatial.transformer import CoordinateTransformer
from backend.geospatial.crs import CRSHelper


def test_synthetic_flat_terrain_dsm():
    """Verify DSM rasterization on a flat horizontal synthetic surface at constant elevation."""
    pc, dsm = SyntheticDatasetGenerator.create_flat_terrain(
        center_easting=500_000.0,
        center_northing=3_000_000.0,
        size_m=40.0,
        elevation_m=120.0,
        point_density_per_m2=4.0,
        resolution_m=0.5,
        crs="EPSG:32643",
    )

    assert dsm.crs == "EPSG:32643"
    assert dsm.units == "meters"
    assert dsm.resolution_m == 0.5
    assert np.isclose(dsm.min_elevation_m, 120.0, atol=0.01)
    assert np.isclose(dsm.max_elevation_m, 120.0, atol=0.01)
    assert np.isclose(dsm.mean_elevation_m, 120.0, atol=0.01)
    assert np.isclose(dsm.std_elevation_m, 0.0, atol=0.01)
    assert dsm.valid_coverage_percent == 100.0

    # Check DSMMetadata
    assert dsm.metadata is not None
    assert dsm.metadata.generation_method == "max_surface_elevation"
    assert dsm.metadata.horizontal_crs == "EPSG:32643"
    assert dsm.metadata.source_points_count == len(pc.points)
    assert dsm.metadata.valid_cells_count > 0


def test_synthetic_stepped_building_dsm():
    """Verify DSM rasterization captures both building roof elevation and ground elevation."""
    ground_elev = 25.0
    building_height = 15.0
    roof_elev = ground_elev + building_height

    pc, dsm = SyntheticDatasetGenerator.create_stepped_building(
        center_easting=500_000.0,
        center_northing=3_000_000.0,
        terrain_size_m=60.0,
        ground_elev_m=ground_elev,
        building_size_m=20.0,
        building_height_m=building_height,
        resolution_m=0.5,
        crs="EPSG:32643",
    )

    assert np.isclose(dsm.min_elevation_m, ground_elev, atol=0.01)
    assert np.isclose(dsm.max_elevation_m, roof_elev, atol=0.01)
    assert dsm.min_elevation_m < dsm.mean_elevation_m < dsm.max_elevation_m
    assert dsm.std_elevation_m > 0.0

    # Check that roof cells exist with exact elevation
    valid_grid = dsm.grid[dsm.grid != dsm.nodata_value]
    assert np.any(np.isclose(valid_grid, roof_elev, atol=0.1))
    assert np.any(np.isclose(valid_grid, ground_elev, atol=0.1))


def test_synthetic_sloped_peak_dsm():
    """Verify DSM on continuous sloped terrain with a Gaussian peak."""
    pc, dsm = SyntheticDatasetGenerator.create_sloped_terrain_with_peak(
        center_easting=500_000.0,
        center_northing=3_000_000.0,
        size_m=50.0,
        base_elev_m=50.0,
        slope_x=0.1,
        peak_height_m=20.0,
        resolution_m=0.5,
        crs="EPSG:32643",
    )

    assert dsm.min_elevation_m >= 45.0  # Lowest slope corner
    assert dsm.max_elevation_m >= 70.0  # Peak elevation (50 + 20)
    assert dsm.valid_pixel_count == dsm.width * dsm.height


def test_geotiff_export_and_readback(tmp_path: Path):
    """Verify GeoTIFF exporter writes standards-compliant GeoTIFF and preserves CRS, transform, nodata, and tags."""
    pc, dsm = SyntheticDatasetGenerator.create_flat_terrain(
        center_easting=600_000.0,
        center_northing=4_000_000.0,
        size_m=30.0,
        elevation_m=75.5,
        resolution_m=0.5,
        crs="EPSG:32632",
    )

    target_file = tmp_path / "test_dsm.tif"
    exported_path = GeoTIFFExporter.save_dsm_geotiff(dsm, str(target_file))

    assert Path(exported_path).exists()
    assert Path(exported_path).stat().st_size > 0

    # Read back using rasterio
    with rasterio.open(exported_path) as src:
        assert src.driver == "GTiff"
        assert src.crs.to_string() == "EPSG:32632"
        assert src.width == dsm.width
        assert src.height == dsm.height
        assert src.nodata == dsm.nodata_value

        # Check Affine transform preservation
        t = dsm.transform
        expected_affine = Affine.from_gdal(t[0], t[1], t[2], t[3], t[4], t[5])
        assert src.transform.almost_equals(expected_affine)

        # Check metadata tags
        tags = src.tags()
        assert tags["SOFTWARE"] == "DepthWizard Depth & Metric Geometry Engine"
        assert tags["ELEVATION_UNITS"] == "meters"
        assert tags["METRIC_CALIBRATED"] == "true"
        assert tags["HORIZONTAL_CRS"] == "EPSG:32632"
        assert tags["RESOLUTION_METERS"] == "0.5"
        assert np.isclose(float(tags["MIN_ELEVATION_M"]), 75.5, atol=0.01)
        assert np.isclose(float(tags["MAX_ELEVATION_M"]), 75.5, atol=0.01)

        # Check grid array values match
        read_grid = src.read(1)
        assert np.allclose(read_grid, dsm.grid, equal_nan=True)


def test_affine_transform_roundtrip():
    """Verify precision of pixel to world coordinate roundtrip using rasterio transform."""
    _, dsm = SyntheticDatasetGenerator.create_flat_terrain(
        center_easting=500_000.0,
        center_northing=3_000_000.0,
        size_m=20.0,
        resolution_m=0.5,
        crs="EPSG:32643",
    )

    t = dsm.transform
    aff = Affine.from_gdal(t[0], t[1], t[2], t[3], t[4], t[5])

    test_row, test_col = 5, 8
    world_x, world_y = xy(aff, test_row, test_col, offset="center")
    calc_row, calc_col = rowcol(aff, world_x, world_y)

    assert calc_row == test_row
    assert calc_col == test_col


def test_camera_relative_depth_to_world_elevation_transform():
    """Verify rigorous distinction and mathematical conversion from optical depth Z to world elevation Z."""
    camera, metric_depth = SyntheticDatasetGenerator.create_synthetic_camera_and_depth(
        width=100,
        height=100,
        fov_deg=60.0,
        camera_altitude_m=150.0,
        ground_elevation_m=25.0,  # Expected world elevation
        lat=28.6139,
        lon=77.2090,
    )

    # 1. Backproject to Camera 3D points
    # Camera frame: +Z is forward distance from camera lens (here ~125m)
    cam_pc = DepthBackProjector.backproject_to_point_cloud(
        depth_map=metric_depth,
        intrinsics=camera.intrinsics,
    )
    assert cam_pc.coordinate_frame == CoordinateFrame.CAMERA_OPTICAL
    assert np.isclose(np.mean(cam_pc.points[:, 2]), 125.0, atol=1.0)

    # 2. Transform Camera Frame to Projected World CRS (UTM Zone 43N)
    utm_crs = CRSHelper.get_utm_crs_for_latlon(77.2090, 28.6139)
    world_pc = CoordinateTransformer.transform_camera_to_projected_crs(
        point_cloud=cam_pc,
        camera=camera,
        target_crs=utm_crs,
    )
    assert world_pc.coordinate_frame == CoordinateFrame.PROJECTED_CRS
    assert world_pc.crs == "EPSG:32643"
    # In world frame, Z is elevation above reference datum (~25m)
    assert np.isclose(np.mean(world_pc.points[:, 2]), 25.0, atol=1.0)

    # 3. Rasterize into DSM
    dsm = DSMRasterizer.rasterize(world_pc, resolution_m=1.0)
    assert dsm.crs == "EPSG:32643"
    assert np.isclose(dsm.mean_elevation_m, 25.0, atol=1.5)


def test_dsm_rasterizer_validation_guards():
    """Verify strict rejection when point clouds lack metric calibration and proper handling of local point clouds."""
    # 1. Non-metric point cloud must be rejected
    uncalibrated_pc = PointCloud3D(
        points=np.array([[0, 0, 10], [1, 1, 12]], dtype=np.float32),
        coordinate_frame=CoordinateFrame.CAMERA_OPTICAL,
        units=Units.NORMALIZED,
        is_metric=False,
        crs=None,
    )
    with pytest.raises(ValueError, match="uncalibrated relative point cloud"):
        DSMRasterizer.rasterize(uncalibrated_pc)

    # 2. Metric point cloud without CRS produces Local Metric DSM
    no_crs_pc = PointCloud3D(
        points=np.array([[0, 0, 10], [1, 1, 12]], dtype=np.float32),
        coordinate_frame=CoordinateFrame.CAMERA_OPTICAL,
        units=Units.METERS,
        is_metric=True,
        crs=None,
    )
    local_dsm = DSMRasterizer.rasterize(no_crs_pc, is_local=True)
    assert local_dsm.is_local is True
    assert local_dsm.dsm_type == "local_metric"
    assert local_dsm.crs is None

    # 3. Empty point cloud must be rejected
    empty_pc = PointCloud3D(
        points=np.zeros((0, 3), dtype=np.float32),
        coordinate_frame=CoordinateFrame.PROJECTED_CRS,
        units=Units.METERS,
        is_metric=True,
        crs="EPSG:32643",
    )
    with pytest.raises(ValueError, match="empty point cloud"):
        DSMRasterizer.rasterize(empty_pc)
