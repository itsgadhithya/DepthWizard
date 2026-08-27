"""Unit tests for DSM rasterization and GeoTIFF export."""

import pytest
import numpy as np
import rasterio
from pathlib import Path

from backend.models.geometry import PointCloud3D, CoordinateFrame, Units
from backend.models.dsm import DSMResult
from backend.dsm.rasterizer import DSMRasterizer
from backend.dsm.geotiff_exporter import GeoTIFFExporter


def test_dsm_rasterizer():
    """Verify rasterizing georeferenced 3D point cloud onto a regular elevation grid."""
    # Synthetic terrain points: 10m x 10m area with elevation 50m to 60m
    x = np.linspace(500000.0, 500010.0, 100)
    y = np.linspace(4000000.0, 4000010.0, 100)
    xx, yy = np.meshgrid(x, y)
    zz = 50.0 + 0.5 * (xx - 500000.0) + 0.5 * (yy - 4000000.0)  # Sloping plane

    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
    pc = PointCloud3D(
        points=pts,
        colors=np.zeros_like(pts, dtype=np.uint8),
        coordinate_frame=CoordinateFrame.PROJECTED_CRS,
        units=Units.METERS,
        is_metric=True,
        crs="EPSG:32643",
    )
    pc.compute_bounds()

    dsm = DSMRasterizer.rasterize(pc, resolution_m=1.0)

    assert isinstance(dsm, DSMResult)
    assert dsm.crs == "EPSG:32643"
    assert dsm.dsm_type == "georeferenced_metric"
    assert dsm.is_local is False
    assert dsm.units == "meters"
    assert dsm.resolution_m == 1.0
    assert dsm.min_elevation_m >= 50.0
    assert dsm.max_elevation_m <= 65.0
    assert dsm.valid_coverage_percent > 80.0
    assert dsm.width > 5
    assert dsm.height > 5


def test_local_metric_dsm_rasterization():
    """Verify Local Metric DSM generation when metric depth is available but no CRS exists."""
    # Metric camera frame point cloud: 20m x 20m footprint, depth 10m to 20m
    x = np.linspace(-10.0, 10.0, 50)
    y = np.linspace(-10.0, 10.0, 50)
    xx, yy = np.meshgrid(x, y)
    zz = 15.0 + 0.2 * xx - 0.1 * yy

    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
    pc = PointCloud3D(
        points=pts,
        colors=np.zeros_like(pts, dtype=np.uint8),
        coordinate_frame=CoordinateFrame.CAMERA_FRAME,
        units=Units.METERS,
        is_metric=True,
        crs=None,  # No CRS in camera frame
    )

    dsm = DSMRasterizer.rasterize(pc, resolution_m=0.5, is_local=True)

    assert isinstance(dsm, DSMResult)
    assert dsm.is_local is True
    assert dsm.dsm_type == "local_metric"
    assert dsm.crs is None
    assert dsm.units == "meters"
    assert dsm.resolution_m == 0.5
    assert dsm.width >= 40
    assert dsm.height >= 40
    assert 10.0 <= dsm.min_elevation_m <= dsm.max_elevation_m <= 25.0


def test_dsm_uncalibrated_relative_rejection():
    """Verify that uncalibrated relative point clouds strictly refuse DSM rasterization."""
    pc = PointCloud3D(
        points=np.array([[0, 0, 1.5], [1, 1, 2.0]], dtype=np.float32),
        coordinate_frame=CoordinateFrame.CAMERA_FRAME,
        units=Units.RELATIVE,
        is_metric=False,
    )
    with pytest.raises(ValueError, match="uncalibrated relative point cloud"):
        DSMRasterizer.rasterize(pc)


def test_dsm_nodata_preservation():
    """Verify NoData sentinel values (-9999.0) are preserved for sparse grids."""
    # Only 4 isolated corner points
    pts = np.array([
        [-5.0, -5.0, 12.0],
        [5.0, -5.0, 12.0],
        [-5.0, 5.0, 14.0],
        [5.0, 5.0, 14.0],
    ], dtype=np.float32)
    pc = PointCloud3D(
        points=pts,
        coordinate_frame=CoordinateFrame.CAMERA_FRAME,
        units=Units.METERS,
        is_metric=True,
        crs=None,
    )

    # Disable void filling to test raw nodata cells
    dsm = DSMRasterizer.rasterize(pc, resolution_m=1.0, fill_voids=False, nodata_value=-9999.0)

    assert np.any(dsm.grid == -9999.0)
    assert dsm.nodata_value == -9999.0
    assert dsm.valid_coverage_percent < 100.0


def test_geotiff_exporter(tmp_path):
    """Verify GeoTIFF export writes a readable GeoTIFF with correct metadata."""
    w, h = 20, 20
    grid = np.full((h, w), 55.5, dtype=np.float32)
    transform = [500000.0, 1.0, 0.0, 4000020.0, 0.0, -1.0]

    from backend.models.geospatial import GeoBounds
    bounds = GeoBounds(min_x=500000.0, min_y=4000000.0, max_x=500020.0, max_y=4000020.0, crs="EPSG:32643")

    dsm = DSMResult(
        grid=grid,
        width=w,
        height=h,
        crs="EPSG:32643",
        dsm_type="georeferenced_metric",
        is_local=False,
        transform=transform,
        bounds=bounds,
        resolution_m=1.0,
        min_elevation_m=55.5,
        max_elevation_m=55.5,
        mean_elevation_m=55.5,
        nodata_value=-9999.0,
        valid_pixel_count=w * h,
        valid_coverage_percent=100.0,
        units="meters",
    )

    out_file = tmp_path / "test_dsm.tif"
    saved = GeoTIFFExporter.save_dsm_geotiff(dsm, str(out_file))

    assert Path(saved).exists()
    with rasterio.open(saved) as src:
        assert src.width == w
        assert src.height == h
        assert "32643" in src.crs.to_string()
        assert src.nodata == -9999.0
        data = src.read(1)
        assert pytest.approx(data[0, 0]) == 55.5


def test_local_dsm_geotiff_export(tmp_path):
    """Verify Local Metric DSM exports cleanly as standard TIFF without fabricated CRS."""
    w, h = 15, 15
    grid = np.full((h, w), 12.3, dtype=np.float32)
    transform = [-7.5, 1.0, 0.0, 7.5, 0.0, -1.0]

    from backend.models.geospatial import GeoBounds
    bounds = GeoBounds(min_x=-7.5, min_y=-7.5, max_x=7.5, max_y=7.5, crs="LOCAL")

    dsm = DSMResult(
        grid=grid,
        width=w,
        height=h,
        crs=None,
        dsm_type="local_metric",
        is_local=True,
        transform=transform,
        bounds=bounds,
        resolution_m=1.0,
        min_elevation_m=12.3,
        max_elevation_m=12.3,
        mean_elevation_m=12.3,
        nodata_value=-9999.0,
        valid_pixel_count=w * h,
        valid_coverage_percent=100.0,
        units="meters",
    )

    out_file = tmp_path / "test_local_dsm.tif"
    saved = GeoTIFFExporter.save_dsm_geotiff(dsm, str(out_file))

    assert Path(saved).exists()
    with rasterio.open(saved) as src:
        assert src.width == w
        assert src.height == h
        assert src.crs is None
        assert src.nodata == -9999.0
        assert src.tags()["DSM_TYPE"] == "local_metric"
        assert src.tags()["IS_LOCAL"] == "True"
        assert src.tags()["ELEVATION_UNITS"] == "meters"
