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
    """Verify rasterizing 3D point cloud onto a regular elevation grid."""
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
    assert dsm.resolution_m == 1.0
    assert dsm.min_elevation_m >= 50.0
    assert dsm.max_elevation_m <= 65.0
    assert dsm.valid_coverage_percent > 80.0
    assert dsm.width > 5
    assert dsm.height > 5


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
