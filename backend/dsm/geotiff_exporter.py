"""GeoTIFF exporter generating geospatial raster files with CRS and affine transforms."""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

from backend.models.dsm import DSMResult


class GeoTIFFExporter:
    """Exports DSMResult to standards-compliant GeoTIFF raster files."""

    @classmethod
    def save_dsm_geotiff(
        cls,
        dsm: DSMResult,
        file_path: str,
        tiled: bool = True,
    ) -> str:
        """Write a DSMResult to a standards-compliant GeoTIFF raster file.

        Args:
            dsm: DSMResult instance containing 2D elevation grid and CRS.
            file_path: Target GeoTIFF file destination.
            tiled: Whether to write as a tiled Cloud-Optimized GeoTIFF structure.

        Returns:
            Absolute path to saved GeoTIFF file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        t = dsm.transform
        # Convert 6-parameter GDAL list [c, a, b, f, d, e] to rasterio Affine
        # Affine(a, b, c, d, e, f)
        affine_transform = Affine.from_gdal(t[0], t[1], t[2], t[3], t[4], t[5])

        crs_obj = CRS.from_user_input(dsm.crs)

        profile: Dict[str, Any] = {
            "driver": "GTiff",
            "height": dsm.height,
            "width": dsm.width,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": crs_obj,
            "transform": affine_transform,
            "nodata": dsm.nodata_value,
            "compress": "deflate",
        }

        # Enable tiled structure for rasters >= 64x64
        if tiled and dsm.width >= 64 and dsm.height >= 64:
            block_size = 256 if (dsm.width >= 256 and dsm.height >= 256) else 64
            profile["tiled"] = True
            profile["blockxsize"] = block_size
            profile["blockysize"] = block_size
            # Predictor 3 is optimal for 32-bit floating point elevation values
            profile["predictor"] = 3

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(dsm.grid.astype(np.float32), 1)

            # Write standard geospatial provenance and elevation tags
            tags = {
                "SOFTWARE": "DepthWizard Depth & Metric Geometry Engine",
                "ELEVATION_UNITS": "meters",
                "METRIC_CALIBRATED": "true",
                "HORIZONTAL_CRS": str(dsm.crs),
                "VERTICAL_DATUM": "WGS84 Ellipsoid / Projected Datum",
                "RESOLUTION_METERS": str(dsm.resolution_m),
                "MIN_ELEVATION_M": str(dsm.min_elevation_m),
                "MAX_ELEVATION_M": str(dsm.max_elevation_m),
                "MEAN_ELEVATION_M": str(dsm.mean_elevation_m),
                "STD_ELEVATION_M": str(dsm.std_elevation_m),
                "VALID_COVERAGE_PERCENT": str(dsm.valid_coverage_percent),
            }

            if dsm.metadata:
                tags["GENERATION_METHOD"] = dsm.metadata.generation_method
                tags["VOID_FILLING_APPLIED"] = str(dsm.metadata.void_filling_applied)
                tags["SOURCE_POINTS_COUNT"] = str(dsm.metadata.source_points_count)
                tags["CREATED_AT"] = dsm.metadata.created_at

            dst.update_tags(**tags)

        return str(path.resolve())
