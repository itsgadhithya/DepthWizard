"""Digital Surface Model (DSM) rasterizer generating gridded surface elevation models."""

import math
from typing import Optional, Tuple, Literal
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin, Affine

from backend.config import settings
from backend.models.geometry import PointCloud3D, CoordinateFrame
from backend.models.dsm import DSMResult, DSMMetadata
from backend.models.geospatial import GeoBounds


class DSMRasterizer:
    """Rasterizes 3D point clouds into 2D Digital Surface Models (DSM) with standard geospatial transforms."""

    @classmethod
    def rasterize(
        cls,
        point_cloud: PointCloud3D,
        resolution_m: float = settings.default_dsm_resolution,
        nodata_value: float = settings.default_nodata_value,
        fill_voids: bool = True,
        method: Literal["max", "mean"] = "max",
    ) -> DSMResult:
        """Rasterize a georeferenced metric point cloud into a DSM elevation grid.

        Args:
            point_cloud: Georeferenced PointCloud3D in a Projected CRS (Units: Meters).
            resolution_m: Ground sampling distance / grid resolution in meters per pixel.
            nodata_value: Sentinel value for missing/void elevation cells.
            fill_voids: Whether to interpolate internal void cells.
            method: 'max' for Digital Surface Model (highest surface), 'mean' for average surface.

        Returns:
            DSMResult containing 2D float32 elevation array, CRS, Affine transform, and metadata.
        """
        if not point_cloud.is_metric:
            raise ValueError(
                "Cannot generate metric DSM from uncalibrated relative point cloud. "
                "Metric depth calibration must be performed first."
            )

        if not point_cloud.crs:
            raise ValueError(
                "Point cloud lacks a Projected CRS. "
                "Geospatial georeferencing is required for DSM generation."
            )

        pts = point_cloud.points
        if pts is None or len(pts) == 0:
            raise ValueError("Cannot rasterize an empty point cloud.")

        # Extract X (Easting), Y (Northing), Z (Elevation above datum)
        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        z_elevation = pts[:, 2].astype(np.float64)

        # Filter NaNs / Infs
        finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z_elevation)
        if not np.any(finite_mask):
            raise ValueError("Point cloud contains no finite 3D coordinates.")

        x = x[finite_mask]
        y = y[finite_mask]
        z_elevation = z_elevation[finite_mask]

        min_x, max_x = float(np.min(x)), float(np.max(x))
        min_y, max_y = float(np.min(y)), float(np.max(y))

        # Enforce positive resolution
        resolution_m = max(0.01, float(resolution_m))

        # Compute grid extent
        grid_w = max(1, int(math.ceil((max_x - min_x) / resolution_m)))
        grid_h = max(1, int(math.ceil((max_y - min_y) / resolution_m)))

        # Ensure valid grid dimensions (minimum 2x2 cells)
        grid_w = max(2, grid_w)
        grid_h = max(2, grid_h)

        # Define grid bounds:
        # west = min_x, north = min_y + grid_h * resolution_m (aligned to encompass max_y)
        west = min_x
        north = min_y + (grid_h * resolution_m)

        # Rasterio Affine transform: top-left corner origin
        # transform: [west, resolution_m, 0.0, north, 0.0, -resolution_m]
        aff = from_origin(west, north, resolution_m, resolution_m)
        transform_list = [aff.c, aff.a, aff.b, aff.f, aff.d, aff.e]

        # Initialize elevation grid with nodata sentinel
        dsm_grid = np.full((grid_h, grid_w), nodata_value, dtype=np.float32)

        # Map continuous (x, y) coordinates to discrete pixel column and row indices
        col_indices = np.clip(((x - west) / resolution_m).astype(np.int32), 0, grid_w - 1)
        row_indices = np.clip(((north - y) / resolution_m).astype(np.int32), 0, grid_h - 1)

        flat_indices = row_indices * grid_w + col_indices

        # Surface accumulation
        if method == "max":
            # Digital Surface Model: Maximum elevation per cell (captures roof tops, canopies, towers)
            np.maximum.at(dsm_grid.ravel(), flat_indices, z_elevation.astype(np.float32))
        else:
            # Mean elevation
            counts = np.zeros(dsm_grid.size, dtype=np.int32)
            sums = np.zeros(dsm_grid.size, dtype=np.float64)
            np.add.at(sums, flat_indices, z_elevation)
            np.add.at(counts, flat_indices, 1)
            valid_idx = counts > 0
            dsm_grid.ravel()[valid_idx] = (sums[valid_idx] / counts[valid_idx]).astype(np.float32)

        valid_mask = dsm_grid != nodata_value

        # Optional void filling using nearest neighbor interpolation
        voids_filled = False
        if fill_voids and np.any(valid_mask) and not np.all(valid_mask):
            valid_rows, valid_cols = np.where(valid_mask)
            valid_elevations = dsm_grid[valid_mask]
            missing_rows, missing_cols = np.where(~valid_mask)

            # Subsample if valid set is exceptionally large for fast interpolation
            if len(valid_rows) > 50_000:
                step = len(valid_rows) // 50_000
                v_r = valid_rows[::step]
                v_c = valid_cols[::step]
                v_z = valid_elevations[::step]
            else:
                v_r = valid_rows
                v_c = valid_cols
                v_z = valid_elevations

            try:
                interp_z = griddata(
                    points=(v_r, v_c),
                    values=v_z,
                    xi=(missing_rows, missing_cols),
                    method="nearest",
                )
                dsm_grid[missing_rows, missing_cols] = interp_z.astype(np.float32)
                valid_mask = np.isfinite(dsm_grid) & (dsm_grid != nodata_value)
                voids_filled = True
            except Exception:
                pass

        valid_elevs = dsm_grid[valid_mask]
        if len(valid_elevs) > 0:
            min_elev = float(np.min(valid_elevs))
            max_elev = float(np.max(valid_elevs))
            mean_elev = float(np.mean(valid_elevs))
            std_elev = float(np.std(valid_elevs))
        else:
            min_elev = max_elev = mean_elev = std_elev = 0.0

        valid_count = int(np.sum(valid_mask))
        nodata_count = int(dsm_grid.size - valid_count)
        coverage_pct = float(valid_count / dsm_grid.size * 100.0)

        bounds = GeoBounds(
            min_x=west,
            min_y=north - (grid_h * resolution_m),
            max_x=west + (grid_w * resolution_m),
            max_y=north,
            min_z=min_elev,
            max_z=max_elev,
            crs=point_cloud.crs,
        )

        metadata = DSMMetadata(
            generation_method="max_surface_elevation" if method == "max" else "mean_surface_elevation",
            void_filling_applied=voids_filled,
            source_points_count=int(len(x)),
            valid_cells_count=valid_count,
            nodata_cells_count=nodata_count,
            valid_coverage_percent=round(coverage_pct, 2),
            min_elevation_m=round(min_elev, 3),
            max_elevation_m=round(max_elev, 3),
            mean_elevation_m=round(mean_elev, 3),
            std_elevation_m=round(std_elev, 3),
            horizontal_crs=point_cloud.crs,
            vertical_datum="WGS84 Ellipsoidal Height / Projected Datum",
            elevation_units="meters",
            resolution_m=resolution_m,
        )

        return DSMResult(
            grid=dsm_grid,
            width=grid_w,
            height=grid_h,
            crs=point_cloud.crs,
            transform=transform_list,
            bounds=bounds,
            resolution_m=resolution_m,
            min_elevation_m=round(min_elev, 3),
            max_elevation_m=round(max_elev, 3),
            mean_elevation_m=round(mean_elev, 3),
            std_elevation_m=round(std_elev, 3),
            nodata_value=nodata_value,
            valid_pixel_count=valid_count,
            valid_coverage_percent=round(coverage_pct, 2),
            units="meters",
            metadata=metadata,
        )
