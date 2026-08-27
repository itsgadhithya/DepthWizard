"""Synthetic dataset generator for geospatial DSM testing and benchmarking without real aerial flight data."""

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np

from backend.models.geometry import PointCloud3D, CoordinateFrame, Units
from backend.models.dsm import DSMResult
from backend.models.camera import CameraModel, CameraIntrinsics, CameraExtrinsics
from backend.models.depth import MetricDepthMap, DepthMetadata
from backend.dsm.rasterizer import DSMRasterizer


class SyntheticDatasetGenerator:
    """Generates mathematically defined 3D point clouds and reference DSM grids for validation."""

    @classmethod
    def create_flat_terrain(
        cls,
        center_easting: float = 500_000.0,
        center_northing: float = 3_000_000.0,
        size_m: float = 50.0,
        elevation_m: float = 100.0,
        point_density_per_m2: float = 4.0,
        resolution_m: float = 0.5,
        crs: str = "EPSG:32643",
    ) -> Tuple[PointCloud3D, DSMResult]:
        """Generate a flat horizontal terrain plane at known constant elevation."""
        half_s = size_m / 2.0
        step = 1.0 / math.sqrt(point_density_per_m2)
        
        xs = np.arange(center_easting - half_s, center_easting + half_s, step)
        ys = np.arange(center_northing - half_s, center_northing + half_s, step)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, elevation_m, dtype=np.float32)

        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
        colors = np.full((len(points), 3), 128, dtype=np.uint8)

        pc = PointCloud3D(
            points=points,
            colors=colors,
            coordinate_frame=CoordinateFrame.PROJECTED_CRS,
            units=Units.METERS,
            is_metric=True,
            crs=crs,
        )
        pc.compute_bounds()

        dsm = DSMRasterizer.rasterize(pc, resolution_m=resolution_m)
        return pc, dsm

    @classmethod
    def create_stepped_building(
        cls,
        center_easting: float = 500_000.0,
        center_northing: float = 3_000_000.0,
        terrain_size_m: float = 60.0,
        ground_elev_m: float = 20.0,
        building_size_m: float = 20.0,
        building_height_m: float = 15.0,
        point_density_per_m2: float = 4.0,
        resolution_m: float = 0.5,
        crs: str = "EPSG:32643",
    ) -> Tuple[PointCloud3D, DSMResult]:
        """Generate ground terrain with a sharp rectangular building structure."""
        half_t = terrain_size_m / 2.0
        half_b = building_size_m / 2.0
        step = 1.0 / math.sqrt(point_density_per_m2)

        xs = np.arange(center_easting - half_t, center_easting + half_t, step)
        ys = np.arange(center_northing - half_t, center_northing + half_t, step)
        xx, yy = np.meshgrid(xs, ys)

        # Ground elevation
        zz = np.full_like(xx, ground_elev_m, dtype=np.float32)

        # Building mask
        b_mask = (
            (xx >= center_easting - half_b) & (xx <= center_easting + half_b) &
            (yy >= center_northing - half_b) & (yy <= center_northing + half_b)
        )
        zz[b_mask] = ground_elev_m + building_height_m  # Roof top elevation

        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[~b_mask.ravel()] = [34, 139, 34]   # Forest Green for ground
        colors[b_mask.ravel()] = [178, 34, 34]    # Brick Red for building

        pc = PointCloud3D(
            points=points,
            colors=colors,
            coordinate_frame=CoordinateFrame.PROJECTED_CRS,
            units=Units.METERS,
            is_metric=True,
            crs=crs,
        )
        pc.compute_bounds()

        dsm = DSMRasterizer.rasterize(pc, resolution_m=resolution_m, method="max")
        return pc, dsm

    @classmethod
    def create_sloped_terrain_with_peak(
        cls,
        center_easting: float = 500_000.0,
        center_northing: float = 3_000_000.0,
        size_m: float = 80.0,
        base_elev_m: float = 50.0,
        slope_x: float = 0.05,
        peak_height_m: float = 20.0,
        resolution_m: float = 0.5,
        crs: str = "EPSG:32643",
    ) -> Tuple[PointCloud3D, DSMResult]:
        """Generate smooth sloped terrain with a Gaussian peak / hill."""
        half_s = size_m / 2.0
        step = 0.5
        xs = np.arange(center_easting - half_s, center_easting + half_s, step)
        ys = np.arange(center_northing - half_s, center_northing + half_s, step)
        xx, yy = np.meshgrid(xs, ys)

        # Base slope
        dx = xx - center_easting
        dy = yy - center_northing
        base_surface = base_elev_m + slope_x * dx

        # Gaussian hill peak in the center
        sigma = size_m / 6.0
        hill = peak_height_m * np.exp(- (dx**2 + dy**2) / (2.0 * sigma**2))

        zz = (base_surface + hill).astype(np.float32)

        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

        pc = PointCloud3D(
            points=points,
            coordinate_frame=CoordinateFrame.PROJECTED_CRS,
            units=Units.METERS,
            is_metric=True,
            crs=crs,
        )
        pc.compute_bounds()

        dsm = DSMRasterizer.rasterize(pc, resolution_m=resolution_m)
        return pc, dsm

    @classmethod
    def create_synthetic_camera_and_depth(
        cls,
        width: int = 640,
        height: int = 480,
        fov_deg: float = 60.0,
        camera_altitude_m: float = 150.0,
        ground_elevation_m: float = 25.0,
        lat: float = 28.6139,
        lon: float = 77.2090,
    ) -> Tuple[CameraModel, MetricDepthMap]:
        """Generate synthetic camera model with extrinsics and corresponding metric depth map."""
        fx = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            is_calibrated=True,
            estimation_method="synthetic_ideal",
        )

        extrinsics = CameraExtrinsics(
            position_x=lon,
            position_y=lat,
            position_z=camera_altitude_m,
            yaw_deg=0.0,
            pitch_deg=-90.0,  # Nadir looking down
            roll_deg=0.0,
            coordinate_system="WGS84",
        )

        camera = CameraModel(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            make="SyntheticAerialSensor",
            model="VirtualUAV-4K",
        )

        # For a flat ground at ground_elevation_m, depth is constant along optical axis: Z = altitude - ground
        flight_height_agl = camera_altitude_m - ground_elevation_m
        
        # Add slight perspective slant: Z increases towards image corners
        u_coords = np.arange(width)
        v_coords = np.arange(height)
        uu, vv = np.meshgrid(u_coords, v_coords)

        # Distance along camera optical axis Z
        # For a horizontal flat plane at nadir, optical axis depth Z = flight_height_agl
        depth_array = np.full((height, width), flight_height_agl, dtype=np.float32)

        metadata = DepthMetadata(
            model_name="synthetic_ground_truth",
            encoder="virtual",
            input_width=width,
            input_height=height,
            is_metric=True,
            units="meters",
            min_depth_m=float(np.min(depth_array)),
            max_depth_m=float(np.max(depth_array)),
            mean_depth_m=float(np.mean(depth_array)),
        )

        metric_depth = MetricDepthMap(
            array=depth_array,
            width=width,
            height=height,
            min_depth_m=float(np.min(depth_array)),
            max_depth_m=float(np.max(depth_array)),
            mean_depth_m=float(np.mean(depth_array)),
            std_depth_m=float(np.std(depth_array)),
            calibration_method="ground_truth_synthetic",
            scale_factor=1.0,
            confidence_score=1.0,
        )

        return camera, metric_depth
