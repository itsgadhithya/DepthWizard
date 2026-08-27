"""Coordinate transformation engine converting Camera 3D points to Projected World CRS."""

import math
from typing import Optional, Tuple
import numpy as np
import pyproj
from pyproj import Transformer

from backend.models.geometry import PointCloud3D, CoordinateFrame, Units
from backend.models.camera import CameraModel
from backend.geospatial.crs import CRSHelper


class CoordinateTransformer:
    """Rigid and geodetic coordinate transformations for 3D point clouds."""

    @classmethod
    def transform_camera_to_projected_crs(
        cls,
        point_cloud: PointCloud3D,
        camera: CameraModel,
        target_crs: Optional[str] = None,
    ) -> PointCloud3D:
        """Transform 3D point cloud from Camera Frame into a target Projected CRS (e.g. UTM).

        Args:
            point_cloud: Metric point cloud in camera frame (Units: Meters).
            camera: CameraModel with valid extrinsics (GPS lat, lon, altitude).
            target_crs: Optional target EPSG code (e.g. 'EPSG:32643'). Auto-detected if None.

        Returns:
            Georeferenced PointCloud3D in Projected CRS.
        """
        if not point_cloud.is_metric:
            raise ValueError("Cannot georeference uncalibrated relative point cloud. Metric calibration is required.")

        if camera.extrinsics is None:
            raise ValueError("Camera extrinsics (GPS position) are missing. Georeferencing cannot proceed.")

        ext = camera.extrinsics
        lon = ext.longitude if ext.longitude is not None else (ext.position_x if abs(ext.position_x or 0.0) <= 180.0 else None)
        lat = ext.latitude if ext.latitude is not None else (ext.position_y if abs(ext.position_y or 0.0) <= 90.0 else None)
        alt = ext.altitude_m if ext.altitude_m is not None else (ext.position_z if ext.position_z is not None else 100.0)

        # Determine target CRS
        if not target_crs:
            if ext.projected_crs:
                target_crs = ext.projected_crs
            elif lon is not None and lat is not None:
                target_crs = CRSHelper.get_utm_crs_for_latlon(lon, lat)
            else:
                target_crs = "EPSG:32633"

        # Obtain center Easting / Northing in target projected CRS
        if ext.position_x is not None and ext.position_y is not None and ext.projected_crs == target_crs:
            center_easting = float(ext.position_x)
            center_northing = float(ext.position_y)
        elif lon is not None and lat is not None:
            # Create transformer from WGS84 to Target Projected CRS
            transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
            center_easting, center_northing = transformer.transform(lon, lat)
        else:
            raise ValueError("Camera geographic position (latitude/longitude) or projected position is required.")

        # Orientation angles (in radians)
        yaw_rad = math.radians(ext.yaw_deg if ext.yaw_deg is not None else 0.0)
        pitch_rad = math.radians(ext.pitch_deg if ext.pitch_deg is not None else -90.0)
        roll_rad = math.radians(ext.roll_deg if ext.roll_deg is not None else 0.0)

        # Camera frame to Local ENU rotation
        # Base optical nadir alignment (pitch = -90):
        # Cam +X (right) -> ENU East (+X)
        # Cam +Y (down) -> ENU South (-Y)
        # Cam +Z (forward) -> ENU Down (-Z)
        R_base = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)

        # Yaw rotation (around Z up)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        R_yaw = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y, cos_y, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Pitch offset from nadir (-90)
        pitch_offset = pitch_rad - math.radians(-90.0)
        cos_p, sin_p = math.cos(pitch_offset), math.sin(pitch_offset)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p],
            [0.0, sin_p, cos_p],
        ], dtype=np.float64)

        R_roll = np.array([
            [math.cos(roll_rad), 0.0, math.sin(roll_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(roll_rad), 0.0, math.cos(roll_rad)],
        ], dtype=np.float64)

        R_total = R_yaw @ R_pitch @ R_roll @ R_base

        cam_pts = point_cloud.points.astype(np.float64)  # (N, 3)

        # Rotate camera vectors into local ENU
        enu_pts = cam_pts @ R_total.T  # (N, 3)

        # Translate to Projected CRS coordinates
        world_x = (center_easting + enu_pts[:, 0]).astype(np.float32)
        world_y = (center_northing + enu_pts[:, 1]).astype(np.float32)
        world_z = (alt + enu_pts[:, 2]).astype(np.float32)  # Elevation in meters

        world_points = np.stack([world_x, world_y, world_z], axis=-1)

        # Transform normals if present
        world_normals = None
        if point_cloud.normals is not None:
            cam_normals = point_cloud.normals.astype(np.float64)
            world_normals = (cam_normals @ R_total.T).astype(np.float32)

        geo_pc = PointCloud3D(
            points=world_points,
            colors=point_cloud.colors,
            normals=world_normals,
            coordinate_frame=CoordinateFrame.PROJECTED_CRS,
            units=Units.METERS,
            is_metric=True,
            crs=target_crs,
        )
        geo_pc.compute_bounds()
        return geo_pc
