"""3D back-projection engine converting depth maps to 3D point clouds."""

from typing import Optional, Union, Tuple
import numpy as np

from backend.models.depth import RelativeDepthMap, MetricDepthMap
from backend.models.camera import CameraIntrinsics
from backend.models.geometry import PointCloud3D, CoordinateFrame, Units


class DepthBackProjector:
    """Back-projects 2D pixels with depth into 3D Cartesian point clouds."""

    @classmethod
    def backproject(
        cls,
        depth_map: Union[RelativeDepthMap, MetricDepthMap],
        intrinsics: CameraIntrinsics,
        image_rgb: Optional[np.ndarray] = None,
        compute_normals: bool = True,
        subsample_step: int = 1,
    ) -> PointCloud3D:
        """Backproject a depth map into a PointCloud3D in the camera coordinate frame.

        Args:
            depth_map: RelativeDepthMap or MetricDepthMap.
            intrinsics: CameraIntrinsics parameters.
            image_rgb: Optional (H, W, 3) uint8 image for vertex colors.
            compute_normals: Whether to compute local surface normal vectors.
            subsample_step: Step size for grid subsampling (1 = full dense resolution).

        Returns:
            PointCloud3D with points, colors, normals, and coordinate frame.
        """
        depth_arr = depth_map.array
        h, w = depth_arr.shape[:2]

        # Generate pixel grid coordinates
        u_coords = np.arange(0, w, subsample_step, dtype=np.float32)
        v_coords = np.arange(0, h, subsample_step, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        z_grid = depth_arr[::subsample_step, ::subsample_step]

        # Valid depth mask (Z > 0 and finite)
        valid_mask = np.isfinite(z_grid) & (z_grid > 1e-4)

        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy

        # Pinhole backprojection
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x_grid = (u_grid - cx) * z_grid / fx
        y_grid = (v_grid - cy) * z_grid / fy

        # Reshape to (N, 3)
        pts_x = x_grid[valid_mask]
        pts_y = y_grid[valid_mask]
        pts_z = z_grid[valid_mask]
        points_3d = np.stack([pts_x, pts_y, pts_z], axis=-1).astype(np.float32)

        # Colors
        if image_rgb is not None:
            rgb_sub = image_rgb[::subsample_step, ::subsample_step]
            colors_3d = rgb_sub[valid_mask].astype(np.uint8)
        else:
            # Default gray
            colors_3d = np.full((len(points_3d), 3), 180, dtype=np.uint8)

        # Normals computation
        normals_3d: Optional[np.ndarray] = None
        if compute_normals and len(points_3d) > 0:
            # Surface gradient cross-product on regular grid
            dz_du = np.gradient(z_grid, axis=1)
            dz_dv = np.gradient(z_grid, axis=0)

            # Unnormalized normal vector: (-dz/du * fx, -dz/dv * fy, 1) or (-fx * dz/du, -fy * dz/dv, Z)
            nx = -dz_du * fx
            ny = -dz_dv * fy
            nz = np.ones_like(z_grid, dtype=np.float32)

            norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
            nx /= norm
            ny /= norm
            nz /= norm

            nx_valid = nx[valid_mask]
            ny_valid = ny[valid_mask]
            nz_valid = nz[valid_mask]
            normals_3d = np.stack([nx_valid, ny_valid, nz_valid], axis=-1).astype(np.float32)

        is_metric = depth_map.is_metric
        units = Units.METERS if is_metric else Units.RELATIVE

        pc = PointCloud3D(
            points=points_3d,
            colors=colors_3d,
            normals=normals_3d,
            coordinate_frame=CoordinateFrame.CAMERA,
            units=units,
            is_metric=is_metric,
            crs=None,
        )
        pc.compute_bounds()
        return pc

    # Alias for API backwards compatibility
    backproject_to_point_cloud = backproject
