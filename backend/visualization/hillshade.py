"""Analytical hillshade and shaded relief calculation for Digital Surface Models."""

import math
from typing import Optional
import numpy as np


class HillshadeGenerator:
    """Computes analytical shaded relief (hillshade) from 2D elevation models."""

    @classmethod
    def compute_hillshade(
        cls,
        elevation_grid: np.ndarray,
        resolution_m: float = 0.5,
        altitude_deg: float = 45.0,
        azimuth_deg: float = 315.0,
        nodata_value: float = -9999.0,
        z_factor: float = 1.0,
    ) -> np.ndarray:
        """Calculate standard 8-bit shaded relief map.

        Args:
            elevation_grid: 2D float32 array of surface elevations.
            resolution_m: Grid cell resolution in meters.
            altitude_deg: Sun elevation angle above horizon (default 45°).
            azimuth_deg: Sun direction angle clockwise from North (default 315° NW).
            nodata_value: Sentinal void elevation value.
            z_factor: Vertical exaggeration factor.

        Returns:
            (H, W) uint8 grayscale hillshade array [0, 255].
        """
        grid = elevation_grid.astype(np.float32)
        valid_mask = (grid != nodata_value) & np.isfinite(grid)

        # Replace nodata temporarily for finite difference computation
        if np.any(valid_mask):
            mean_val = float(np.nanmean(grid[valid_mask]))
            clean_grid = np.where(valid_mask, grid * z_factor, mean_val)
        else:
            return np.zeros_like(grid, dtype=np.uint8)

        # Spatial gradients (Horn's method or central differences)
        dz_dy, dz_dx = np.gradient(clean_grid, resolution_m, resolution_m)

        # Slope and Aspect
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect_rad = np.arctan2(dz_dy, -dz_dx)

        # Sun position angles in radians
        zenith_rad = math.radians(90.0 - altitude_deg)
        azimuth_math_rad = math.radians(360.0 - azimuth_deg + 90.0) % (2.0 * math.pi)

        # Shading equation: cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)
        shaded = (
            math.cos(zenith_rad) * np.cos(slope_rad)
            + math.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_math_rad - aspect_rad)
        )

        hillshade_8u = np.clip(255.0 * shaded, 0.0, 255.0).astype(np.uint8)
        hillshade_8u[~valid_mask] = 0

        return hillshade_8u

    @classmethod
    def blend_color_and_hillshade(
        cls,
        color_rgb: np.ndarray,
        hillshade_8u: np.ndarray,
        blend_factor: float = 0.5,
    ) -> np.ndarray:
        """Blend colorized elevation map with grayscale hillshade for draped 3D relief."""
        hill_3ch = np.stack([hillshade_8u] * 3, axis=-1).astype(np.float32) / 255.0
        color_float = color_rgb.astype(np.float32)

        # Multiply blending: color * hillshade
        blended = color_float * (hill_3ch * (1.0 - blend_factor) + blend_factor)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)
