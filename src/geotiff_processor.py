"""
GeoTIFF Extraction Module
Handles single-band DEM and multi-band RGB GeoTIFFs with float64 precision.
Generates perceptually-accurate terrain colormap when no RGB bands exist.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import rasterio
import cv2

from src.config import pipeline_logger, SPATIAL_DTYPE, MAX_RENDER_WIDTH, MAX_RENDER_HEIGHT


class GeoTIFFProcessorError(Exception):
    pass


# ── Elevation-to-color lookup (realistic terrain palette) ────────────────────
# Breakpoints: (elevation_m, R, G, B)
_TERRAIN_PALETTE = np.array([
    [0,     0,   80, 140],   # deep water
    [1,    50,  150,  80],   # sea level / coast (green)
    [300,  90,  140,  60],   # lowland grass
    [800, 120,  110,  50],   # hills / shrubland
    [1800,130,  100,  75],   # highland forest
    [3000,160,  140, 110],   # alpine / barren
    [4500,200,  190, 175],   # snow line
    [6000,230,  225, 220],   # snow
    [9000,255,  255, 255],   # peak / glacier
], dtype=np.float64)


def _elevation_to_rgb(dem: np.ndarray) -> np.ndarray:
    """
    Maps float64 DEM elevation values to realistic RGB terrain colours.
    Uses piecewise linear interpolation across the _TERRAIN_PALETTE breakpoints.
    Returns uint8 (H, W, 3) array.
    """
    h, w = dem.shape
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    elev_pts = _TERRAIN_PALETTE[:, 0]
    r_pts    = _TERRAIN_PALETTE[:, 1]
    g_pts    = _TERRAIN_PALETTE[:, 2]
    b_pts    = _TERRAIN_PALETTE[:, 3]

    rgb[..., 0] = np.interp(dem, elev_pts, r_pts)
    rgb[..., 1] = np.interp(dem, elev_pts, g_pts)
    rgb[..., 2] = np.interp(dem, elev_pts, b_pts)

    # Subtle shading from surface normals for realism
    gy, gx = np.gradient(dem)
    slope = np.arctan(np.sqrt(gx**2 + gy**2)) / (np.pi / 2)   # [0..1]
    shade = (1.0 - 0.35 * slope)[..., np.newaxis]
    rgb = np.clip(rgb * shade, 0, 255).astype(np.uint8)
    return rgb


def _downsample(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bicubic downsample preserving float64 for DEMs, uint8 for images."""
    if arr.shape[:2] == (target_h, target_w):
        return arr
    if arr.dtype == np.uint8:
        return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    # float64 DEM — use float32 for cv2 then restore
    f32 = arr.astype(np.float32)
    if len(f32.shape) == 2:
        out = cv2.resize(f32, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        out = cv2.resize(f32, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return out.astype(SPATIAL_DTYPE)


class GeoTIFFProcessor:
    """
    Reads DEM elevation and generates RGB texture from any GeoTIFF.
    Automatically downsamples to MAX_RENDER_WIDTH × MAX_RENDER_HEIGHT
    so the viewer always runs at interactive frame rates.
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise GeoTIFFProcessorError(f"GeoTIFF not found: {self.filepath}")
        pipeline_logger.info(f"GeoTIFFProcessor: {self.filepath.name}")

    def read_dem_and_texture(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns:
            dem   – float64 2D array of absolute elevation (metres)
            rgb   – uint8  (H, W, 3) terrain colour texture
            meta  – spatial metadata dict
        """
        with rasterio.open(self.filepath) as src:
            meta: Dict[str, Any] = {
                "crs":       src.crs.to_string() if src.crs else "EPSG:4326",
                "transform": src.transform,
                "bounds":    src.bounds,
                "count":     src.count,
                "width":     src.width,
                "height":    src.height,
                "nodata":    src.nodata,
            }
            pipeline_logger.debug(f"Raster meta: {meta}")

            nodata = src.nodata

            if src.count >= 3:
                # Multi-band: treat first 3 as RGB, last band as DEM (or derive)
                r = src.read(1).astype(SPATIAL_DTYPE)
                g = src.read(2).astype(SPATIAL_DTYPE)
                b = src.read(3).astype(SPATIAL_DTYPE)
                if src.count >= 4:
                    dem = src.read(4).astype(SPATIAL_DTYPE)
                else:
                    # Derive rough luminance DEM for geometry only
                    dem = (0.2989 * r + 0.5870 * g + 0.1140 * b)
                rgb_raw = np.stack([r, g, b], axis=-1)
                # Normalise to uint8 if values exceed 255
                if rgb_raw.max() > 255:
                    rgb_raw = (rgb_raw / rgb_raw.max() * 255)
                rgb_raw = rgb_raw.astype(np.uint8)

            elif src.count == 1:
                # Pure single-band DEM — generate realistic terrain colours
                dem = src.read(1).astype(SPATIAL_DTYPE)
                if nodata is not None:
                    dem[dem == nodata] = np.nan
                # Fill any remaining NaN with median
                if np.isnan(dem).any():
                    dem = np.where(np.isnan(dem), np.nanmedian(dem), dem)
                rgb_raw = _elevation_to_rgb(dem)
            else:
                raise GeoTIFFProcessorError(f"Unsupported band count: {src.count}")

            # Handle nodata in DEM
            if nodata is not None:
                mask = (dem == nodata) | np.isinf(dem)
                if mask.any():
                    dem[mask] = np.nanmedian(dem[~mask])
            dem = np.nan_to_num(dem, nan=np.nanmedian(dem))

        # ── Downsample to interactive resolution ────────────────────────────
        orig_h, orig_w = dem.shape
        scale = min(MAX_RENDER_WIDTH / orig_w, MAX_RENDER_HEIGHT / orig_h, 1.0)
        tgt_w = int(orig_w * scale)
        tgt_h = int(orig_h * scale)

        if scale < 1.0:
            pipeline_logger.info(
                f"Downsampling {orig_w}x{orig_h} -> {tgt_w}x{tgt_h} "
                f"(scale={scale:.3f}) for GPU real-time performance."
            )
            dem     = _downsample(dem, tgt_h, tgt_w)
            rgb_raw = _downsample(rgb_raw, tgt_h, tgt_w)

        pipeline_logger.info(
            f"GeoTIFF loaded: DEM {dem.shape} dtype={dem.dtype} "
            f"elev=[{dem.min():.1f}m, {dem.max():.1f}m]  "
            f"RGB {rgb_raw.shape}"
        )
        return dem, rgb_raw, meta
