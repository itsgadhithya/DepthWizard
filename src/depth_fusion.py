"""
Spatial Data Fusion Module
Fuses absolute macro DEM (GeoTIFF) with relative DepthAnythingV2 micro-detail maps.
All operations in float64.

High-fidelity mode
------------------
The Gaussian high-pass filter sigma controls which spatial frequencies are
treated as "micro-detail":

  sigma=3.0  → removes features > ~3 pixels wide (kills buildings/houses)
  sigma=0.8  → retains features > ~0.8 pixels wide (preserves rooftops/walls)

Default is now sigma=0.8 so man-made structures are faithfully retained.
modulation_weight=0.35 increases their vertical contribution.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from src.config import pipeline_logger, SPATIAL_DTYPE


class FusionPipelineError(Exception):
    pass


class PrecisionTerrainFuser:
    """
    Injects high-frequency micro terrain detail from DepthAnythingV2 into
    the absolute elevation DEM via Gaussian residual modulation.

    Parameters
    ----------
    modulation_weight : float
        Alpha factor.  Higher = stronger depth detail injection.
        Recommended: 0.25-0.50 for building-level detail.
    highpass_sigma : float
        Gaussian blur sigma for the high-pass filter.
        LOWER = finer structures retained.
        Recommended: 0.5-1.5 for man-made structures; 3.0+ for smooth terrain only.
    """

    def __init__(self, modulation_weight: float = 0.35, highpass_sigma: float = 0.8) -> None:
        self.alpha = float(modulation_weight)
        self.sigma = float(highpass_sigma)
        pipeline_logger.info(
            f"TerrainFuser: alpha={self.alpha}  sigma={self.sigma}  "
            f"(sigma<1.5 = high-fidelity building/structure mode)"
        )

    def load_depth_anything_map(self, depth_path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
        """Load .npy or image depth map, normalise to [0,1], resample to target_shape."""
        depth_path = Path(depth_path)
        if not depth_path.exists():
            raise FusionPipelineError(f"Depth map not found: {depth_path}")

        pipeline_logger.info(f"Loading depth map: {depth_path.name}")
        if depth_path.suffix.lower() == ".npy":
            raw = np.load(depth_path).astype(SPATIAL_DTYPE)
        else:
            img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FusionPipelineError(f"Cannot read image: {depth_path}")
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            raw = img.astype(SPATIAL_DTYPE)

        # Normalise to [0, 1]
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)

        # Bicubic resample to match downsampled DEM
        th, tw = target_shape
        if norm.shape != (th, tw):
            pipeline_logger.info(f"Resampling depth {norm.shape} -> {target_shape}")
            norm = cv2.resize(norm.astype(np.float32), (tw, th),
                              interpolation=cv2.INTER_CUBIC).astype(SPATIAL_DTYPE)
        return norm

    def fuse(self, dem_macro: np.ndarray, depth_relative: np.ndarray) -> np.ndarray:
        """
        Z_fused = Z_macro + alpha * std(Z_macro) * (depth - GaussBlur(depth))

        The high-pass residual isolates fine features.  With sigma=0.8,
        rooftop edges, walls, and road curbs ~1-3 pixels wide are preserved.
        """
        if dem_macro.shape != depth_relative.shape:
            raise FusionPipelineError(
                f"Shape mismatch: DEM {dem_macro.shape} vs Depth {depth_relative.shape}"
            )
        dem64   = dem_macro.astype(SPATIAL_DTYPE)
        depth64 = depth_relative.astype(SPATIAL_DTYPE)

        std_dem = np.std(dem64)
        if std_dem < 1e-6:
            std_dem = 1.0

        high_pass  = depth64 - gaussian_filter(depth64, sigma=self.sigma)
        modulation = self.alpha * std_dem * high_pass
        dem_fused  = dem64 + modulation

        pipeline_logger.info(
            f"Fusion done: [{dem64.min():.1f}m, {dem64.max():.1f}m] -> "
            f"[{dem_fused.min():.1f}m, {dem_fused.max():.1f}m]  "
            f"(mod +/-{np.abs(modulation).max():.2f}m)"
        )
        return dem_fused
