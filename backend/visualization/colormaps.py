"""Colormap generation for depth and elevation visualizations."""

from typing import Optional, Union
import numpy as np
import matplotlib.cm as cm
from PIL import Image


class DepthColorMapper:
    """Applies scientific colormaps to 2D continuous arrays for visualization.

    IMPORTANT: Visualizations are strictly for display and MUST NEVER be used as computational depth.
    """

    AVAILABLE_COLORMAPS = ["turbo", "viridis", "plasma", "inferno", "magma", "terrain", "gray"]

    @classmethod
    def apply_colormap(
        cls,
        data_2d: np.ndarray,
        colormap_name: str = "turbo",
        valid_mask: Optional[np.ndarray] = None,
        invert: bool = False,
    ) -> np.ndarray:
        """Map a 2D float array to an 8-bit (H, W, 3) RGB image.

        Args:
            data_2d: (H, W) float32 array.
            colormap_name: Matplotlib colormap identifier.
            valid_mask: Optional boolean mask where True indicates valid data.
            invert: If True, reverses colormap direction.

        Returns:
            (H, W, 3) uint8 numpy array.
        """
        arr = data_2d.astype(np.float32)

        if valid_mask is None:
            valid_mask = np.isfinite(arr) & (arr > -9000.0)

        if not np.any(valid_mask):
            return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

        min_val = float(np.nanmin(arr[valid_mask]))
        max_val = float(np.nanmax(arr[valid_mask]))

        # Normalized to [0, 1]
        denom = max_val - min_val if max_val > min_val else 1.0
        norm_arr = np.clip((arr - min_val) / denom, 0.0, 1.0)

        if invert:
            norm_arr = 1.0 - norm_arr

        # Retrieve colormap
        cmap_name = colormap_name.lower() if colormap_name.lower() in cls.AVAILABLE_COLORMAPS else "turbo"
        cmap = getattr(cm, cmap_name, cm.turbo)

        # RGBA float [0, 1]
        rgba = cmap(norm_arr)
        rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)

        # Set invalid pixels to black
        rgb[~valid_mask] = 0

        return rgb

    @classmethod
    def save_image(
        cls,
        rgb_array: np.ndarray,
        file_path: str,
    ) -> str:
        """Save RGB uint8 array to PNG image."""
        img = Image.fromarray(rgb_array)
        img.save(file_path, format="PNG", optimize=True)
        return file_path
