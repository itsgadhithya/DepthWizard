"""
Synthetic GeoTIFF & Depth Anything Data Generator
Generates high-precision test datasets for end-to-end pipeline verification.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import cv2

from src.config import INPUT_DIR, DEPTHANYTHING_OUTPUT_DIR, SPATIAL_DTYPE, setup_logger

logger = setup_logger("SampleDataGen")


def generate_fractal_terrain(width: int = 512, height: int = 512) -> np.ndarray:
    """Generates realistic synthetic mountain terrain elevation (DEM) using Perlin-like noise."""
    x = np.linspace(0, 4 * np.pi, width, dtype=SPATIAL_DTYPE)
    y = np.linspace(0, 4 * np.pi, height, dtype=SPATIAL_DTYPE)
    xx, yy = np.meshgrid(x, y)

    # Multi-octave sinusoidal mountain ridges
    macro_dem = (
        150.0 * np.sin(xx * 0.5) * np.cos(yy * 0.5)
        + 80.0 * np.sin(xx * 1.2) * np.sin(yy * 0.8)
        + 40.0 * np.cos(xx * 2.5) * np.cos(yy * 2.2)
        + 500.0  # Base elevation offset (meters above sea level)
    )
    return macro_dem


def generate_sample_dataset() -> None:
    """Generates sample GeoTIFF and relative depth map."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEPTHANYTHING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    width, height = 512, 512
    logger.info(f"Generating synthetic terrain dataset ({width}x{height})...")

    # 1. Generate DEM elevation band (float64)
    dem_data = generate_fractal_terrain(width, height)

    # 2. Generate RGB orthophoto bands (Viridis / Satellite terrain palette)
    dem_norm = cv2.normalize(dem_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
    rgb_colored = cv2.applyColorMap(dem_norm, cv2.COLORMAP_DEEPGREEN)
    rgb_colored = cv2.cvtColor(rgb_colored, cv2.COLOR_BGR2RGB)

    r_band = rgb_colored[:, :, 0]
    g_band = rgb_colored[:, :, 1]
    b_band = rgb_colored[:, :, 2]

    # 3. Save multi-band GeoTIFF (Bands 1-3: RGB, Band 4: DEM Elevation)
    geotiff_path = INPUT_DIR / "sample_terrain.tif"
    transform = from_origin(77.5946, 12.9716, 0.0001, 0.0001)  # ISRO Bengaluru coordinates

    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=rasterio.float64,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(r_band.astype(np.float64), 1)
        dst.write(g_band.astype(np.float64), 2)
        dst.write(b_band.astype(np.float64), 3)
        dst.write(dem_data, 4)

    logger.info(f"Generated sample GeoTIFF at: {geotiff_path}")

    # 4. Generate high-frequency micro depth map (DepthAnythingV2 relative output)
    x = np.linspace(0, 20 * np.pi, width, dtype=SPATIAL_DTYPE)
    y = np.linspace(0, 20 * np.pi, height, dtype=SPATIAL_DTYPE)
    xx, yy = np.meshgrid(x, y)

    # Micro-relief high frequency ripples and features
    micro_depth = 0.5 + 0.3 * np.sin(xx) * np.cos(yy) + 0.2 * np.random.uniform(-0.1, 0.1, (height, width))
    micro_depth = np.clip(micro_depth, 0.0, 1.0)

    # Save as .npy array and 16-bit PNG
    npy_path = DEPTHANYTHING_OUTPUT_DIR / "sample_depth_map.npy"
    png_path = DEPTHANYTHING_OUTPUT_DIR / "sample_depth_map.png"

    np.save(npy_path, micro_depth)
    cv2.imwrite(str(png_path), (micro_depth * 65535).astype(np.uint16))

    logger.info(f"Generated sample DepthAnythingV2 outputs at: {npy_path} and {png_path}")
    logger.info("Sample dataset generation complete.")


if __name__ == "__main__":
    generate_sample_dataset()
