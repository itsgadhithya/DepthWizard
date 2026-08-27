"""Image loading and format validation with GeoTIFF awareness."""

import io
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image

# Disable PIL decompression bomb limit for ultra-high-resolution aerial / GeoTIFF imagery
Image.MAX_IMAGE_PIXELS = None


class ImageReader:
    """Robust image ingestion supporting JPEG, PNG, TIFF, and GeoTIFF."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    @classmethod
    def read_image(
        cls,
        image_input: Union[str, Path, bytes, io.BytesIO],
    ) -> Tuple[np.ndarray, str, dict]:
        """Load an image into a normalized RGB uint8 numpy array and format info.

        Args:
            image_input: File path or raw bytes.

        Returns:
            Tuple of:
                - RGB image as numpy.ndarray (H, W, 3) uint8
                - format_name: "JPEG", "PNG", "TIFF", or "GeoTIFF"
                - raw_info: Dictionary containing image format properties and DEM array if applicable.
        """
        is_geotiff = False
        is_dem = False
        raw_info = {}
        dem_raw: Optional[np.ndarray] = None
        nodata_val: Optional[float] = None

        # 1. Attempt GeoTIFF and DEM inspection via rasterio if available
        rasterio_handled = False
        try:
            import rasterio
            from rasterio.io import MemoryFile

            src_context = None
            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.exists():
                    raise FileNotFoundError(f"Input image does not exist at {path}")
                if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
                    raise ValueError(
                        f"Unsupported image extension '{path.suffix}'. Supported: {cls.SUPPORTED_EXTENSIONS}"
                    )
                src_context = rasterio.open(path)
                raw_info["filename"] = path.name
            else:
                if isinstance(image_input, bytes):
                    mem_bytes = image_input
                else:
                    image_input.seek(0)
                    mem_bytes = image_input.read()
                raw_info["filename"] = "uploaded_image"
                memfile = MemoryFile(mem_bytes)
                src_context = memfile.open()

            with src_context as src:
                if src.crs is not None or (src.driver and src.driver.upper() == "GTIFF"):
                    is_geotiff = True
                    band_count = src.count
                    dtype_str = src.dtypes[0] if len(src.dtypes) > 0 else "uint8"
                    nodata_val = float(src.nodata) if src.nodata is not None else None

                    # Check if single band elevation raster (DEM / DSM)
                    if band_count == 1:
                        dem_raw = src.read(1).astype(np.float32)
                        valid_mask = np.isfinite(dem_raw)
                        if nodata_val is not None:
                            valid_mask = valid_mask & (dem_raw != nodata_val)

                        # Confirm DEM characteristics (single-band float/int with elevation variation)
                        if dtype_str.startswith("float") or dtype_str.startswith("int") or dtype_str.startswith("uint16"):
                            is_dem = True
                            raw_info["is_dem"] = True
                            raw_info["dem_array"] = dem_raw
                            raw_info["nodata"] = nodata_val

                        # Generate RGB visualization for DepthAnything V2 inference
                        if np.any(valid_mask):
                            v_min = float(np.percentile(dem_raw[valid_mask], 2))
                            v_max = float(np.percentile(dem_raw[valid_mask], 98))
                            span = max(1e-6, v_max - v_min)
                            norm_2d = np.clip((dem_raw - v_min) / span, 0.0, 1.0)
                            norm_8u = (norm_2d * 255.0).astype(np.uint8)
                            image_np = np.stack([norm_8u, norm_8u, norm_8u], axis=-1)
                        else:
                            image_np = np.zeros((src.height, src.width, 3), dtype=np.uint8)

                        raw_info["width"] = src.width
                        raw_info["height"] = src.height
                        raw_info["is_geotiff"] = True
                        raw_info["format"] = "GeoTIFF"
                        rasterio_handled = True

                    elif band_count >= 3:
                        # Optical RGB GeoTIFF (Orthophoto)
                        r = src.read(1)
                        g = src.read(2)
                        b = src.read(3)
                        # Normalize 16-bit to 8-bit if needed
                        if r.dtype == np.uint16 or g.dtype == np.uint16:
                            r = (np.clip(r / 256.0, 0, 255)).astype(np.uint8)
                            g = (np.clip(g / 256.0, 0, 255)).astype(np.uint8)
                            b = (np.clip(b / 256.0, 0, 255)).astype(np.uint8)
                        else:
                            r, g, b = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

                        image_np = np.stack([r, g, b], axis=-1)
                        raw_info["width"] = src.width
                        raw_info["height"] = src.height
                        raw_info["is_geotiff"] = True
                        raw_info["is_dem"] = False
                        raw_info["format"] = "GeoTIFF"
                        rasterio_handled = True

        except Exception:
            # Fall back to PIL loader
            rasterio_handled = False

        if not rasterio_handled:
            # Standard PIL fallback (JPEG, PNG, standard TIFF)
            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                pil_img = Image.open(path)
                raw_info["filename"] = path.name
            else:
                if isinstance(image_input, bytes):
                    stream = io.BytesIO(image_input)
                else:
                    image_input.seek(0)
                    stream = image_input
                pil_img = Image.open(stream)
                raw_info["filename"] = "uploaded_image"

            img_format = pil_img.format or "UNKNOWN"
            if img_format.upper() in {"TIFF", "TIF"}:
                if hasattr(pil_img, "tag_v2") and pil_img.tag_v2:
                    geotiff_keys = {34735, 33550, 33922, 34264}
                    if any(k in pil_img.tag_v2 for k in geotiff_keys):
                        is_geotiff = True
                format_name = "GeoTIFF" if is_geotiff else "TIFF"
            else:
                format_name = img_format.upper()

            if pil_img.mode != "RGB":
                if pil_img.mode == "RGBA":
                    bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                    bg.paste(pil_img, mask=pil_img.split()[3])
                    pil_img = bg
                elif pil_img.mode in ("I", "I;16", "F"):
                    arr = np.array(pil_img, dtype=np.float32)
                    dem_raw = arr
                    is_dem = True
                    arr_norm = ((arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-8) * 255.0).astype(np.uint8)
                    pil_img = Image.fromarray(arr_norm).convert("RGB")
                    raw_info["dem_array"] = dem_raw
                    raw_info["is_dem"] = True
                else:
                    pil_img = pil_img.convert("RGB")

            image_np = np.array(pil_img, dtype=np.uint8)
            raw_info["width"] = pil_img.width
            raw_info["height"] = pil_img.height
            raw_info["is_geotiff"] = is_geotiff
            raw_info["is_dem"] = is_dem
            raw_info["format"] = format_name

        return image_np, raw_info.get("format", "JPEG"), raw_info

