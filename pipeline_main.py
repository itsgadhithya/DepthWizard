"""
Main Pipeline Orchestrator — GeoTIFF + DepthAnythingV2 3D Terrain Fusion.
ISRO / Military-grade: strict typing, ISO logging, zero silent failures.
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional
import numpy as np

from src.config import (
    INPUT_DIR, DEPTHANYTHING_OUTPUT_DIR,
    setup_logger, ensure_directories_exist,
)
from src.geotiff_processor import GeoTIFFProcessor, GeoTIFFProcessorError
from src.depth_fusion      import PrecisionTerrainFuser, FusionPipelineError
from src.mesh_exporter     import MeshExporter, MeshExporterError
from src.viewer_fps        import TerrainFlythroughViewer

logger = setup_logger("MainPipeline")


def _find_first(directory: Path, patterns: list[str]) -> Optional[Path]:
    for p in patterns:
        hits = list(directory.glob(p))
        if hits:
            return hits[0]
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ISRO & Military-Grade 3D GeoTIFF+DepthAnythingV2 Terrain Fusion"
    )
    ap.add_argument("--geotiff",            type=Path,  default=None)
    ap.add_argument("--depth-map",          type=Path,  default=None)
    ap.add_argument("--modulation-weight",  type=float, default=0.35,
                    help="Depth detail injection weight (default 0.35 for building-level detail).")
    ap.add_argument("--z-exaggeration",     type=float, default=1.0,
                    help="Vertical exaggeration factor (e.g. 2.0 makes mountains twice as tall).")
    ap.add_argument("--no-viewer",          action="store_true")
    ap.add_argument("--no-export",          action="store_true",
                    help="Skip PLY export (faster for viewer-only runs).")
    ap.add_argument("--model-name",         type=str,   default="terrain_fused_model")
    return ap.parse_args()


def run_pipeline() -> None:
    t0 = time.perf_counter()
    logger.info("=" * 72)
    logger.info("  MISSION-CRITICAL 3D TERRAIN FUSION PIPELINE — START")
    logger.info("=" * 72)

    args = parse_args()
    ensure_directories_exist()

    # ── Resolve input files ─────────────────────────────────────────────────
    geo_path   = args.geotiff   or _find_first(INPUT_DIR,               ["*.tif", "*.tiff"])
    depth_path = args.depth_map or _find_first(DEPTHANYTHING_OUTPUT_DIR, ["*.npy", "*.png"])

    if not geo_path or not geo_path.exists():
        logger.error(f"No GeoTIFF found in {INPUT_DIR}.  "
                     "Run generate_sample_data.py or place your .tif there.")
        sys.exit(1)

    depth_available = depth_path and depth_path.exists()
    if not depth_available:
        logger.warning("No DepthAnythingV2 map found — running with macro DEM only.")

    logger.info(f"GeoTIFF   : {geo_path}")
    logger.info(f"Depth map : {depth_path or 'N/A'}")
    logger.info(f"Mod weight: {args.modulation_weight}   Z-exag: {args.z_exaggeration}")

    try:
        # -- Step 1 -- GeoTIFF extraction ---
        logger.info("\n--- STEP 1  GeoTIFF extraction (float64) ---")
        proc = GeoTIFFProcessor(geo_path)
        dem, rgb, meta = proc.read_dem_and_texture()

        # Compute real-world pixel spacing in metres from GeoTIFF bounds
        import math
        bounds       = meta["bounds"]
        mid_lat      = math.radians((bounds.top + bounds.bottom) / 2)
        x_span_m     = (bounds.right - bounds.left)  * 111_320 * math.cos(mid_lat)
        y_span_m     = (bounds.top   - bounds.bottom) * 111_320
        h, w         = dem.shape
        px_x_m       = x_span_m / w
        px_y_m       = y_span_m / h
        pixel_spacing = (px_x_m + px_y_m) / 2
        logger.info(f"Pixel spacing ~{pixel_spacing:.1f} m  (lon {px_x_m:.1f}m x lat {px_y_m:.1f}m)")

        # -- Step 2 -- Depth fusion (optional) ---
        if depth_available:
            logger.info("\n--- STEP 2  Depth map fusion (float64 residual modulation) ---")
            fuser = PrecisionTerrainFuser(
                modulation_weight=args.modulation_weight
            )
            depth_rel = fuser.load_depth_anything_map(depth_path, target_shape=dem.shape)
            dem       = fuser.fuse(dem, depth_rel)
        else:
            logger.info("\n--- STEP 2  (skipped - no depth map) ---")

        # -- Step 3 -- Mesh build ---
        logger.info("\n--- STEP 3  3D StructuredGrid construction ---")
        exporter = MeshExporter(
            dem_fused       = dem,
            rgb_texture     = rgb,
            pixel_spacing_m = pixel_spacing,
            z_exaggeration  = args.z_exaggeration,
        )
        grid = exporter.create_structured_grid()

        # Compute elevation extremes for topographic annotation
        elev_extremes = exporter.get_elevation_extremes()
        logger.info(
            f"Elevation extremes: MAX={elev_extremes[0][3]:.1f}m  "
            f"MIN={elev_extremes[1][3]:.1f}m"
        )

        # -- Step 4 -- Export ---
        if not args.no_export:
            logger.info("\n--- STEP 4  Persistent PLY export ---")
            exporter.export_ply(base_name=args.model_name)

        elapsed = time.perf_counter() - t0
        logger.info("=" * 72)
        logger.info(f"  PIPELINE COMPLETE  ({elapsed:.1f} s)")
        logger.info("=" * 72)

        # -- Step 5 -- Interactive viewer ---
        if not args.no_viewer:
            logger.info("\n--- STEP 5  Launching WASD fly-through viewer ---")
            viewer = TerrainFlythroughViewer(
                grid,
                elevation_extremes=elev_extremes,
                pixel_spacing_m=pixel_spacing,
            )
            viewer.start_flythrough()

    except (GeoTIFFProcessorError, FusionPipelineError, MeshExporterError) as e:
        logger.error(f"PIPELINE FAILURE: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"UNEXPECTED FAILURE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
