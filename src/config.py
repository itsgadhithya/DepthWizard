"""
Mission-Critical Configuration & Logging Module
ISRO / Military-grade standards: strict typing, high precision, robust logging.
"""

import sys
import logging
from pathlib import Path
from typing import Final
import numpy as np

# ── Directory Layout ────────────────────────────────────────────────────────
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
INPUT_DIR: Final[Path] = BASE_DIR / "input"
DEPTHANYTHING_OUTPUT_DIR: Final[Path] = BASE_DIR / "depthanything_output"
RESULT_DIR: Final[Path] = BASE_DIR / "result"
ARTIFACTS_DIR: Final[Path] = BASE_DIR / "artifacts"

# ── Spatial Precision ────────────────────────────────────────────────────────
SPATIAL_DTYPE: Final[type] = np.float64

# ── Viewer Performance Settings ───────────────────────────────────────────────
# Max grid resolution fed to the renderer.  Actual DEM is downsampled to this.
# Tweak upward if GPU VRAM allows (RTX cards handle 2048×2048 easily).
MAX_RENDER_WIDTH: Final[int]  = 1536
MAX_RENDER_HEIGHT: Final[int] = 1024


def ensure_directories_exist() -> None:
    """Creates required directory architecture if not present."""
    for d in [INPUT_DIR, DEPTHANYTHING_OUTPUT_DIR, RESULT_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str = "GIS_Pipeline") -> logging.Logger:
    """
    Military-grade logger: ISO-8601 timestamps, dual console+file handlers.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    ensure_directories_exist()
    fh = logging.FileHandler(RESULT_DIR / "pipeline_execution.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


pipeline_logger = setup_logger()
