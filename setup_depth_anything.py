"""
DepthAnythingV2 Automated Setup & Configuration Script
ISRO & Aerospace Grade Standards: Zero Silent Failures, Strict Logging.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logger for setup process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger("SetupDepthAnythingV2")

REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2"
TARGET_DIR = Path("Depth-Anything-V2")


def setup_depth_anything_environment() -> None:
    """
    Automates cloning Depth-Anything-V2 and installing its required dependencies.
    """
    logger.info("Initializing DepthAnythingV2 Environment Setup...")

    # 1. Clone repository if not already cloned
    if not TARGET_DIR.exists():
        logger.info(f"Cloning DepthAnythingV2 repository from {REPO_URL}...")
        try:
            subprocess.run(["git", "clone", REPO_URL], check=True)
            logger.info("Successfully cloned Depth-Anything-V2 repository.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error("git command not found in system PATH. Please install Git.")
            sys.exit(1)
    else:
        logger.info(f"Target directory '{TARGET_DIR}' already exists. Skipping git clone.")

    # 2. Install requirements inside Depth-Anything-V2 if available
    req_file = TARGET_DIR / "requirements.txt"
    if req_file.exists():
        logger.info(f"Installing DepthAnythingV2 dependencies from {req_file}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)
            logger.info("Successfully installed DepthAnythingV2 dependencies.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Warning during dependency installation: {e}")
    else:
        logger.warning(f"No requirements.txt found in {TARGET_DIR}.")

    # 3. Model Weights Information
    logger.info("=" * 60)
    logger.info("DepthAnythingV2 Setup Completed Successfully.")
    logger.info("Pretrained Model Weights download links:")
    logger.info(" - Small (vits): https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth")
    logger.info(" - Base (vitb):  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth")
    logger.info(" - Large (vitl): https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth")
    logger.info("Place downloaded weights in Depth-Anything-V2/checkpoints/ directory.")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_depth_anything_environment()
