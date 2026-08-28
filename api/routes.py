"""
FastAPI Route Definitions for DepthWizard API
Provides health check, image processing, artifact serving, and 3D viewer endpoints.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from src.config import ARTIFACTS_DIR, pipeline_logger
from api.models import DepthWizardResponse, HealthResponse
from api.service import DepthWizardService

logger = pipeline_logger
router = APIRouter(prefix="/api/v1", tags=["DepthWizard V2 API"])
service = DepthWizardService()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & System Status Check",
    description="Returns service health, loaded depth estimation engine, PyTorch device, and supported formats."
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        api_version="v1.0.0",
        depth_model=service.depth_engine.model_name,
        device=service.depth_engine.device,
        supported_formats=["JPEG", "JPG", "PNG", "TIFF", "TIF", "GeoTIFF"],
    )


@router.post(
    "/process",
    response_model=DepthWizardResponse,
    summary="Process Image to Depth, DSM & 3D Model",
    description="Upload an image (optical or GeoTIFF) to compute relative/metric depth, fuse terrain DEMs, and generate 3D models."
)
async def process_image(
    file: UploadFile = File(..., description="Input image file (JPEG, PNG, TIFF, GeoTIFF)"),
    depth_file: Optional[UploadFile] = File(None, description="Optional pre-computed relative depth map (.npy, .png)"),
    modulation_weight: float = Form(0.35, description="High-frequency detail injection weight (default 0.35)"),
    z_exaggeration: float = Form(1.0, description="Vertical terrain exaggeration multiplier (default 1.0)"),
    pixel_spacing_m: Optional[float] = Form(None, description="Pixel resolution in meters (auto-extracted for GeoTIFFs)"),
    focal_length_px: Optional[float] = Form(None, description="Focal length in pixels for pinhole back-projection"),
) -> DepthWizardResponse:
    try:
        image_bytes = await file.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        depth_bytes = None
        depth_filename = None
        if depth_file is not None:
            depth_bytes = await depth_file.read()
            depth_filename = depth_file.filename

        response = service.process_request(
            image_bytes=image_bytes,
            filename=file.filename or "input.jpg",
            depth_bytes=depth_bytes,
            depth_filename=depth_filename,
            modulation_weight=modulation_weight,
            z_exaggeration=z_exaggeration,
            pixel_spacing_m=pixel_spacing_m,
            focal_length_px=focal_length_px,
        )
        return response

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Internal processing failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DepthWizard processing failure: {str(e)}"
        )


@router.get(
    "/artifacts/{request_id}/{filename}",
    summary="Download Generated Artifact",
    description="Retrieves a specific artifact (.npy, .png, .tif, .ply) produced during request execution."
)
async def get_artifact(request_id: str, filename: str):
    # Sanitize request_id and filename to prevent path traversal
    safe_request_id = Path(request_id).name
    safe_filename = Path(filename).name
    artifact_path = ARTIFACTS_DIR / safe_request_id / safe_filename

    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{filename}' not found for request '{request_id}'."
        )

    media_type = service._get_media_type(safe_filename)
    return FileResponse(
        path=artifact_path,
        media_type=media_type,
        filename=safe_filename
    )


@router.get(
    "/viewer/{request_id}",
    response_class=HTMLResponse,
    summary="Interactive 3D Web Viewer",
    description="Opens the WebGL Three.js interactive 3D viewer for the given request ID."
)
async def view_3d_model(request_id: str):
    static_viewer = Path(__file__).resolve().parent.parent / "static" / "viewer.html"
    if not static_viewer.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Viewer template static/viewer.html not found."
        )

    with open(static_viewer, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Pre-inject request_id into HTML
    html_content = html_content.replace(
        "window.location.pathname.split('/').pop()",
        f"'{request_id}'"
    )
    return HTMLResponse(content=html_content)


@router.post(
    "/viewer/{request_id}/launch-desktop",
    summary="Launch Native PyVista WASD Fly-Through Viewer",
    description="Spawns the native desktop PyVista WASD viewer window on the server host."
)
async def launch_desktop_viewer(request_id: str):
    safe_request_id = Path(request_id).name
    mesh_path = ARTIFACTS_DIR / safe_request_id / "terrain_3d_mesh.ply"

    if not mesh_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"3D mesh model not found for request '{request_id}'."
        )

    # Spawn desktop viewer as independent process
    viewer_script = f"""
import pyvista as pv
import trimesh
from pathlib import Path

mesh = trimesh.load(r"{mesh_path}")
pv_mesh = pv.wrap(mesh)

pl = pv.Plotter(title="DepthWizard Native 3D Viewer - Request {request_id}")
pl.set_background("black")
pl.add_mesh(pv_mesh, scalars="RGB", rgb=True, smooth_shading=True)
pl.show()
"""
    try:
        subprocess.Popen([sys.executable, "-c", viewer_script])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "launched", "message": f"Native 3D Viewer opened for request {request_id}"}
        )
    except Exception as e:
        logger.error(f"Failed to launch native viewer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to spawn desktop viewer: {str(e)}"
        )
