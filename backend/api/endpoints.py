"""FastAPI route handlers for DepthWizard engine."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import settings
from backend.models.results import ProcessingSummary
from backend.models.calibration import CalibrationReference, CalibrationMethod, GCPPoint, DistanceMeasurement
from backend.models.validation import ValidationReport
from backend.depth.model_loader import model_manager
from backend.validation.metrics import ValidationEngine
from backend.pipeline import SingleImagePipeline
from backend.storage.artifact_manager import ArtifactManager

router = APIRouter(prefix=settings.api_prefix, tags=["Depth & Geometry Engine"])


class ValidationRequest(BaseModel):
    """Validation input request body."""
    predicted_values: List[float]
    reference_values: List[float]
    tolerance_m: float = 1.0
    reference_source: str = "Ground Truth Reference"


@router.post("/depth/process", response_model=ProcessingSummary, summary="Process single aerial image")
async def process_image(
    file: UploadFile = File(..., description="JPEG, PNG, TIFF, or GeoTIFF aerial image"),
    calibration_mode: Optional[str] = Form(default="none", description="Calibration strategy ('none', 'altitude_ground', 'gcp', 'known_distance', 'manual_scale')"),
    calibration_params: Optional[str] = Form(default=None, description="Optional JSON string of calibration measurements"),
    camera_params: Optional[str] = Form(default=None, description="Optional JSON string of camera parameters overrides"),
    target_crs: Optional[str] = Form(default=None, description="Optional target projected CRS (e.g. 'EPSG:32643')"),
    dsm_resolution_m: Optional[float] = Form(default=settings.default_dsm_resolution, description="DSM grid resolution in meters/pixel"),
) -> ProcessingSummary:
    """Execute complete single-image depth, camera model, metric calibration, 3D point cloud, and DSM pipeline."""
    # Read uploaded file bytes
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Clean optional Swagger UI placeholder strings ("string")
    if calibration_params and (calibration_params.strip() == "" or calibration_params.strip().lower() == "string"):
        calibration_params = None
    if camera_params and (camera_params.strip() == "" or camera_params.strip().lower() == "string"):
        camera_params = None
    if target_crs and (target_crs.strip() == "" or target_crs.strip().lower() == "string"):
        target_crs = None
    if calibration_mode and (calibration_mode.strip() == "" or calibration_mode.strip().lower() == "string"):
        calibration_mode = "none"

    # Parse calibration reference
    calib_ref: Optional[CalibrationReference] = None
    calib_dict: Dict[str, Any] = {}
    if calibration_params:
        try:
            calib_dict = json.loads(calibration_params)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON in 'calibration_params'.")

    mode_enum = CalibrationMethod.NONE
    if calibration_mode:
        try:
            mode_enum = CalibrationMethod(calibration_mode.lower())
        except ValueError:
            mode_enum = CalibrationMethod.NONE

    if mode_enum != CalibrationMethod.NONE or calib_dict:
        # Build CalibrationReference
        gcps_list = None
        if "gcps" in calib_dict:
            gcps_list = [GCPPoint(**g) for g in calib_dict["gcps"]]

        dist_list = None
        if "distance_references" in calib_dict:
            dist_list = [DistanceMeasurement(**d) for d in calib_dict["distance_references"]]

        calib_ref = CalibrationReference(
            method=mode_enum,
            camera_altitude_m=calib_dict.get("camera_altitude_m") or calib_dict.get("camera_altitude"),
            ground_elevation_m=calib_dict.get("ground_elevation_m") or calib_dict.get("ground_elevation"),
            flight_height_agl_m=calib_dict.get("flight_height_agl_m") or calib_dict.get("flight_height_agl"),
            gcps=gcps_list,
            distance_references=dist_list,
            manual_scale_factor=calib_dict.get("manual_scale_factor") or calib_dict.get("scale_factor"),
        )

    # Parse camera overrides
    camera_overrides_dict = None
    if camera_params:
        try:
            camera_overrides_dict = json.loads(camera_params)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON in 'camera_params'.")

    try:
        summary = SingleImagePipeline.process(
            image_input=contents,
            filename=file.filename or "image.jpg",
            calibration_ref=calib_ref,
            camera_overrides=camera_overrides_dict,
            target_crs=target_crs,
            dsm_resolution_m=dsm_resolution_m or 0.5,
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")


@router.post("/depth/validate", response_model=ValidationReport, summary="Validate predicted geometry against ground truth")
async def validate_predictions(req: ValidationRequest) -> ValidationReport:
    """Evaluate accuracy and error metrics between predicted values and ground truth references."""
    try:
        report = ValidationEngine.validate_points(
            predicted_values=req.predicted_values,
            reference_values=req.reference_values,
            tolerance_m=req.tolerance_m,
            reference_name=req.reference_source,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation evaluation failed: {str(e)}")


@router.get("/depth/models", summary="List depth models and device status")
async def get_models_info() -> Dict[str, Any]:
    """Retrieve operational details and hardware status for loaded depth models."""
    return model_manager.get_info()


@router.get("/artifacts/{request_id}", summary="View or list output artifact files for request")
@router.get("/depth/artifacts/{request_id}", summary="View or list output artifact files for request")
async def list_artifacts(request_id: str, format: Optional[str] = None):
    """Interactive visual viewer and artifact manager for processed requests."""
    req_dir = (settings.artifacts_dir / request_id).resolve()
    if not req_dir.is_dir() or not str(req_dir).startswith(str(settings.artifacts_dir.resolve())):
        raise HTTPException(status_code=404, detail=f"No artifacts found for request ID '{request_id}'.")

    # If format is explicit JSON, return JSON manifest
    if format == "json":
        return ArtifactManager.list_request_artifacts(request_id)

    # Load summary if available
    summary_path = req_dir / "summary.json"
    summary_data = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    # Find visual image and input image
    visual_img = None
    for v in ["metric_depth_visual.png", "relative_depth_visual.png"]:
        if (req_dir / v).exists():
            visual_img = v
            break

    input_img = None
    for inp in ["input_image.jpg", "input_image.png", "input_image.jpeg", "input_image.tif"]:
        if (req_dir / inp).exists():
            input_img = inp
            break

    state = summary_data.get("state", "STATE_A")
    model_name = summary_data.get("model_name", settings.default_model_name)
    device_used = summary_data.get("device_used", "cpu")
    total_time = summary_data.get("total_time_ms", 0.0)
    inference_time = summary_data.get("timings_ms", {}).get("depth_inference_ms", 0.0)
    artifacts_dict = summary_data.get("artifacts", {})

    # Generate interactive HTML Viewer
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DepthWizard Viewer — {request_id}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-card: rgba(18, 24, 38, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.35);
            --accent-green: #10b981;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            max-width: 1200px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .header {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }}
        .title-group h1 {{
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge-state {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge-device {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .viewer-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            backdrop-filter: blur(12px);
        }}
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .img-container {{
            width: 100%;
            height: 480px;
            background: #05070d;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }}
        .img-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.2s ease;
        }}
        .actions-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .downloads-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }}
        .btn-download {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 12px 16px;
            text-decoration: none;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .btn-download:hover {{
            background: var(--accent);
            border-color: var(--accent);
            box-shadow: 0 0 16px var(--accent-glow);
            transform: translateY(-1px);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
        }}
        .stat-box {{
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 10px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .stat-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; color: #38bdf8; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="title-group">
                <h1>DepthWizard Interactive Viewer</h1>
                <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px;">
                    <span class="badge badge-state">{state}</span>
                    <span class="badge badge-device">{device_used.upper()}</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Request ID: <code>{request_id}</code></span>
                </div>
            </div>
            <div style="display: flex; gap: 10px;">
                <a href="/api/v1/artifacts/{request_id}?format=json" class="btn-download" style="padding: 8px 14px; font-size: 0.8rem;">View Raw JSON Manifest</a>
                <a href="/docs" class="btn-download" style="padding: 8px 14px; font-size: 0.8rem;">API Docs</a>
            </div>
        </div>

        <!-- Viewer Cards -->
        <div class="viewer-grid">
            <!-- Original Input Image -->
            <div class="card">
                <div class="card-title">
                    <span>Input Aerial Image</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">{input_img or 'Source'}</span>
                </div>
                <div class="img-container">
                    {'<img src="/api/v1/artifacts/' + request_id + '/' + input_img + '" alt="Input Image">' if input_img else '<span style="color: var(--text-muted);">No input preview stored</span>'}
                </div>
            </div>

            <!-- DepthAnything V2 Output -->
            <div class="card">
                <div class="card-title">
                    <span>Relative Depth Map (Turbo Colormap)</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">{model_name}</span>
                </div>
                <div class="img-container">
                    {'<img src="/api/v1/artifacts/' + request_id + '/' + visual_img + '" alt="Depth Visualization">' if visual_img else '<span style="color: var(--text-muted);">No depth visualization</span>'}
                </div>
            </div>
        </div>

        <!-- Download & Telemetry Panel -->
        <div class="actions-panel">
            <div class="card-title">
                <span>Download Artifacts</span>
                <span style="font-size: 0.8rem; color: var(--text-muted);">Click any button below to download the corresponding file</span>
            </div>
            <div class="downloads-grid">
                {'<a href="/api/v1/artifacts/' + request_id + '/' + visual_img + '?download=true" class="btn-download">📥 Download Visual PNG</a>' if visual_img else ''}
                <a href="/api/v1/artifacts/{request_id}/raw_relative_depth.npy?download=true" class="btn-download">📥 Download Raw Depth (.npy)</a>
                <a href="/api/v1/artifacts/{request_id}/raw_relative_depth_32f.tif?download=true" class="btn-download">📥 Download 32-bit Float TIFF</a>
                <a href="/api/v1/artifacts/{request_id}/point_cloud.ply?download=true" class="btn-download">📥 Download Point Cloud (.ply)</a>
            </div>

            <div class="card-title" style="margin-top: 8px;">
                <span>Pipeline Telemetry & Diagnostics</span>
            </div>
            <div class="stats-grid">
                <div class="stat-box">
                    <span class="stat-label">Model Name</span>
                    <span class="stat-val">{model_name}</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Device Used</span>
                    <span class="stat-val">{device_used}</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Inference Latency</span>
                    <span class="stat-val">{inference_time:.1f} ms</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Total Processing</span>
                    <span class="stat-val">{total_time:.1f} ms</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


@router.get("/artifacts/{request_id}/{filename}", summary="Retrieve or download output artifact file")
@router.get("/depth/artifacts/{request_id}/{filename}", summary="Retrieve or download output artifact file")
async def get_artifact(request_id: str, filename: str, download: bool = False):
    """Retrieve an artifact file inline for in-browser viewing, or download as an attachment."""
    file_path = ArtifactManager.get_artifact_file(request_id, filename)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found for request '{request_id}'.")

    # Determine media type
    ext = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".npy": "application/octet-stream",
        ".ply": "application/octet-stream",
        ".json": "application/json",
        ".txt": "text/plain",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    if download:
        # User explicitly requested download
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type,
            content_disposition_type="attachment",
        )

    # In-browser view (inline display, no forced download)
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        content_disposition_type="inline",
    )


@router.get("/health", summary="Health check endpoint")
async def health_check() -> Dict[str, Any]:
    """System health check and environmental capabilities verification."""
    return {
        "status": "healthy",
        "service": settings.api_title,
        "version": settings.api_version,
        "model_manager": model_manager.get_info(),
    }
