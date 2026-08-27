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

    # Check for 3D model, DSM, visual image, and input image
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

    dsm_visual_img = None
    for d in ["dsm_visual.png", "dsm_color_relief.png", "dsm_hillshade.png"]:
        if (req_dir / d).exists():
            dsm_visual_img = d
            break

    has_dsm_glb = (req_dir / "dsm_model.glb").exists()
    has_dsm_obj = (req_dir / "dsm_model.obj").exists()
    has_dsm_tif = (req_dir / "dsm.tif").exists()
    has_dsm_npy = (req_dir / "dsm_elevation_meters.npy").exists()
    has_metric_npy = (req_dir / "metric_depth_meters.npy").exists()
    has_ply = (req_dir / "point_cloud.ply").exists()
    has_web_json = (req_dir / "point_cloud_preview.json").exists()

    # Read DSM and Mesh metadata if available
    mesh_meta = {}
    dsm_meta_file = req_dir / "dsm_metadata.json"
    if dsm_meta_file.exists():
        try:
            with open(dsm_meta_file, "r") as f:
                dsm_meta_raw = json.load(f)
                mesh_meta = dsm_meta_raw.get("mesh", {})
        except Exception:
            mesh_meta = {}

    mesh_w = mesh_meta.get("width_m", 0.0)
    mesh_l = mesh_meta.get("length_m", 0.0)
    mesh_zmin = mesh_meta.get("height_min_m", 0.0)
    mesh_zmax = mesh_meta.get("height_max_m", 0.0)
    mesh_zrange = mesh_meta.get("height_range_m", 0.0)
    mesh_verts = mesh_meta.get("vertex_count", 0)
    mesh_tris = mesh_meta.get("triangle_count", 0)

    state = summary_data.get("state", "STATE_A")
    dsm_type = summary_data.get("dsm_type") or ("georeferenced_metric" if state == "STATE_D" else ("local_metric" if (has_dsm_tif or state == "STATE_C") else None))
    crs_info = summary_data.get("crs") or summary_data.get("target_crs")
    model_name = summary_data.get("model_name", settings.default_model_name)
    device_used = summary_data.get("device_used", "cpu")
    total_time = summary_data.get("total_time_ms", 0.0)
    inference_time = summary_data.get("timings_ms", {}).get("depth_inference_ms", 0.0)
    artifacts_dict = summary_data.get("artifacts", {})

    dsm_badge_html = ""
    if dsm_type == "georeferenced_metric":
        crs_label = f"Georeferenced Metric DSM ({crs_info})" if crs_info else "Georeferenced Metric DSM"
        dsm_badge_html = f'<span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">{crs_label}</span>'
    elif dsm_type == "local_metric":
        dsm_badge_html = '<span class="badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3);">Local Metric DSM</span>'

    # Generate interactive HTML Viewer
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DepthWizard 3D DSM Terrain Viewer — {request_id}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- Three.js + OrbitControls + GLTFLoader -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <style>
        :root {{
            --bg-primary: #070a13;
            --bg-card: rgba(15, 23, 42, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.35);
            --accent-green: #10b981;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            max-width: 1440px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }}
        .header {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 18px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }}
        .title-group h1 {{
            font-size: 1.35rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #34d399, #c084fc);
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
        
        /* 3D Viewport Hero Section */
        .viewport-section {{
            background: var(--bg-card);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 16px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
            display: flex;
            flex-direction: column;
        }}
        .viewport-header {{
            padding: 14px 20px;
            background: rgba(10, 15, 30, 0.92);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .viewport-title {{
            font-size: 1.05rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #38bdf8;
        }}
        .viewport-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 0, 0, 0.4);
            padding: 5px 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.8rem;
        }}
        .preset-btn {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-secondary);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .preset-btn:hover, .preset-btn.active {{
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }}
        .control-btn {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .control-btn:hover {{
            background: var(--accent);
            border-color: var(--accent);
        }}
        .control-btn.active {{
            background: #2563eb;
            border-color: #60a5fa;
        }}
        #viewport-3d {{
            width: 100%;
            height: 680px;
            background: radial-gradient(circle at center, #0e172a 0%, #020617 100%);
            position: relative;
            cursor: grab;
            outline: none;
        }}
        #viewport-3d:active {{ cursor: grabbing; }}
        
        /* 3D Telemetry Overlay HUD */
        .hud-overlay {{
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(8, 14, 28, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 12px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 5px;
            pointer-events: none;
            backdrop-filter: blur(8px);
            z-index: 10;
        }}
        .hud-row {{
            display: flex;
            justify-content: space-between;
            gap: 16px;
        }}
        .hud-label {{ color: var(--text-muted); }}
        .hud-val {{ color: #38bdf8; font-weight: 600; }}

        /* Flythrough Crosshair & Instructions */
        #fly-overlay {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(8, 14, 28, 0.92);
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 16px;
            padding: 24px 32px;
            text-align: center;
            z-index: 25;
            cursor: pointer;
            backdrop-filter: blur(12px);
            display: none;
        }}
        #fly-overlay h2 {{ font-size: 1.2rem; color: #38bdf8; margin-bottom: 8px; }}
        #fly-overlay p {{ font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 14px; }}
        .key-badge {{
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #f8fafc;
        }}
        
        /* Elevation Legend Overlay */
        .legend-overlay {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(8, 14, 28, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            backdrop-filter: blur(8px);
            z-index: 10;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        }}
        .legend-title {{
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: center;
        }}
        .legend-body {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-bar {{
            width: 14px;
            height: 130px;
            border-radius: 6px;
            background: linear-gradient(to top, 
                #1b4332 0%, 
                #40916c 18%, 
                #d4a373 40%, 
                #9c6644 65%, 
                #78828c 85%, 
                #f8fafc 100%);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .legend-ticks {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 130px;
            font-size: 0.7rem;
            color: var(--text-secondary);
        }}
        .legend-ticks span {{ line-height: 1; }}

        /* Instructions Pill */
        .hud-instructions {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            padding: 6px 18px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            pointer-events: none;
            backdrop-filter: blur(6px);
            z-index: 10;
        }}

        .viewer-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
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
            height: 380px;
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
        .btn-primary {{
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            border-color: #3b82f6;
            font-weight: 600;
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
        #loading-spinner {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            color: var(--text-secondary);
            font-size: 0.9rem;
            z-index: 20;
        }}
        .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top-color: #38bdf8;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="title-group">
                <h1>DepthWizard 3D Terrain & DSM Surface Viewer</h1>
                <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px; flex-wrap: wrap;">
                    <span class="badge badge-state">{state}</span>
                    {dsm_badge_html}
                    <span class="badge badge-device">{device_used.upper()}</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Request ID: <code>{request_id}</code></span>
                </div>
            </div>
            <div style="display: flex; gap: 10px;">
                <a href="/api/v1/artifacts/{request_id}?format=json" class="btn-download" style="padding: 8px 14px; font-size: 0.8rem;">Clean JSON Manifest</a>
                <a href="/docs" class="btn-download" style="padding: 8px 14px; font-size: 0.8rem;">API Docs</a>
            </div>
        </div>

        <!-- 3D WebGL DSM Viewport -->
        <div class="viewport-section" id="viewport-card">
            <div class="viewport-header">
                <div class="viewport-title">
                    <span>🏔️ Interactive 3D Terrain Model (Digital Surface Model)</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted); font-weight: normal;">(Physical Metric Scale: 1 unit = 1 metre)</span>
                </div>
                <div class="viewport-controls">
                    <!-- Color Mode Selector -->
                    <div class="control-group">
                        <span style="color: var(--text-muted); font-size: 0.75rem;">Color:</span>
                        <button class="preset-btn active" id="btn-col-rgb" title="Realistic Aerial Photographic Texture">🎨 RGB Texture</button>
                        <button class="preset-btn" id="btn-col-elevation" title="Scientific Topographic Elevation Colormap">🏔️ Elevation</button>
                        <button class="preset-btn" id="btn-col-plasma" title="High-Contrast Relief Plasma">⚡ Plasma</button>
                    </div>

                    <!-- Camera Mode Selector -->
                    <div class="control-group">
                        <span style="color: var(--text-muted); font-size: 0.75rem;">Camera:</span>
                        <button class="preset-btn active" id="btn-cam-orbit" title="360 Orbit & Turntable Mode">🔄 Orbit</button>
                        <button class="preset-btn" id="btn-cam-fly" title="WASD First-Person Flythrough Mode">🚀 Flythrough</button>
                    </div>

                    <!-- Vertical Exaggeration -->
                    <div class="control-group">
                        <label for="exaggeration-slider" style="color: var(--text-secondary); cursor: pointer;">Exaggeration:</label>
                        <input type="range" id="exaggeration-slider" min="1.0" max="10.0" step="0.5" value="1.0" style="width: 70px; cursor: pointer;">
                        <span id="exaggeration-val" style="font-family: 'JetBrains Mono'; color: #38bdf8; min-width: 30px;">1.0x</span>
                        <div style="display: flex; gap: 3px; margin-left: 2px;">
                            <button class="preset-btn active" data-exag="1.0">1x</button>
                            <button class="preset-btn" data-exag="2.0">2x</button>
                            <button class="preset-btn" data-exag="5.0">5x</button>
                            <button class="preset-btn" data-exag="10.0">10x</button>
                        </div>
                    </div>

                    <!-- Background Theme -->
                    <div class="control-group">
                        <span style="color: var(--text-muted); font-size: 0.75rem;">Theme:</span>
                        <button class="preset-btn active" id="btn-bg-dark">Dark</button>
                        <button class="preset-btn" id="btn-bg-space">Space</button>
                        <button class="preset-btn" id="btn-bg-fog">Fog</button>
                    </div>

                    <!-- Action Buttons -->
                    <button class="control-btn" id="btn-fit-view" title="Auto-Frame Terrain">🎯 Fit View</button>
                    <button class="control-btn" id="btn-top-view" title="Top-Down Nadir View">🧭 Top View</button>
                    <button class="control-btn" id="btn-wireframe" title="Toggle Wireframe Mesh">📐 Wireframe</button>
                    <button class="control-btn" id="btn-autorotate" title="Toggle 360 Turntable Rotation">🔄 Auto-Rotate</button>
                    <button class="control-btn" id="btn-fullscreen" title="Toggle Fullscreen View">⛶ Fullscreen</button>
                </div>
            </div>

            <!-- Viewport Canvas Container -->
            <div id="viewport-3d" tabindex="0">
                <div id="loading-spinner">
                    <div class="spinner"></div>
                    <span>Constructing 3D Terrain Surface Mesh...</span>
                </div>

                <!-- Flythrough Click to Start Overlay -->
                <div id="fly-overlay">
                    <h2>🚀 First-Person Flythrough</h2>
                    <p>Click anywhere inside this box to lock the cursor and fly over the terrain.</p>
                    <div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 12px;">
                        <div><span class="key-badge">W A S D</span> Move</div>
                        <div><span class="key-badge">Q / E</span> Down / Up</div>
                        <div><span class="key-badge">Mouse</span> Look</div>
                        <div><span class="key-badge">Esc</span> Release</div>
                    </div>
                </div>

                <!-- Live Telemetry HUD -->
                <div class="hud-overlay" id="hud-panel">
                    <div class="hud-row"><span class="hud-label">Model:</span><span class="hud-val">Digital Surface Model</span></div>
                    <div class="hud-row"><span class="hud-label">Type:</span><span class="hud-val">{dsm_type or 'Local Metric DSM'}</span></div>
                    <div class="hud-row"><span class="hud-label">CRS:</span><span class="hud-val">{crs_info or 'Local (Camera Frame)'}</span></div>
                    <div class="hud-row"><span class="hud-label">Units:</span><span class="hud-val">metres</span></div>
                    <div class="hud-row"><span class="hud-label">Elevation Min:</span><span class="hud-val" id="hud-zmin">{mesh_zmin:.1f} m</span></div>
                    <div class="hud-row"><span class="hud-label">Elevation Max:</span><span class="hud-val" id="hud-zmax">{mesh_zmax:.1f} m</span></div>
                    <div class="hud-row"><span class="hud-label">Elevation Range:</span><span class="hud-val" id="hud-zrange">{mesh_zrange:.1f} m</span></div>
                    <div class="hud-row"><span class="hud-label">Width (X):</span><span class="hud-val" id="hud-w">{mesh_w:.1f} m</span></div>
                    <div class="hud-row"><span class="hud-label">Length (Y):</span><span class="hud-val" id="hud-l">{mesh_l:.1f} m</span></div>
                    <div class="hud-row"><span class="hud-label">Triangles:</span><span class="hud-val" id="hud-tris">{mesh_tris:,}</span></div>
                    <div class="hud-row"><span class="hud-label">Exaggeration:</span><span class="hud-val" id="hud-exag">1.0x</span></div>
                    <div class="hud-row"><span class="hud-label">FPS:</span><span class="hud-val" id="hud-fps">60 fps</span></div>
                </div>

                <!-- Elevation Legend -->
                <div class="legend-overlay" id="elevation-legend">
                    <div class="legend-title">Elevation (m)</div>
                    <div class="legend-body">
                        <div class="legend-bar"></div>
                        <div class="legend-ticks">
                            <span id="leg-max">{mesh_zmax:.1f} m</span>
                            <span id="leg-75">{(mesh_zmin + mesh_zrange * 0.75):.1f} m</span>
                            <span id="leg-50">{(mesh_zmin + mesh_zrange * 0.50):.1f} m</span>
                            <span id="leg-25">{(mesh_zmin + mesh_zrange * 0.25):.1f} m</span>
                            <span id="leg-min">{mesh_zmin:.1f} m</span>
                        </div>
                    </div>
                </div>

                <!-- Navigation Guide -->
                <div class="hud-instructions" id="instructions-pill">
                    🖱️ Left Click: 360° Orbit &nbsp;|&nbsp; Right Click: Pan &nbsp;|&nbsp; Scroll: Zoom &nbsp;|&nbsp; Double Click: Focus
                </div>
            </div>
        </div>

        <!-- 2D Previews Grid -->
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

            <!-- Depth Map Output -->
            <div class="card">
                <div class="card-title">
                    <span>{'Metric Depth (Meters)' if state in ['STATE_C', 'STATE_D'] else 'Relative Depth Map'}</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">{model_name}</span>
                </div>
                <div class="img-container">
                    {'<img src="/api/v1/artifacts/' + request_id + '/' + visual_img + '" alt="Depth Visualization">' if visual_img else '<span style="color: var(--text-muted);">No depth visualization</span>'}
                </div>
            </div>

            <!-- DSM 2D Visualization Card -->
            {f'''<div class="card">
                <div class="card-title">
                    <span>{"Georeferenced DSM" if dsm_type == "georeferenced_metric" else "Local Metric DSM"} (Relief Image)</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">DSM Visualization (2D PNG)</span>
                </div>
                <div class="img-container">
                    <img src="/api/v1/artifacts/{request_id}/{dsm_visual_img}" alt="DSM Surface Visualization">
                </div>
            </div>''' if dsm_visual_img else ''}
        </div>

        <!-- Download & Telemetry Panel -->
        <div class="actions-panel">
            <div class="card-title">
                <span>Download Generated 3D & Geospatial Artifacts</span>
                <span style="font-size: 0.8rem; color: var(--text-muted);">Click any button below to download the corresponding file</span>
            </div>
            <div class="downloads-grid">
                {'<a href="/api/v1/artifacts/' + request_id + '/dsm_model.glb?download=true" class="btn-download btn-primary">📦 Download 3D DSM Model (.glb)</a>' if has_dsm_glb else ''}
                {'<a href="/api/v1/artifacts/' + request_id + '/dsm_model.obj?download=true" class="btn-download">🏛️ Download 3D Mesh (.obj)</a>' if has_dsm_obj else ''}
                {'<a href="/api/v1/artifacts/' + request_id + '/dsm.tif?download=true" class="btn-download">🗺️ Download DSM Raster (.tif)</a>' if has_dsm_tif else ''}
                {'<a href="/api/v1/artifacts/' + request_id + '/dsm_elevation_meters.npy?download=true" class="btn-download">🔢 Download DSM Array (.npy)</a>' if has_dsm_npy else ''}
                {'<a href="/api/v1/artifacts/' + request_id + '/dsm_visual.png?download=true" class="btn-download">🖼️ Download DSM Visual (.png)</a>' if dsm_visual_img else ''}
                {'<a href="/api/v1/artifacts/' + request_id + '/metric_depth_meters.npy?download=true" class="btn-download">📥 Metric Depth Array (.npy)</a>' if has_metric_npy else ''}
                {'<a href="/api/v1/artifacts/' + request_id + '/point_cloud.ply?download=true" class="btn-download">☁️ 3D Point Cloud (.ply)</a>' if has_ply else ''}
                <a href="/api/v1/artifacts/{request_id}/raw_relative_depth.npy?download=true" class="btn-download">📥 Raw Relative Depth (.npy)</a>
            </div>

            <div class="card-title" style="margin-top: 8px;">
                <span>Pipeline Diagnostics & Model Parameters</span>
            </div>
            <div class="stats-grid">
                <div class="stat-box">
                    <span class="stat-label">Model Architecture</span>
                    <span class="stat-val">{model_name}</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">DSM Status</span>
                    <span class="stat-val" style="font-size: 0.85rem;">{dsm_type or 'None'}</span>
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

    <!-- Three.js 3D Viewport Script -->
    <script>
        const container = document.getElementById('viewport-3d');
        const spinner = document.getElementById('loading-spinner');
        const flyOverlay = document.getElementById('fly-overlay');
        const instructionsPill = document.getElementById('instructions-pill');
        const glbUrl = '/api/v1/artifacts/{request_id}/dsm_model.glb';

        let scene, camera, renderer, controls;
        let dsmMesh = null;
        let dsmGroup = null;
        let sunLight, fillLight, ambientLight, hemiLight;
        let modelCenter = new THREE.Vector3();
        let modelSize = new THREE.Vector3();
        let autoRotate = false;
        let wireframeMode = false;
        let defaultCameraPos = new THREE.Vector3();
        let defaultTarget = new THREE.Vector3();

        // Color buffers
        let activeColorMode = 'rgb'; // 'rgb', 'elevation', 'plasma'
        let cachedRgbColors = null;
        let cachedElevationColors = null;
        let cachedPlasmaColors = null;
        let activeGeometry = null;

        // Camera control modes: 'orbit' vs 'fly'
        let cameraMode = 'orbit';
        let isPointerLocked = false;
        let flyMoveSpeed = 30.0;
        const flyKeys = {{}};
        const flyEuler = new THREE.Euler(0, 0, 0, 'YXZ');
        const flyDelta = new THREE.Vector3();

        // FPS counter
        let lastTime = performance.now();
        let frameCount = 0;
        let fpsAccum = 0;

        function init() {{
            const width = container.clientWidth;
            const height = container.clientHeight;

            // 1. Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x070b14);

            // 2. Camera
            camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 50000);

            // 3. Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: "high-performance" }});
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.15;
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);

            // 4. OrbitControls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.screenSpacePanning = true;
            controls.maxPolarAngle = Math.PI / 2 + 0.15;
            controls.minDistance = 0.1;
            controls.maxDistance = 50000;

            // 5. Lighting Setup: Sun Directional + Sky Ambient + Opposite Fill
            ambientLight = new THREE.AmbientLight(0xdde8ff, 0.55);
            scene.add(ambientLight);

            sunLight = new THREE.DirectionalLight(0xfff5e4, 1.35);
            sunLight.position.set(300, 500, 300);
            scene.add(sunLight);

            fillLight = new THREE.DirectionalLight(0x88aacc, 0.45);
            fillLight.position.set(-300, 200, -300);
            scene.add(fillLight);

            hemiLight = new THREE.HemisphereLight(0xffffff, 0x1a243b, 0.4);
            scene.add(hemiLight);

            // DSM Container Group
            dsmGroup = new THREE.Group();
            scene.add(dsmGroup);

            // Resize handler
            window.addEventListener('resize', onWindowResize);

            // Setup UI Controls & Fly Listeners
            setupControls();

            // Load Model
            loadGLB();

            // Render loop
            animate();
        }}

        // Colormap generators
        function makeElevationRGB(t) {{
            // 6-stop natural terrain ramp: valley green -> emerald -> ochre -> ridge brown -> slate rock -> snow white
            const stops = [
                {{ t: 0.00, c: [27/255, 67/255, 50/255] }},
                {{ t: 0.18, c: [64/255, 145/255, 108/255] }},
                {{ t: 0.40, c: [212/255, 163/255, 115/255] }},
                {{ t: 0.65, c: [156/255, 102/255, 68/255] }},
                {{ t: 0.85, c: [120/255, 130/255, 140/255] }},
                {{ t: 1.00, c: [248/255, 250/255, 252/255] }}
            ];
            t = Math.max(0.0, Math.min(1.0, t));
            for (let i = 0; i < stops.length - 1; i++) {{
                if (t >= stops[i].t && t <= stops[i+1].t) {{
                    const span = stops[i+1].t - stops[i].t;
                    const f = (t - stops[i].t) / (span || 1e-4);
                    return [
                        stops[i].c[0] + f * (stops[i+1].c[0] - stops[i].c[0]),
                        stops[i].c[1] + f * (stops[i+1].c[1] - stops[i].c[1]),
                        stops[i].c[2] + f * (stops[i+1].c[2] - stops[i].c[2]),
                    ];
                }}
            }}
            return [1.0, 1.0, 1.0];
        }}

        function makePlasmaRGB(t) {{
            t = Math.max(0.0, Math.min(1.0, t));
            return [
                Math.sin(t * Math.PI * 0.8) * 0.9 + 0.1,
                Math.sin(t * Math.PI * 1.5) * 0.8 + 0.1,
                Math.cos(t * Math.PI * 0.9) * 0.9 + 0.1
            ];
        }}

        function loadGLB() {{
            const loader = new THREE.GLTFLoader();
            loader.load(
                glbUrl,
                function (gltf) {{
                    spinner.style.display = 'none';
                    dsmMesh = gltf.scene;

                    // Traverse mesh to configure materials and precompute color buffers
                    dsmMesh.traverse(function (child) {{
                        if (child.isMesh && child.geometry) {{
                            activeGeometry = child.geometry;
                            child.material.side = THREE.DoubleSide;
                            child.material.roughness = 0.65;
                            child.material.metalness = 0.02;
                            child.material.vertexColors = true;

                            // Cache original RGB colors
                            const colorAttr = activeGeometry.getAttribute('color');
                            if (colorAttr) {{
                                cachedRgbColors = new Float32Array(colorAttr.array);
                            }}

                            // Compute elevation buffer
                            const posAttr = activeGeometry.getAttribute('position');
                            const count = posAttr.count;
                            cachedElevationColors = new Float32Array(count * 3);
                            cachedPlasmaColors = new Float32Array(count * 3);

                            let minZ = Infinity, maxZ = -Infinity;
                            for (let i = 0; i < count; i++) {{
                                const z = posAttr.getZ(i);
                                if (z < minZ) minZ = z;
                                if (z > maxZ) maxZ = z;
                            }}
                            const rangeZ = (maxZ - minZ) || 1.0;

                            for (let i = 0; i < count; i++) {{
                                const z = posAttr.getZ(i);
                                const normZ = (z - minZ) / rangeZ;

                                const [er, eg, eb] = makeElevationRGB(normZ);
                                cachedElevationColors[i * 3] = er;
                                cachedElevationColors[i * 3 + 1] = eg;
                                cachedElevationColors[i * 3 + 2] = eb;

                                const [pr, pg, pb] = makePlasmaRGB(normZ);
                                cachedPlasmaColors[i * 3] = pr;
                                cachedPlasmaColors[i * 3 + 1] = pg;
                                cachedPlasmaColors[i * 3 + 2] = pb;
                            }}

                            if (!cachedRgbColors) {{
                                cachedRgbColors = new Float32Array(cachedElevationColors);
                            }}
                        }}
                    }});

                    // Compute Bounding Box & Center
                    const box = new THREE.Box3().setFromObject(dsmMesh);
                    box.getSize(modelSize);
                    box.getCenter(modelCenter);

                    // Add to scene group
                    dsmGroup.add(dsmMesh);

                    // Update HUD telemetry with actual calculated dimensions
                    document.getElementById('hud-w').textContent = modelSize.x.toFixed(1) + ' m';
                    document.getElementById('hud-l').textContent = modelSize.y.toFixed(1) + ' m';
                    document.getElementById('hud-zrange').textContent = modelSize.z.toFixed(1) + ' m';
                    document.getElementById('hud-zmin').textContent = box.min.z.toFixed(1) + ' m';
                    document.getElementById('hud-zmax').textContent = box.max.z.toFixed(1) + ' m';

                    // Update Elevation Legend tick values
                    const zMin = box.min.z;
                    const zMax = box.max.z;
                    const zRange = zMax - zMin;
                    document.getElementById('leg-max').textContent = zMax.toFixed(1) + ' m';
                    document.getElementById('leg-75').textContent = (zMin + zRange * 0.75).toFixed(1) + ' m';
                    document.getElementById('leg-50').textContent = (zMin + zRange * 0.50).toFixed(1) + ' m';
                    document.getElementById('leg-25').textContent = (zMin + zRange * 0.25).toFixed(1) + ' m';
                    document.getElementById('leg-min').textContent = zMin.toFixed(1) + ' m';

                    // Automatic Framing Calculation (fills 85% of viewport)
                    frameModel(box);
                }},
                undefined,
                function (error) {{
                    console.error('Error loading GLB:', error);
                    spinner.innerHTML = '<span style="color:#ef4444;">⚠️ 3D Terrain GLB model loading error</span>';
                }}
            );
        }}

        function frameModel(box) {{
            const size = new THREE.Vector3();
            const center = new THREE.Vector3();
            box.getSize(size);
            box.getCenter(center);

            const maxDim = Math.max(size.x, size.y, size.z, 1.0);
            const fov = camera.fov * (Math.PI / 180);
            const cameraDistance = (maxDim / 2) / Math.tan(fov / 2) * 1.20;

            // Position lights relative to terrain dimensions
            sunLight.position.set(center.x + cameraDistance * 0.9, center.y + cameraDistance * 1.6, center.z + cameraDistance * 0.9);
            fillLight.position.set(center.x - cameraDistance * 0.9, center.y + cameraDistance * 0.6, center.z - cameraDistance * 0.9);

            // Oblique 45-degree elevated perspective
            defaultCameraPos.set(
                center.x + cameraDistance * 0.68,
                center.y + cameraDistance * 0.72,
                center.z + cameraDistance * 0.68
            );
            defaultTarget.copy(center);

            camera.position.copy(defaultCameraPos);
            camera.near = Math.max(0.01, maxDim / 1000);
            camera.far = maxDim * 50;
            camera.updateProjectionMatrix();

            controls.target.copy(center);
            controls.update();
            flyEuler.setFromQuaternion(camera.quaternion, 'YXZ');
        }}

        function setColorMode(mode) {{
            activeColorMode = mode;
            if (!activeGeometry) return;
            const attr = activeGeometry.getAttribute('color');
            if (!attr) return;

            let source = cachedRgbColors;
            if (mode === 'elevation') source = cachedElevationColors;
            else if (mode === 'plasma') source = cachedPlasmaColors;

            if (source) {{
                attr.array.set(source);
                attr.needsUpdate = true;
            }}

            document.getElementById('btn-col-rgb').classList.toggle('active', mode === 'rgb');
            document.getElementById('btn-col-elevation').classList.toggle('active', mode === 'elevation');
            document.getElementById('btn-col-plasma').classList.toggle('active', mode === 'plasma');
        }}

        function setBackground(theme) {{
            if (theme === 'dark') {{
                scene.background = new THREE.Color(0x070b14);
                scene.fog = null;
            }} else if (theme === 'space') {{
                scene.background = new THREE.Color(0x000004);
                scene.fog = null;
            }} else if (theme === 'fog') {{
                scene.background = new THREE.Color(0x0a1224);
                scene.fog = new THREE.FogExp2(0x0a1224, 0.005);
            }}

            document.getElementById('btn-bg-dark').classList.toggle('active', theme === 'dark');
            document.getElementById('btn-bg-space').classList.toggle('active', theme === 'space');
            document.getElementById('btn-bg-fog').classList.toggle('active', theme === 'fog');
        }}

        function setCameraMode(mode) {{
            cameraMode = mode;
            document.getElementById('btn-cam-orbit').classList.toggle('active', mode === 'orbit');
            document.getElementById('btn-cam-fly').classList.toggle('active', mode === 'fly');

            if (mode === 'fly') {{
                controls.enabled = false;
                flyOverlay.style.display = 'block';
                instructionsPill.innerHTML = '🚀 Flythrough Mode: Click canvas to lock mouse. <span class="key-badge">WASD</span> move, <span class="key-badge">Q/E</span> elevation, <span class="key-badge">Esc</span> exit';
            }} else {{
                controls.enabled = true;
                flyOverlay.style.display = 'none';
                instructionsPill.innerHTML = '🖱️ Left Click: 360° Orbit &nbsp;|&nbsp; Right Click: Pan &nbsp;|&nbsp; Scroll: Zoom &nbsp;|&nbsp; Double Click: Focus';
                if (document.pointerLockElement === container) {{
                    document.exitPointerLock();
                }}
            }}
        }}

        function setupControls() {{
            // Color Mode buttons
            document.getElementById('btn-col-rgb').addEventListener('click', () => setColorMode('rgb'));
            document.getElementById('btn-col-elevation').addEventListener('click', () => setColorMode('elevation'));
            document.getElementById('btn-col-plasma').addEventListener('click', () => setColorMode('plasma'));

            // Camera Mode buttons
            document.getElementById('btn-cam-orbit').addEventListener('click', () => setCameraMode('orbit'));
            document.getElementById('btn-cam-fly').addEventListener('click', () => setCameraMode('fly'));

            // Background Theme buttons
            document.getElementById('btn-bg-dark').addEventListener('click', () => setBackground('dark'));
            document.getElementById('btn-bg-space').addEventListener('click', () => setBackground('space'));
            document.getElementById('btn-bg-fog').addEventListener('click', () => setBackground('fog'));

            // Vertical Exaggeration Slider & Preset buttons
            const slider = document.getElementById('exaggeration-slider');
            const exagVal = document.getElementById('exaggeration-val');
            const hudExag = document.getElementById('hud-exag');
            const presetBtns = document.querySelectorAll('.preset-btn[data-exag]');

            function setExaggeration(factor) {{
                slider.value = factor;
                exagVal.textContent = factor.toFixed(1) + 'x';
                hudExag.textContent = factor.toFixed(1) + 'x';

                presetBtns.forEach(btn => {{
                    btn.classList.toggle('active', parseFloat(btn.getAttribute('data-exag')) === factor);
                }});

                if (dsmMesh) {{
                    dsmMesh.scale.set(1.0, 1.0, factor);
                }}
            }}

            slider.addEventListener('input', function (e) {{
                setExaggeration(parseFloat(e.target.value));
            }});

            presetBtns.forEach(btn => {{
                btn.addEventListener('click', function () {{
                    setExaggeration(parseFloat(this.getAttribute('data-exag')));
                }});
            }});

            // Fit View
            document.getElementById('btn-fit-view').addEventListener('click', function () {{
                if (dsmMesh) {{
                    const box = new THREE.Box3().setFromObject(dsmMesh);
                    frameModel(box);
                }}
            }});

            // Top View
            document.getElementById('btn-top-view').addEventListener('click', function () {{
                if (dsmMesh) {{
                    const box = new THREE.Box3().setFromObject(dsmMesh);
                    const center = new THREE.Vector3();
                    const size = new THREE.Vector3();
                    box.getCenter(center);
                    box.getSize(size);
                    const maxDim = Math.max(size.x, size.y);
                    const fov = camera.fov * (Math.PI / 180);
                    const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.18;

                    camera.position.set(center.x, center.y + dist, center.z);
                    controls.target.copy(center);
                    controls.update();
                }}
            }});

            // Wireframe
            const btnWireframe = document.getElementById('btn-wireframe');
            btnWireframe.addEventListener('click', function () {{
                wireframeMode = !wireframeMode;
                btnWireframe.classList.toggle('active', wireframeMode);
                if (dsmMesh) {{
                    dsmMesh.traverse(function (child) {{
                        if (child.isMesh && child.material) {{
                            child.material.wireframe = wireframeMode;
                        }}
                    }});
                }}
            }});

            // Auto-Rotate
            const btnRotate = document.getElementById('btn-autorotate');
            btnRotate.addEventListener('click', function () {{
                autoRotate = !autoRotate;
                btnRotate.classList.toggle('active', autoRotate);
                controls.autoRotate = autoRotate;
                controls.autoRotateSpeed = 2.0;
            }});

            // Fullscreen Toggle
            const btnFullscreen = document.getElementById('btn-fullscreen');
            const viewportCard = document.getElementById('viewport-card');
            btnFullscreen.addEventListener('click', function () {{
                if (!document.fullscreenElement) {{
                    viewportCard.requestFullscreen().catch(err => {{
                        console.error('Fullscreen error:', err);
                    }});
                    btnFullscreen.textContent = '✕ Exit Fullscreen';
                }} else {{
                    document.exitFullscreen();
                    btnFullscreen.textContent = '⛶ Fullscreen';
                }}
            }});

            document.addEventListener('fullscreenchange', function () {{
                if (!document.fullscreenElement) {{
                    btnFullscreen.textContent = '⛶ Fullscreen';
                }}
                onWindowResize();
            }});

            // Pointer Lock & Flythrough listeners
            container.addEventListener('click', () => {{
                if (cameraMode === 'fly' && !isPointerLocked) {{
                    container.requestPointerLock();
                }}
            }});

            flyOverlay.addEventListener('click', () => {{
                container.requestPointerLock();
            }});

            document.addEventListener('pointerlockchange', () => {{
                isPointerLocked = (document.pointerLockElement === container);
                if (cameraMode === 'fly') {{
                    flyOverlay.style.display = isPointerLocked ? 'none' : 'block';
                }}
            }});

            document.addEventListener('mousemove', (e) => {{
                if (cameraMode === 'fly' && isPointerLocked) {{
                    const sens = 0.0022;
                    flyEuler.y -= e.movementX * sens;
                    flyEuler.x -= e.movementY * sens;
                    flyEuler.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, flyEuler.x));
                    camera.quaternion.setFromEuler(flyEuler);
                }}
            }});

            document.addEventListener('keydown', (e) => {{ flyKeys[e.code] = true; }});
            document.addEventListener('keyup', (e) => {{ flyKeys[e.code] = false; }});
        }}

        function onWindowResize() {{
            const width = container.clientWidth;
            const height = container.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }}

        function animate() {{
            requestAnimationFrame(animate);
            const now = performance.now();
            const dt = (now - lastTime) / 1000.0;
            lastTime = now;

            // FPS Counter
            frameCount++;
            fpsAccum += dt;
            if (fpsAccum >= 0.5) {{
                document.getElementById('hud-fps').textContent = Math.round(frameCount / fpsAccum) + ' fps';
                frameCount = 0;
                fpsAccum = 0;
            }}

            // Flythrough Movement handling
            if (cameraMode === 'fly' && isPointerLocked) {{
                const speed = flyMoveSpeed * dt;
                flyDelta.set(0, 0, 0);
                if (flyKeys['KeyW'] || flyKeys['ArrowUp']) flyDelta.z -= speed;
                if (flyKeys['KeyS'] || flyKeys['ArrowDown']) flyDelta.z += speed;
                if (flyKeys['KeyA'] || flyKeys['ArrowLeft']) flyDelta.x -= speed;
                if (flyKeys['KeyD'] || flyKeys['ArrowRight']) flyDelta.x += speed;
                if (flyKeys['KeyE'] || flyKeys['Space']) flyDelta.y += speed;
                if (flyKeys['KeyQ'] || flyKeys['ShiftLeft']) flyDelta.y -= speed;
                flyDelta.applyQuaternion(camera.quaternion);
                camera.position.add(flyDelta);
            }} else if (cameraMode === 'orbit') {{
                controls.update();
            }}

            renderer.render(scene, camera);
        }}

        // Run on load
        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


@router.get("/viewer/{request_id}", summary="Dedicated Standalone 3D DSM Terrain Viewport")
@router.get("/viewer/{request_id}/3d", summary="Dedicated Standalone 3D DSM Terrain Viewport")
async def get_standalone_terrain_viewer(request_id: str):
    """Serve the dedicated interactive 3D terrain viewport for a request ID."""
    return await list_artifacts(request_id)


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
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "text/plain",
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
