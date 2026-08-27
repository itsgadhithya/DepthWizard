"""Integration tests for FastAPI HTTP REST endpoints."""

import io
import json
import pytest
from fastapi.testclient import TestClient


def test_api_health_endpoint(test_client: TestClient):
    """Verify /api/v1/health status and model info."""
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_manager" in data


def test_api_models_endpoint(test_client: TestClient):
    """Verify /api/v1/depth/models endpoint returns device and architecture info."""
    response = test_client.get("/api/v1/depth/models")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "device" in data


def test_api_process_plain_image(test_client: TestClient, synthetic_jpeg_bytes):
    """Test image upload and relative depth processing via POST /api/v1/depth/process."""
    files = {"file": ("test.jpg", synthetic_jpeg_bytes, "image/jpeg")}
    response = test_client.post("/api/v1/depth/process", files=files)

    assert response.status_code == 200
    data = response.json()

    # Core depth and provisional metric invariants
    assert data["relative_depth_available"] is True
    assert data["metric_depth_available"] is True
    assert data["dsm_available"] is True
    assert data["dsm_type"] == "local_metric"
    assert data["state"] == "STATE_C"
    assert data["calibration"]["is_provisional"] is True
    assert data["calibration"]["scale_factor"] == 10.0
    assert "depth_anything_v2" in data["model_name"]
    assert data["device_used"] != ""
    assert data["total_time_ms"] > 0
    assert "depth_inference_ms" in data["timings_ms"]
    assert data["timings_ms"]["depth_inference_ms"] > 0

    # Clean JSON checks: ensure internal paths & bulky structures are NOT leaked
    for art_name, art_obj in data["artifacts"].items():
        assert "file_path" not in art_obj, f"Local file path leaked in artifact {art_name}"
    if "metadata" in data and data["metadata"] is not None:
        assert "provenance" not in data["metadata"], "Bulky provenance dictionary leaked in metadata"
    if "calibration" in data and data["calibration"] is not None:
        assert "details" not in data["calibration"], "Internal calibration details dictionary leaked"

    # Artifacts registry
    assert "artifacts" in data
    assert "raw_relative_depth_npy" in data["artifacts"]
    assert "relative_depth_visual_png" in data["artifacts"]
    assert "metric_depth_npy" in data["artifacts"]
    assert "dsm_geotiff" in data["artifacts"]
    assert "dsm_model_glb" in data["artifacts"]
    assert "dsm_npy" in data["artifacts"]
    assert "dsm_visual_png" in data["artifacts"]

    req_id = data["request_id"]
    visual_filename = data["artifacts"]["relative_depth_visual_png"]["filename"]
    npy_filename = data["artifacts"]["raw_relative_depth_npy"]["filename"]
    dsm_filename = data["artifacts"]["dsm_geotiff"]["filename"]
    glb_filename = data["artifacts"]["dsm_model_glb"]["filename"]
    dsm_npy_filename = data["artifacts"]["dsm_npy"]["filename"]

    # Test downloading visual PNG
    art_response = test_client.get(f"/api/v1/artifacts/{req_id}/{visual_filename}")
    assert art_response.status_code == 200
    assert art_response.headers["content-type"] == "image/png"
    assert len(art_response.content) > 0

    # Test downloading raw NPY
    npy_response = test_client.get(f"/api/v1/artifacts/{req_id}/{npy_filename}")
    assert npy_response.status_code == 200
    assert len(npy_response.content) > 0

    # Test downloading DSM GeoTIFF / TIFF
    dsm_response = test_client.get(f"/api/v1/artifacts/{req_id}/{dsm_filename}")
    assert dsm_response.status_code == 200
    assert len(dsm_response.content) > 0

    # Test downloading DSM 3D Surface Model (GLB)
    glb_response = test_client.get(f"/api/v1/artifacts/{req_id}/{glb_filename}")
    assert glb_response.status_code == 200
    assert glb_response.headers["content-type"] == "model/gltf-binary"
    assert len(glb_response.content) >= 12
    assert glb_response.content[:4] == b"glTF"

    # Test downloading DSM NumPy array
    dsm_npy_resp = test_client.get(f"/api/v1/artifacts/{req_id}/{dsm_npy_filename}")
    assert dsm_npy_resp.status_code == 200
    assert len(dsm_npy_resp.content) > 0

    # Test HTML viewer endpoint
    viewer_resp = test_client.get(f"/api/v1/artifacts/{req_id}")
    assert viewer_resp.status_code == 200
    assert "text/html" in viewer_resp.headers["content-type"]
    assert "Local Metric DSM" in viewer_resp.text
    assert "THREE.GLTFLoader" in viewer_resp.text
    assert "elevation-legend" in viewer_resp.text

    # Test dedicated standalone 3D viewer endpoint
    standalone_viewer_resp = test_client.get(f"/api/v1/viewer/{req_id}")
    assert standalone_viewer_resp.status_code == 200
    assert "text/html" in standalone_viewer_resp.headers["content-type"]
    assert "Interactive 3D Terrain Model" in standalone_viewer_resp.text



def test_api_process_with_gcp_calibration(test_client: TestClient, synthetic_jpeg_bytes):
    """Test processing with GCP calibration parameters."""
    files = {"file": ("calib_test.jpg", synthetic_jpeg_bytes, "image/jpeg")}
    calib_params = json.dumps({
        "gcps": [
            {"pixel_u": 15, "pixel_v": 15, "depth_z": 25.0},
            {"pixel_u": 45, "pixel_v": 45, "depth_z": 25.0},
        ]
    })
    form_data = {
        "calibration_mode": "gcp",
        "calibration_params": calib_params,
    }

    response = test_client.post("/api/v1/depth/process", files=files, data=form_data)
    assert response.status_code == 200
    data = response.json()
    assert data["metric_depth_available"]
    assert data["dsm_available"]
    assert data["dsm_type"] == "local_metric"
    assert data["state"] == "STATE_C"


def test_api_validate_endpoint(test_client: TestClient):
    """Test validation endpoint POST /api/v1/depth/validate."""
    payload = {
        "predicted_values": [12.0, 24.5, 36.2],
        "reference_values": [12.1, 24.3, 36.0],
        "tolerance_m": 0.5,
        "reference_source": "Survey Benchmarks",
    }
    response = test_client.post("/api/v1/depth/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["passed"]
    assert data["metrics"]["mae"] < 0.25
