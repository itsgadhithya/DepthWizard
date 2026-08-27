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
    assert data["state"] == "STATE_C"
    assert data["calibration"]["is_provisional"] is True
    assert data["calibration"]["scale_factor"] == 10.0
    assert "depth_anything_v2" in data["model_name"]
    assert data["device_used"] != ""
    assert data["total_time_ms"] > 0
    assert "depth_inference_ms" in data["timings_ms"]
    assert data["timings_ms"]["depth_inference_ms"] > 0

    # Artifacts registry
    assert "artifacts" in data
    assert "raw_relative_depth_npy" in data["artifacts"]
    assert "relative_depth_visual_png" in data["artifacts"]
    assert "metric_depth_npy" in data["artifacts"]

    req_id = data["request_id"]
    visual_filename = data["artifacts"]["relative_depth_visual_png"]["filename"]
    npy_filename = data["artifacts"]["raw_relative_depth_npy"]["filename"]

    # Test downloading visual PNG
    art_response = test_client.get(f"/api/v1/artifacts/{req_id}/{visual_filename}")
    assert art_response.status_code == 200
    assert art_response.headers["content-type"] == "image/png"
    assert len(art_response.content) > 0

    # Test downloading raw NPY
    npy_response = test_client.get(f"/api/v1/artifacts/{req_id}/{npy_filename}")
    assert npy_response.status_code == 200
    assert len(npy_response.content) > 0


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
