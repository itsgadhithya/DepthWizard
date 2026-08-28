# DepthWizard V2 — Backend API & GIS 3D Terrain Engine

Mission-critical Python API and computational pipeline that processes single optical and GeoTIFF imagery to generate relative depth, calibrated Digital Surface Models (DSM), and persistent 3D terrain meshes with interactive browser and desktop fly-through 3D viewers.

---

## 🌟 Key Capabilities

- **FastAPI Backend**: Clean REST API wrapping DepthWizard V2 with standardized DepthWizard JSON contracts and Swagger UI.
- **Robust Ingestion**: Supports standard optical formats (`.jpg`, `.jpeg`, `.png`, `.tif`) and multi-band / single-band GeoTIFF DEMs.
- **Monocular Depth Engine**: Relative depth estimation via DepthAnything V2 with PyTorch and automatic CUDA/CPU acceleration.
- **High-Precision Matrix Modulation**: High-pass spatial frequency decomposition in `float64` to fuse macro DEM elevation with DepthAnything V2 micro-details.
- **Scientific Integrity**: Zero metric or coordinate fabrication — conceptual `STATE_B` for uncalibrated optical imagery and `STATE_C` for georeferenced/DEM inputs.
- **Complete Artifact System**: Exports raw `.npy` depth arrays, 16-bit PNGs, colorized maps, GeoTIFF DSMs, and binary `.ply` 3D meshes.
- **Dual 3D Viewer Experience**:
  1. **Web 3D Viewer**: Embedded WebGL Three.js viewer accessible directly via browser URL (`/api/v1/viewer/{request_id}`).
  2. **Native Desktop WASD Viewer**: PyVista + VTK fly-through viewer with real-time FPS camera controls and topographic scale bars.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. (Optional) Generate Synthetic Test Data
```bash
python generate_sample_data.py
```

### 3. Start the API Server
```bash
python main.py --host 127.0.0.1 --port 8000
```
Or directly with Uvicorn:
```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System health, loaded model name, device (CUDA/CPU), supported formats |
| `POST` | `/api/v1/process` | Main image processing endpoint (Multipart upload) |
| `GET` | `/api/v1/artifacts/{request_id}/{filename}` | Download generated artifacts (`.npy`, `.png`, `.tif`, `.ply`) |
| `GET` | `/api/v1/viewer/{request_id}` | Interactive WebGL 3D model viewer in the browser |
| `POST` | `/api/v1/viewer/{request_id}/launch-desktop` | Spawn native desktop PyVista WASD fly-through window on host |

---

## 📥 Example API Request

### Uploading an Optical Image (JPEG/PNG)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/process" \
  -F "file=@sample_photo.jpg" \
  -F "modulation_weight=0.35" \
  -F "z_exaggeration=1.0"
```

### Uploading a GeoTIFF with Optional Pre-computed Depth Map
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/process" \
  -F "file=@input/sample_terrain.tif" \
  -F "depth_file=@depthanything_output/sample_depth_map.npy" \
  -F "modulation_weight=0.35" \
  -F "z_exaggeration=1.5"
```

---

## 📋 JSON Response Contract

```json
{
  "request_id": "a1b2c3d4e5f6",
  "state": "STATE_C",
  "relative_depth_available": true,
  "camera_model_available": true,
  "metric_depth_available": true,
  "georeferencing_available": true,
  "dsm_available": true,
  "validation_available": false,
  "model_name": "DepthAnythingV2",
  "device_used": "cpu",
  "total_time_ms": 1420.5,
  "timings_ms": {
    "ingestion_ms": 15.2,
    "metadata_extraction_ms": 4.8,
    "depth_inference_ms": 320.1,
    "camera_modeling_ms": 2.0,
    "3d_reconstruction_ms": 780.4,
    "metric_calibration_ms": 0.0,
    "geospatial_and_dsm_ms": 298.0
  },
  "messages": [
    "Image ingested successfully (512x512, 4 channels).",
    "Depth estimation completed via DepthAnythingV2.",
    "Executing high-precision DEM + DepthAnythingV2 fusion."
  ],
  "warnings": [],
  "metadata": {
    "image_width": 512,
    "image_height": 512,
    "channels": 4,
    "format": "TIF",
    "exif": {},
    "geospatial": {
      "crs": "EPSG:4326",
      "bounds": {
        "left": 77.5946,
        "bottom": 12.9204,
        "right": 77.6458,
        "top": 12.9716
      },
      "transform": [0.0001, 0.0, 77.5946, 0.0, -0.0001, 12.9716],
      "pixel_spacing_m": 11.1,
      "nodata": null
    }
  },
  "camera": {
    "fx": 491.5,
    "fy": 491.5,
    "cx": 256.0,
    "cy": 256.0,
    "fov_deg": 55.0,
    "estimated": true,
    "sensor_width_mm": null,
    "focal_length_mm": null
  },
  "calibration": {
    "method": "geotiff_dem_fusion",
    "scale_factor": 1.0,
    "reference_type": "geotiff_elevation_band",
    "reference_value": 502.4,
    "calibrated": true
  },
  "validation": null,
  "artifacts": {
    "input_image": {
      "filename": "input_image.tif",
      "artifact_type": "input_image",
      "download_url": "/api/v1/artifacts/a1b2c3d4e5f6/input_image.tif",
      "size_bytes": 8388608,
      "media_type": "image/tiff",
      "description": "Original uploaded image"
    },
    "relative_depth": {
      "filename": "relative_depth.npy",
      "artifact_type": "relative_depth",
      "download_url": "/api/v1/artifacts/a1b2c3d4e5f6/relative_depth.npy",
      "size_bytes": 2097280,
      "media_type": "application/octet-stream",
      "description": "Raw relative depth array (float64 numpy format)"
    },
    "dsm": {
      "filename": "fused_dsm.tif",
      "artifact_type": "dsm",
      "download_url": "/api/v1/artifacts/a1b2c3d4e5f6/fused_dsm.tif",
      "size_bytes": 2097152,
      "media_type": "image/tiff",
      "description": "Fused high-resolution Digital Surface Model GeoTIFF"
    },
    "3d_model": {
      "filename": "terrain_3d_mesh.ply",
      "artifact_type": "3d_model",
      "download_url": "/api/v1/artifacts/a1b2c3d4e5f6/terrain_3d_mesh.ply",
      "size_bytes": 12582912,
      "media_type": "application/x-ply",
      "description": "3D terrain surface mesh (Binary PLY with baked RGB vertex colors)"
    }
  }
}
```

---

## 🎮 3D Viewer Access

After processing an image, open:
`http://127.0.0.1:8000/api/v1/viewer/<request_id>`

### Controls (Browser Viewer)
- **Left Click + Drag**: Rotate model
- **Right Click + Drag**: Pan camera
- **Mouse Scroll**: Zoom in / out
- **Reset Camera**: Centers model in viewport
- **Toggle Wireframe**: Inspect triangle mesh surface topology
- **Launch Native PyVista Viewer**: Spawns high-performance desktop PyVista WASD window

---

## 🧪 Running the Tests

Execute the full automated test suite:
```bash
pytest -v tests/test_api.py
```
