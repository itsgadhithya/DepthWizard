# DepthWizard V2 — API Integration Guide

This guide explains how to integrate the **DepthWizard V2 API** into any external front-end codebase (React, Next.js, Vue, Angular, Svelte, Mobile, or Microservices).

---

## 🚀 Quick Start

### 1. Start the API Server
```bash
# Run locally on port 8000
python main.py --host 0.0.0.0 --port 8000
```

### 2. Interactive Swagger / OpenAPI Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## 📡 API Endpoints Reference

### Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description | Content-Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health & system capability status | `application/json` |
| `POST` | `/process` | Process image / GeoTIFF to 3D & DSM | `multipart/form-data` |
| `GET` | `/sessions/{request_id}` | Retrieve stored session telemetry & manifest | `application/json` |
| `GET` | `/samples` | List available sample datasets | `application/json` |
| `GET` | `/samples/{filename}` | Download sample dataset file | Binary / Octet-Stream |
| `POST` | `/samples/process-sample` | 1-Click Server-Side sample execution | `application/x-www-form-urlencoded` |
| `GET` | `/artifacts/{request_id}/{filename}` | Download generated output artifacts | Binary / Octet-Stream |
| `GET` | `/artifacts/{request_id}/bundle.zip` | Download all artifacts as ZIP package | `application/zip` |
| `GET` | `/viewer/{request_id}` | Standalone WebGL 3D Viewer page | `text/html` |
| `POST` | `/viewer/{request_id}/launch-desktop` | Spawn native PyVista WASD window | `application/json` |


---

## 📥 1. Processing an Image (`POST /api/v1/process`)

Upload an optical image (JPEG, PNG, TIFF) or a GeoTIFF DEM.

### Request Parameters (`multipart/form-data`)

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `file` | File | **Yes** | — | Input image (JPEG, PNG, TIFF, GeoTIFF) |
| `depth_file` | File | No | `null` | Optional pre-computed depth array (`.npy`) |
| `modulation_weight` | Float | No | `0.35` | High-frequency detail injection weight (0.0 – 1.0) |
| `z_exaggeration` | Float | No | `1.0` | Vertical terrain exaggeration multiplier |
| `pixel_spacing_m` | Float | No | `null` | Pixel resolution in meters (auto-detected for GeoTIFF) |
| `focal_length_px` | Float | No | `null` | Pinhole camera focal length in pixels |

### Response Schema (`200 OK`)
```json
{
  "request_id": "3cc69191c5fb",
  "state": "STATE_C",
  "relative_depth_available": true,
  "camera_model_available": true,
  "metric_depth_available": true,
  "georeferencing_available": true,
  "dsm_available": true,
  "validation_available": false,
  "model_name": "DepthAnythingV2-Small",
  "device_used": "cpu",
  "total_time_ms": 2840.5,
  "timings_ms": {
    "ingestion_ms": 42.1,
    "metadata_extraction_ms": 15.3,
    "depth_inference_ms": 1820.0,
    "camera_modeling_ms": 2.1,
    "3d_reconstruction_ms": 650.0,
    "metric_calibration_ms": 0.0,
    "geospatial_and_dsm_ms": 311.0
  },
  "messages": [
    "Loaded GeoTIFF dataset with CRS=EPSG:4326",
    "Depth estimation completed via DepthAnythingV2-Small.",
    "Executing high-precision DEM + DepthAnythingV2 fusion."
  ],
  "warnings": [],
  "metadata": {
    "image_width": 1024,
    "image_height": 768,
    "channels": 3,
    "format": "TIF",
    "exif": {},
    "geospatial": {
      "crs": "EPSG:4326",
      "bounds": { "left": 86.8, "bottom": 27.9, "right": 87.1, "top": 28.1 },
      "pixel_spacing_m": 30.0
    }
  },
  "camera": {
    "fx": 980.5,
    "fy": 980.5,
    "cx": 512.0,
    "cy": 384.0,
    "fov_deg": 55.0,
    "estimated": true
  },
  "calibration": {
    "method": "geotiff_dem_fusion",
    "scale_factor": 1.0,
    "reference_type": "geotiff_elevation_band",
    "reference_value": 5340.2,
    "calibrated": true
  },
  "artifacts": {
    "input_image": {
      "filename": "input_image.tif",
      "artifact_type": "input_image",
      "download_url": "/api/v1/artifacts/3cc69191c5fb/input_image.tif",
      "media_type": "image/tiff"
    },
    "relative_depth": {
      "filename": "relative_depth.npy",
      "artifact_type": "relative_depth",
      "download_url": "/api/v1/artifacts/3cc69191c5fb/relative_depth.npy",
      "media_type": "application/octet-stream"
    },
    "relative_depth_colored": {
      "filename": "relative_depth_colored.png",
      "artifact_type": "visualization",
      "download_url": "/api/v1/artifacts/3cc69191c5fb/relative_depth_colored.png",
      "media_type": "image/png"
    },
    "dsm": {
      "filename": "fused_dsm.tif",
      "artifact_type": "dsm",
      "download_url": "/api/v1/artifacts/3cc69191c5fb/fused_dsm.tif",
      "media_type": "image/tiff"
    },
    "3d_model": {
      "filename": "terrain_3d_mesh.ply",
      "artifact_type": "3d_model",
      "download_url": "/api/v1/artifacts/3cc69191c5fb/terrain_3d_mesh.ply",
      "media_type": "application/x-ply"
    }
  }
}
```

---

## 💻 Frontend Integration Examples

### Option A: Embed Ready 3D Viewer via iframe (Easiest)

You can embed the full interactive Three.js 3D viewer directly in your React/Vue/HTML frontend:

```jsx
export function TerrainViewer({ requestId }) {
  const viewerUrl = `http://localhost:8000/api/v1/viewer/${requestId}`;
  
  return (
    <iframe
      src={viewerUrl}
      title="3D Terrain Viewer"
      style={{
        width: "100%",
        height: "600px",
        border: "none",
        borderRadius: "12px",
      }}
      allow="fullscreen"
    />
  );
}
```

---

### Option B: JavaScript / TypeScript Client SDK

Use this helper class in your frontend project:

```typescript
// depthWizardClient.ts

export interface ProcessOptions {
  modulationWeight?: number;
  zExaggeration?: number;
  pixelSpacingM?: number;
  focalLengthPx?: number;
  depthFile?: File;
}

export class DepthWizardClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000/api/v1") {
    this.baseUrl = baseUrl;
  }

  async checkHealth() {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error("API server is unhealthy");
    return res.json();
  }

  async processImage(file: File, options: ProcessOptions = {}) {
    const formData = new FormData();
    formData.append("file", file);

    if (options.depthFile) formData.append("depth_file", options.depthFile);
    if (options.modulationWeight !== undefined) {
      formData.append("modulation_weight", options.modulationWeight.toString());
    }
    if (options.zExaggeration !== undefined) {
      formData.append("z_exaggeration", options.zExaggeration.toString());
    }
    if (options.pixelSpacingM !== undefined) {
      formData.append("pixel_spacing_m", options.pixelSpacingM.toString());
    }
    if (options.focalLengthPx !== undefined) {
      formData.append("focal_length_px", options.focalLengthPx.toString());
    }

    const res = await fetch(`${this.baseUrl}/process`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Image processing failed");
    }

    return res.json();
  }

  getArtifactUrl(requestId: string, filename: string): string {
    return `${this.baseUrl}/artifacts/${requestId}/${filename}`;
  }

  getViewerUrl(requestId: string): string {
    return `${this.baseUrl}/viewer/${requestId}`;
  }
}
```

---

### Option C: React 3D Mesh Loader (Three.js / PLYLoader)

To render the 3D `.ply` model directly inside your custom Three.js canvas:

```typescript
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";

export function loadTerrainMesh(
  plyUrl: string,
  scene: THREE.Scene,
  onLoaded?: (mesh: THREE.Mesh) => void
) {
  const loader = new PLYLoader();
  
  loader.load(plyUrl, (geometry) => {
    geometry.computeVertexNormals();
    geometry.center();

    const hasColors = geometry.hasAttribute("color");
    const material = new THREE.MeshStandardMaterial({
      vertexColors: hasColors,
      color: hasColors ? 0xffffff : 0x58a6ff,
      roughness: 0.7,
      metalness: 0.05,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2; // Z-up to Three.js Y-up
    scene.add(mesh);

    if (onLoaded) onLoaded(mesh);
  });
}
```

---

## 📡 cURL Examples

### Check Health:
```bash
curl http://localhost:8000/api/v1/health
```

### Upload Image & Process:
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@/path/to/satellite_image.tif" \
  -F "modulation_weight=0.35" \
  -F "z_exaggeration=1.5"
```

### Download 3D Mesh:
```bash
curl -O http://localhost:8000/api/v1/artifacts/3cc69191c5fb/terrain_3d_mesh.ply
```

---

## 🔒 CORS Configuration

CORS is enabled by default in `api/app.py` for all origins (`*`).
To restrict allowed origins for production deployment:
```python
# api/app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
