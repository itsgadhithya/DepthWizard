# DepthWizard — Depth & Metric Geometry Backend Engine

Production-quality modular Python depth, 3D geometry, and Digital Surface Model (DSM) backend engine implementing DepthAnything V2 inference, camera modeling, metric calibration, 3D back-projection, geospatial projection, and accuracy validation.

Adheres strictly to [`DEPTHWIZARD_DEPTH_ENGINE_SPEC.md`](DEPTHWIZARD_DEPTH_ENGINE_SPEC.md) and [`DEPTHWIZARD_ASSUMPTIONS.md`](DEPTHWIZARD_ASSUMPTIONS.md).

---

## 1. Core Principles & Scientific Integrity

1. **Strict Distinction of Representations**:
   - **Relative Depth**: Dimensionless, model-space disparity/depth from DepthAnything V2. Never treated as meters without calibration.
   - **Metric Depth**: Calibrated physical distance (meters) along the camera optical axis ($Z > 0$).
   - **Elevation**: Vertical height $Z$ relative to a vertical datum / projected coordinate system (e.g. WGS84 MSL).
   - **Digital Surface Model (DSM)**: 2D raster grid of surface elevations in a defined CRS.
2. **No Scale Fabrication**: GPS latitude/longitude alone does not determine scene scale. When references are absent, the system gracefully degrades to relative depth mode (State A/B) and explicitly reports why.
3. **Separation of Raw vs Visual Outputs**: Raw floating-point arrays (`.npy`, 32-bit TIFF) are preserved for computation and strictly separated from 8-bit colormapped preview images (`.png`).

---

## 2. Pipeline Execution States & Graceful Degradation

| Pipeline State | Inputs Required | Outputs Generated |
| :--- | :--- | :--- |
| **State A (Relative Depth)** | Single Image (any format) | Raw relative depth (`.npy`, `.tif`), 8-bit Turbo colormap (`.png`), metadata summary |
| **State B (Relative 3D)** | Image + Camera Intrinsics (EXIF / heuristic) | State A outputs + 3D Point Cloud in Camera Frame (`.ply`, WebGL `.json`) |
| **State C (Metric 3D)** | Image + Intrinsics + Metric Reference (AGL / GCPs / Known Distance) | State B outputs + Calibrated Metric Depth (`.npy`, `.png`) + Metric 3D Point Cloud in meters (`.ply`) |
| **State D (Georeferenced DSM)** | Image + Intrinsics + Metric Reference + Geospatial Data (GPS / GeoTIFF) | State C outputs + Georeferenced Point Cloud in Projected CRS + GeoTIFF DSM (`.tif`) + Hillshade (`.png`) + Draped Relief (`.png`) |

---

## 3. Architecture Overview

```
backend/
├── api/                     # FastAPI route handlers (/process, /validate, /models, /artifacts, /health)
├── camera/                  # Camera intrinsics, sensor database, extrinsics modeling
├── config.py                # Engine paths, resolution, device defaults
├── depth/                   # DepthAnything V2 singleton model manager & relative depth estimator
├── dsm/                     # Digital Surface Model rasterizer & GeoTIFF exporter
├── geometry/                # 3D back-projection & PLY/JSON point cloud serializer
├── geospatial/              # PyProj CRS manager & Camera->ENU->Projected CRS transformer
├── ingestion/               # JPEG/PNG/TIFF/GeoTIFF reader & EXIF/GPS metadata extractor
├── main.py                  # FastAPI application entry point
├── metric/                  # Metric calibration strategies (Altitude/Ground, GCPs, Distance, Manual)
├── models/                  # Typed Pydantic and dataclass models
├── pipeline.py              # Pipeline orchestrator managing States A, B, C, D
├── storage/                 # Artifact manager saving per-request files
├── tests/                   # Comprehensive pytest test suite (33 passing tests)
└── visualization/           # Scientific colormapping, hillshades, and blended relief
```

---

## 4. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/depth/process` | Multipart upload for image + optional calibration/camera params |
| `POST` | `/api/v1/depth/validate` | Statistical validation comparing predictions against ground truth (MAE, RMSE, LE90, LE95) |
| `GET` | `/api/v1/depth/models` | Status of loaded DepthAnything V2 model and execution device |
| `GET` | `/api/v1/artifacts/{request_id}/{filename}` | Stream or download generated computational/visual artifacts |
| `GET` | `/api/v1/health` | Health check and system capabilities |

---

## 5. Quick Start & Execution

### Running the API Server

```bash
.venv\Scripts\uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Running Tests

```bash
.venv\Scripts\python -m pytest backend/tests -v
```
