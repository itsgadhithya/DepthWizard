# DepthWizard — Depth & Metric Geometry Engine Specification

## 1. Purpose

DepthWizard accepts a single georeferenced or non-georeferenced aerial image and produces:

1. A relative depth map using DepthAnything V2.
2. A calibrated metric depth representation when sufficient metric/geospatial information is available.
3. A georeferenced 3D representation.
4. A Digital Surface Model (DSM) when metric/geospatial calibration is possible.
5. Validation metrics describing the reliability of the generated geometry.
6. Data suitable for an interactive 3D visualization.

The system must clearly distinguish between:
- relative depth,
- metric depth,
- elevation,
- and DSM.

Never treat relative depth as metres without calibration.

---

## 2. Core Constraint

The current system accepts ONE IMAGE per processing request.

Therefore:
- Do NOT use stereo reconstruction as the primary depth-generation method.
- Do NOT require multiple images.
- Do NOT assume Structure-from-Motion or Multi-View Stereo is possible.
- DepthAnything V2 is the primary monocular depth estimator.

The architecture should remain extensible so multi-image photogrammetry can be added later.

---

## 3. High-Level Pipeline

Input image
    ↓
Image validation
    ↓
Metadata extraction
    ↓
DepthAnything V2
    ↓
Raw relative depth
    ↓
Camera model / intrinsic estimation
    ↓
Metric calibration
    ↓
Metric 3D reconstruction
    ↓
Georeferencing
    ↓
DSM generation
    ↓
Validation
    ↓
3D visualization outputs

Not every input will support every stage.

The system must gracefully degrade.

For example:

Image
 → relative depth
 → no metric calibration available
 → return relative-depth result
 → clearly report that metric geometry could not be established.

---

## 4. DepthAnything V2

DepthAnything V2 is the primary monocular depth model.

Requirements:

- Load the model once and reuse it.
- Use GPU when available.
- Support CPU fallback.
- Do not load the model for every request.
- Preserve raw floating-point model output.
- Never use an 8-bit visualization as computational depth.
- Record model version/configuration.
- Record inference time.
- Record device used.
- Record input dimensions.

Outputs:

- raw depth array
- normalized visualization
- inference metadata

The raw depth must remain separate from the visualization.

---

## 5. Metadata

Extract all available metadata without assuming it exists.

Potential metadata includes:

- image dimensions
- image format
- EXIF
- GPS latitude
- GPS longitude
- GPS altitude
- camera make
- camera model
- focal length
- sensor information if available
- image orientation
- timestamp
- CRS/geospatial metadata where applicable

The system must explicitly indicate which metadata fields are:
- present
- absent
- estimated
- inferred

Never fabricate missing metadata.

---

## 6. Input Formats

The architecture should support:

- JPEG
- PNG
- TIFF
- GeoTIFF

GeoTIFF should be treated differently from ordinary JPEG/PNG because it may contain:
- CRS
- affine transform
- geographic bounds
- pixel resolution
- geospatial coordinates.

Use appropriate geospatial libraries instead of manually parsing GeoTIFF structures.

---

## 7. Camera Model

Represent camera information explicitly.

The camera model should support:

- image width
- image height
- focal length
- fx
- fy
- principal point cx/cy
- distortion parameters where available
- camera position
- camera orientation

Use an explicit camera-intrinsics object/data model.

Do not silently invent highly accurate camera parameters.

If parameters are estimated, mark them as estimated.

---

## 8. Relative Depth

DepthAnything V2 output is relative/model-space depth.

Do not represent it as metres.

Use a dedicated representation such as:

RelativeDepthMap

containing:

- array
- width
- height
- minimum
- maximum
- model information
- normalization information

---

## 9. Metric Calibration

Metric calibration converts relative/model-space depth into a physically meaningful representation.

Possible metric references include:

- known camera altitude
- known ground elevation
- Ground Control Points (GCPs)
- known distances
- known object dimensions
- external DEM/DSM
- other explicitly supplied reference measurements

GPS latitude/longitude alone must NOT be considered sufficient to determine scene depth scale.

Camera position tells the system where the camera is.

It does not by itself determine the 3D position of every visible surface.

The calibration system must therefore identify:
- calibration method
- input references
- estimated parameters
- confidence/quality information
- whether metric calibration succeeded

If there is insufficient information, return a relative-depth result rather than inventing metric scale.

---

## 10. 3D Reconstruction

Where camera intrinsics and calibrated depth are available, back-project image pixels into 3D.

For pixel (u,v) with depth Z:

X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z

Represent the result as a point cloud.

The point cloud must retain:
- coordinate system
- units
- source pixel coordinates where useful
- confidence/validity information where available.

---

## 11. Georeferencing

Use proper geospatial libraries.

Preferred libraries include:

- rasterio
- pyproj
- numpy
- scipy
- GDAL where appropriate

Do not manually implement CRS transformations.

Support:
- WGS84
- projected coordinate systems
- metric coordinate systems

The CRS must always be explicitly associated with geospatial outputs.

---

## 12. DSM

A DSM represents the elevation of the visible surface.

The system must distinguish:

Depth:
distance from camera to surface.

Elevation:
height relative to a reference coordinate system.

DSM:
a rasterized representation of surface elevation.

DSM generation should only be presented as metric/georeferenced when the underlying geometry supports it.

DSM output should preserve:
- CRS
- resolution
- transform
- nodata
- elevation units
- bounds

Prefer GeoTIFF for geospatial raster output.

---

## 13. Validation

The system should support comparison against reference data where available.

Metrics may include:

- MAE
- RMSE
- mean error/bias
- maximum absolute error
- percentage within tolerance
- valid coverage

Validation results must state what reference was used.

Do not present estimated results as ground truth.

---

## 14. API Design

Expose processing through a clean backend API.

Suggested endpoint:

POST /api/v1/depth/process

Input:
- single image
- optional calibration/reference information

Output should include:

- processing status
- relative depth availability
- metric depth availability
- georeferencing availability
- DSM availability
- metadata summary
- calibration method
- confidence/quality information
- output artifact references
- timing information

---

## 15. Graceful Degradation

The pipeline must support these states:

### State A
Image
→ relative depth

### State B
Image + camera information
→ relative depth
→ 3D reconstruction in camera coordinates

### State C
Image + sufficient metric references
→ metric depth
→ metric 3D geometry

### State D
Image + sufficient geospatial information
→ georeferenced metric geometry
→ DSM

Never manufacture missing information.

---

## 16. Architecture

Recommended backend modules:

backend/
    api/
    ingestion/
    depth/
    camera/
    geometry/
    metric/
    geospatial/
    dsm/
    validation/
    models/
    visualization/
    tests/

Keep each responsibility isolated.

Avoid one monolithic processing function.

---

## 17. Data Model

Create explicit models for:

ImageMetadata
CameraModel
RelativeDepthMap
MetricDepthMap
CalibrationReference
CalibrationResult
PointCloud
DSM
ValidationResult
ProcessingResult

Use typed Python models where appropriate.

---

## 18. Reproducibility

Every processing result should record:

- model name
- model version/configuration
- software version
- processing timestamp
- input dimensions
- device
- calibration method
- calibration references
- CRS
- units
- confidence/quality information

---

## 19. Engineering Principles

1. Never fabricate measurements.
2. Never call relative depth "metric depth".
3. Never call depth "elevation".
4. Never assume GPS coordinates alone provide depth scale.
5. Preserve raw outputs.
6. Separate computation from visualization.
7. Make uncertainty explicit.
8. Use established geospatial libraries.
9. Keep the system extensible to multi-image photogrammetry later.
10. Prefer scientifically defensible outputs over visually impressive outputs.

---

## 20. Future Extension

The architecture should allow a future multi-image mode:

Multiple images
    ↓
SfM
    ↓
MVS
    ↓
dense geometry
    ↓
DSM

However, this is NOT part of the current single-image implementation.