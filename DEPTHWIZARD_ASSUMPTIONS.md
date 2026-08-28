# DepthWizard — Current Assumptions

## Current Input

The current product accepts ONE IMAGE per processing request.

## Primary Depth Method

DepthAnything V2 is the primary monocular depth estimator.

## Important Distinction

DepthAnything V2 produces relative/model-space depth.

It does NOT automatically produce reliable metric depth in metres.

## Metric Depth

Metric depth requires additional information or reference measurements.

Potential sources:
- camera altitude
- ground elevation
- GCPs
- known distances
- known object dimensions
- external DEM/DSM

GPS coordinates alone are not considered sufficient to establish depth scale.

## DSM

A DSM represents surface elevation, not simply camera-relative depth.

DSM generation must only be enabled when the available geometry and calibration support it.

## Multi-Image Photogrammetry

Not currently supported.

The architecture should remain extensible for future:
- SfM
- MVS
- stereo
- multi-image reconstruction

## Scientific Integrity

The system must never silently invent:
- scale
- altitude
- camera parameters
- CRS
- elevation
- accuracy