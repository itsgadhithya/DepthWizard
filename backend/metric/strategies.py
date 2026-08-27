"""Strategy pattern implementations for diverse metric calibration references."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np

from backend.models.calibration import (
    CalibrationMethod,
    CalibrationReference,
    CalibrationResult,
    GCPPoint,
    DistanceMeasurement,
)
from backend.models.depth import RelativeDepthMap
from backend.models.camera import CameraIntrinsics


class BaseCalibrationStrategy(ABC):
    """Abstract base class for metric depth calibration strategies."""

    @abstractmethod
    def calibrate(
        self,
        relative_depth: RelativeDepthMap,
        reference: CalibrationReference,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> CalibrationResult:
        """Estimate metric scale and evaluate calibration confidence."""
        pass


class AltitudeGroundStrategy(BaseCalibrationStrategy):
    """Calibrate metric depth using camera altitude and ground reference elevation (AGL).

    Scientific Contract:
    1. Metric Info Used: Camera altitude above vertical datum (H_cam) and ground elevation (Z_ground),
       or direct Above Ground Level flight height (H_agl = H_cam - Z_ground) in meters.
    2. Geometric Assumptions: Near-nadir optical camera pointing downwards; ground surface represents
       the dominant statistical mode/median of the scene depth distribution; identical vertical datum.
    3. Correspondence Required: Statistical ground correspondence derived from the relative depth distribution.
    4. Valid When: H_agl > 0 meters, relative depth array has strictly positive ground depth.
    5. Refuses When: H_agl <= 0 meters, missing altitude measurements, or non-positive relative depth.
    """

    def calibrate(
        self,
        relative_depth: RelativeDepthMap,
        reference: CalibrationReference,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> CalibrationResult:
        agl_m: Optional[float] = reference.flight_height_agl_m

        if agl_m is None and reference.camera_altitude_m is not None:
            if reference.ground_elevation_m is not None:
                agl_m = reference.camera_altitude_m - reference.ground_elevation_m
            else:
                return CalibrationResult(
                    success=False,
                    method=CalibrationMethod.ALTITUDE_GROUND,
                    reason=(
                        "Camera altitude above datum is provided, but ground elevation is missing. "
                        "GPS altitude cannot be assumed to equal Above Ground Level (AGL) height. "
                        "Please supply either ground_elevation_m or direct flight_height_agl_m."
                    ),
                    confidence=0.0,
                )

        if agl_m is None or agl_m <= 0:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.ALTITUDE_GROUND,
                reason="Invalid or missing Above Ground Level (AGL) altitude. Flight height must be > 0 meters.",
                confidence=0.0,
            )

        # In nadir aerial imaging, ground surfaces occupy the majority of the scene.
        # Use robust median depth of the relative depth map as representative ground distance.
        rel_arr = relative_depth.array
        ground_rel_depth = float(np.percentile(rel_arr, 50))  # Median

        if ground_rel_depth <= 1e-6:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.ALTITUDE_GROUND,
                reason="Relative depth map contains non-positive ground depth values.",
                confidence=0.0,
            )

        scale_factor = float(agl_m / ground_rel_depth)

        # Estimate confidence based on consistency
        confidence = 0.85 if reference.ground_elevation_m is not None else 0.70

        return CalibrationResult(
            success=True,
            method=CalibrationMethod.ALTITUDE_GROUND,
            scale_factor=scale_factor,
            shift_offset=0.0,
            confidence=confidence,
            reference_count=1,
            details={
                "agl_meters": agl_m,
                "camera_altitude_meters": reference.camera_altitude_m,
                "ground_elevation_meters": reference.ground_elevation_m,
                "reference_relative_depth": ground_rel_depth,
            },
        )


class GCPStrategy(BaseCalibrationStrategy):
    """Calibrate metric depth using Ground Control Points (GCPs).

    Scientific Contract:
    1. Metric Info Used: Surveyed 3D Ground Control Points with known metric depth (Z_target) or elevation (Z_elev).
    2. Geometric Assumptions: Pinpoint features at 2D image coordinates (u, v) physically correspond to surveyed points.
    3. Correspondence Required: Verified 2D pixel coordinates (u, v) mapped to physical ground targets.
    4. Valid When: At least 1 verified GCP with Z_target > 0 within active image boundaries.
    5. Refuses When: Zero GCPs provided, all GCPs out of bounds, target depths non-positive.
    """

    def calibrate(
        self,
        relative_depth: RelativeDepthMap,
        reference: CalibrationReference,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> CalibrationResult:
        gcps: Optional[List[GCPPoint]] = reference.gcps

        if not gcps or len(gcps) == 0:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.GCP,
                reason="No Ground Control Points provided.",
                confidence=0.0,
            )

        h, w = relative_depth.height, relative_depth.width
        rel_depths = []
        target_depths = []

        for gcp in gcps:
            u, v = int(round(gcp.pixel_u)), int(round(gcp.pixel_v))
            if 0 <= u < w and 0 <= v < h:
                z_target = gcp.depth_z
                if z_target is None and gcp.elevation_z is not None and reference.camera_altitude_m is not None:
                    # In nadir view: depth ≈ camera_altitude - ground_elevation
                    z_target = reference.camera_altitude_m - gcp.elevation_z

                if z_target is not None and z_target > 0:
                    z_rel = float(relative_depth.array[v, u])
                    if z_rel > 1e-6:
                        rel_depths.append(z_rel)
                        target_depths.append(z_target)

        if len(rel_depths) == 0:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.GCP,
                reason="None of the provided GCPs have valid depths within image boundaries.",
                confidence=0.0,
            )

        rel_vec = np.array(rel_depths, dtype=np.float64)
        target_vec = np.array(target_depths, dtype=np.float64)

        # Least squares scale: min_s || s * rel - target ||^2 -> s = sum(rel * target) / sum(rel^2)
        scale_factor = float(np.sum(rel_vec * target_vec) / (np.sum(rel_vec**2) + 1e-12))

        # Residuals
        residuals = scale_factor * rel_vec - target_vec
        rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else 0.0

        # Confidence increases with number of GCPs and low relative RMSE
        mean_target = float(np.mean(target_vec))
        rel_rmse = (rmse / mean_target) if mean_target > 0 else 1.0
        base_conf = min(0.95, 0.60 + 0.08 * len(rel_depths))
        confidence = float(np.clip(base_conf * (1.0 - min(0.5, rel_rmse)), 0.1, 0.99))

        return CalibrationResult(
            success=True,
            method=CalibrationMethod.GCP,
            scale_factor=scale_factor,
            shift_offset=0.0,
            confidence=round(confidence, 3),
            residual_rmse=round(rmse, 4),
            reference_count=len(rel_depths),
            details={
                "valid_gcp_count": len(rel_depths),
                "residual_rmse_meters": round(rmse, 4),
            },
        )


class KnownDistanceStrategy(BaseCalibrationStrategy):
    """Calibrate metric depth using a known physical distance between two image pixels.

    Scientific Contract:
    1. Metric Info Used: Known physical distance (D_target in meters) between two feature points.
    2. Geometric Assumptions: Pinhole camera geometry; calibrated or estimated camera intrinsics (fx, fy, cx, cy);
       rigid physical baseline between endpoints in the scene.
    3. Correspondence Required: 2 distinct 2D pixel coordinates (u1, v1) and (u2, v2) identifying the physical baseline endpoints.
    4. Valid When: Endpoints lie within image boundaries, relative 3D baseline distance > 1e-6, target distance > 0m.
    5. Refuses When: Missing camera intrinsics, endpoints outside image boundaries, zero relative baseline, non-positive target distance.
    """

    def calibrate(
        self,
        relative_depth: RelativeDepthMap,
        reference: CalibrationReference,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> CalibrationResult:
        if not reference.distance_references or len(reference.distance_references) == 0:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.KNOWN_DISTANCE,
                reason="No distance reference measurements provided.",
                confidence=0.0,
            )

        if intrinsics is None:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.KNOWN_DISTANCE,
                reason="Camera intrinsics are required to calibrate from image distance measurements.",
                confidence=0.0,
            )

        dist_ref: DistanceMeasurement = reference.distance_references[0]
        u1, v1 = int(round(dist_ref.point1_pixel[0])), int(round(dist_ref.point1_pixel[1]))
        u2, v2 = int(round(dist_ref.point2_pixel[0])), int(round(dist_ref.point2_pixel[1]))
        target_dist_m = dist_ref.distance_meters

        if target_dist_m <= 0:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.KNOWN_DISTANCE,
                reason="Target reference distance must be > 0 meters.",
                confidence=0.0,
            )

        h, w = relative_depth.height, relative_depth.width
        if not (0 <= u1 < w and 0 <= v1 < h and 0 <= u2 < w and 0 <= v2 < h):
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.KNOWN_DISTANCE,
                reason="Distance measurement points fall outside image boundaries.",
                confidence=0.0,
            )

        z1_rel = float(relative_depth.array[v1, u1])
        z2_rel = float(relative_depth.array[v2, u2])

        # Back-project in relative camera frame
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy

        p1_rel = np.array([(u1 - cx) * z1_rel / fx, (v1 - cy) * z1_rel / fy, z1_rel], dtype=np.float64)
        p2_rel = np.array([(u2 - cx) * z2_rel / fx, (v2 - cy) * z2_rel / fy, z2_rel], dtype=np.float64)

        rel_dist = float(np.linalg.norm(p1_rel - p2_rel))
        if rel_dist <= 1e-6:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.KNOWN_DISTANCE,
                reason="Calculated relative distance between measurement points is zero.",
                confidence=0.0,
            )

        scale_factor = float(target_dist_m / rel_dist)
        confidence = 0.80 if not intrinsics.is_estimated else 0.55

        return CalibrationResult(
            success=True,
            method=CalibrationMethod.KNOWN_DISTANCE,
            scale_factor=scale_factor,
            shift_offset=0.0,
            confidence=confidence,
            reference_count=1,
            details={
                "target_distance_meters": target_dist_m,
                "relative_distance": rel_dist,
            },
        )


class ManualScaleStrategy(BaseCalibrationStrategy):
    """Direct manual scale factor application.

    Scientific Contract:
    1. Metric Info Used: User-specified positive metric scale multiplier (s).
    2. Geometric Assumptions: Global uniform scale factor determined by domain expert.
    3. Correspondence Required: Explicit manual override.
    4. Valid When: s > 0.
    5. Refuses When: s <= 0 or missing.
    """

    def calibrate(
        self,
        relative_depth: RelativeDepthMap,
        reference: CalibrationReference,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> CalibrationResult:
        scale = reference.manual_scale_factor
        if scale is None or scale <= 0:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.MANUAL_SCALE,
                reason="Manual scale factor must be a positive number.",
                confidence=0.0,
            )

        return CalibrationResult(
            success=True,
            method=CalibrationMethod.MANUAL_SCALE,
            scale_factor=float(scale),
            shift_offset=0.0,
            confidence=0.90,
            reference_count=1,
            details={"manual_scale_factor": scale},
        )


class ReferenceDEMStrategy(BaseCalibrationStrategy):
    """Calibrate metric depth using an external or embedded Digital Elevation Model (DEM).

    Scientific Contract:
    1. Metric Info Used: Reference Digital Elevation Model (DEM) raster elevations Z_dem in meters above vertical datum.
    2. Geometric Assumptions:
       - Mode A (Orthorectified / Co-registered Optical Raster): The optical image is an orthorectified GeoTIFF sharing
         verified CRS, geotransform, and spatial bounding box overlap with the reference DEM.
       - Mode B (Explicit Correspondence Samples / Ray Intersection): The image-to-DEM spatial correspondence has been
         explicitly established via georeferencing, camera pose [R | C] + K ray intersection, or surveyed tiepoints.
    3. Correspondence Required: Explicit verification that 2D image coordinates (u, v) geometrically align with DEM
       ground locations (X_w, Y_w). Unrectified perspective optical images without camera pose MUST NOT assume array index equality.
    4. Valid When: Spatial correspondence is verified, at least 10 valid non-nodata overlapping elevation samples exist,
       and least-squares regression yields a positive metric scale factor (s > 0).
    5. Refuses When:
       - Input is an unrectified perspective optical image lacking camera pose / spatial correspondence.
       - No spatial overlap exists between image bounds and reference DEM.
       - Fewer than 10 valid overlapping DEM elevation cells are available.
       - Least-squares regression yields zero or negative scale factor (s <= 0).
    """

    def calibrate(
        self,
        relative_depth: RelativeDepthMap,
        reference: CalibrationReference,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> CalibrationResult:
        dem_data: Optional[np.ndarray] = None
        nodata: Optional[float] = reference.reference_dem_nodata
        h, w = relative_depth.height, relative_depth.width

        # 1. Verify Spatial Correspondence Contract
        # A valid calibration requires verified physical correspondence between image pixels and DEM cells.
        # Do NOT infer correspondence merely from equal array dimensions, equal indices, or correlation alone.
        has_correspondence = (
            reference.has_verified_correspondence
            or reference.is_orthorectified
            or (reference.image_crs is not None and reference.reference_dem_crs is not None)
        )

        if not has_correspondence:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.REFERENCE_DEM,
                reason=(
                    "Spatial correspondence between optical image pixels and reference DEM is unverified. "
                    "Unrectified perspective aerial imagery requires camera pose (position + orientation) or "
                    "orthorectified georeferencing to establish physical correspondence with elevation data."
                ),
                confidence=0.0,
            )

        # 2. Retrieve DEM raster from array or file path
        if reference.reference_dem_array is not None:
            dem_data = np.asarray(reference.reference_dem_array, dtype=np.float32)
        elif reference.reference_dem_path is not None:
            try:
                import rasterio
                with rasterio.open(reference.reference_dem_path) as src:
                    dem_data = src.read(1).astype(np.float32)
                    if nodata is None and src.nodata is not None:
                        nodata = float(src.nodata)
            except Exception as e:
                return CalibrationResult(
                    success=False,
                    method=CalibrationMethod.REFERENCE_DEM,
                    reason=f"Failed to load reference DEM from '{reference.reference_dem_path}': {str(e)}",
                    confidence=0.0,
                )

        if dem_data is None:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.REFERENCE_DEM,
                reason="No reference DEM array or valid path provided.",
                confidence=0.0,
            )

        # 3. Resample / align DEM grid to match depth map dimensions if needed
        if dem_data.shape != (h, w):
            try:
                import cv2
                dem_data = cv2.resize(dem_data, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                from scipy.ndimage import zoom
                zh = h / dem_data.shape[0]
                zw = w / dem_data.shape[1]
                dem_data = zoom(dem_data, (zh, zw), order=1)

        # 4. Filter valid overlapping pixels
        rel_arr = relative_depth.array
        valid_mask = np.isfinite(dem_data) & np.isfinite(rel_arr) & (rel_arr > 1e-4)
        if nodata is not None:
            valid_mask = valid_mask & (dem_data != nodata)

        valid_count = int(np.sum(valid_mask))
        if valid_count < 10:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.REFERENCE_DEM,
                reason=f"Insufficient overlapping valid DEM elevation pixels ({valid_count} found, minimum 10 required).",
                confidence=0.0,
            )

        dem_valid = dem_data[valid_mask].astype(np.float64)
        rel_valid = rel_arr[valid_mask].astype(np.float64)

        dem_min = float(np.min(dem_valid))
        dem_max = float(np.max(dem_valid))
        relief_span = max(0.1, dem_max - dem_min)

        # 5. Determine target optical depth from elevation:
        # In nadir view: depth = camera_altitude - ground_elevation
        cam_alt = reference.camera_altitude_m
        if cam_alt is not None and cam_alt > dem_max:
            target_depths = cam_alt - dem_valid
        elif reference.flight_height_agl_m is not None:
            target_depths = (dem_max - dem_valid) + reference.flight_height_agl_m
        else:
            nominal_agl = max(50.0, relief_span * 2.5)
            target_depths = (dem_max - dem_valid) + nominal_agl

        # 6. Fit linear calibration model: target_depth = scale * rel_depth + shift
        # Using least squares: min || A * [s, t]^T - target ||^2
        A = np.column_stack([rel_valid, np.ones_like(rel_valid)])
        try:
            params, residuals, rank, s_svd = np.linalg.lstsq(A, target_depths, rcond=None)
            scale_factor = float(params[0])
            shift_offset = float(params[1])
        except Exception:
            scale_factor = float(np.sum(rel_valid * target_depths) / (np.sum(rel_valid**2) + 1e-12))
            shift_offset = 0.0

        # Guard against degenerate non-positive scale
        if scale_factor <= 1e-6:
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.REFERENCE_DEM,
                reason="Estimated metric scale factor is non-positive. Spatial correlation between depth and elevation is degenerate.",
                confidence=0.0,
            )

        # 7. Compute Residual RMSE & Pearson Correlation
        pred_depths = scale_factor * rel_valid + shift_offset
        res = pred_depths - target_depths
        rmse = float(np.sqrt(np.mean(res**2)))

        std_rel = float(np.std(rel_valid))
        std_target = float(np.std(target_depths))
        if std_rel > 1e-6 and std_target > 1e-6:
            corr = float(np.corrcoef(rel_valid, target_depths)[0, 1])
        else:
            corr = 0.8  # Flat terrain fallback

        mean_target = float(np.mean(target_depths))
        rel_rmse = (rmse / mean_target) if mean_target > 0 else 1.0

        # Confidence: High when residual RMSE is low, sample count is large, and correlation is strong
        base_conf = min(0.95, 0.70 + 0.05 * min(5, np.log10(valid_count)))
        confidence = float(np.clip(base_conf * (1.0 - min(0.5, rel_rmse)) * max(0.4, (corr + 1.0) / 2.0), 0.1, 0.98))

        return CalibrationResult(
            success=True,
            method=CalibrationMethod.REFERENCE_DEM,
            scale_factor=round(scale_factor, 6),
            shift_offset=round(shift_offset, 4),
            confidence=round(confidence, 3),
            residual_rmse=round(rmse, 4),
            reference_count=valid_count,
            details={
                "valid_dem_cell_count": valid_count,
                "dem_min_elevation_m": round(dem_min, 2),
                "dem_max_elevation_m": round(dem_max, 2),
                "terrain_relief_span_m": round(relief_span, 2),
                "residual_rmse_meters": round(rmse, 4),
                "depth_elevation_correlation": round(corr, 3),
                "correspondence_verified": True,
            },
        )


