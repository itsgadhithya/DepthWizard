"""Single-image depth and metric geometry pipeline orchestrator with graceful degradation."""

import time
import uuid
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

from backend.models.results import PipelineState, ProcessingSummary
from backend.models.calibration import CalibrationReference, CalibrationResult, CalibrationMethod
from backend.models.metadata import ImageMetadata
from backend.models.camera import CameraModel
from backend.models.depth import RelativeDepthMap, MetricDepthMap
from backend.models.geometry import PointCloud3D
from backend.models.dsm import DSMResult
from backend.models.validation import ValidationReport

from backend.ingestion.reader import ImageReader
from backend.ingestion.metadata_extractor import MetadataExtractor
from backend.depth.estimator import RelativeDepthEstimator
from backend.camera.model import CameraModelBuilder
from backend.metric.calibrator import MetricCalibrator
from backend.geometry.backprojector import DepthBackProjector
from backend.geospatial.transformer import CoordinateTransformer
from backend.dsm.rasterizer import DSMRasterizer
from backend.validation.metrics import ValidationEngine
from backend.storage.artifact_manager import ArtifactManager


class SingleImagePipeline:
    """Production single-image depth, geometry, and DSM processing pipeline."""

    @classmethod
    def process(
        cls,
        image_input: Union[str, Path, bytes],
        filename: str = "image.jpg",
        calibration_ref: Optional[CalibrationReference] = None,
        camera_overrides: Optional[Dict[str, Any]] = None,
        target_crs: Optional[str] = None,
        dsm_resolution_m: float = 0.5,
        validation_ref_points: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ProcessingSummary:
        """Execute the full single-image depth and metric geometry pipeline.

        Pipeline Stages:
            1. Image Validation & Ingestion
            2. Metadata Extraction (EXIF / GPS / GeoTIFF) with provenance
            3. DepthAnything V2 Relative Depth Estimation
            4. Camera Modeling (Intrinsics & Extrinsics)
            5. Relative 3D Back-projection (State B)
            6. Optional Metric Calibration (State C)
            7. Optional Geospatial Transformation & DSM Rasterization (State D)
            8. Optional Validation
            9. Output Artifact Generation

        Graceful Degradation Guarantees:
            - State A: Relative Depth only.
            - State B: Relative Depth + Relative 3D Point Cloud.
            - State C: Metric Depth (meters) + Metric 3D Point Cloud.
            - State D: Georeferenced Metric Point Cloud + GeoTIFF DSM.
        """
        t_start = time.perf_counter()
        req_id = request_id or str(uuid.uuid4())[:8]

        timings: Dict[str, float] = {}
        messages = []
        warnings = []
        all_artifacts = {}

        current_state = PipelineState.STATE_A

        # ---------------------------------------------------------
        # STAGE 1: Image Ingestion
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                img_bytes = f.read()
            filename = Path(image_input).name
        else:
            img_bytes = image_input

        image_rgb, format_name, raw_info = ImageReader.read_image(img_bytes)
        input_art = ArtifactManager.save_uploaded_image(req_id, filename, img_bytes)
        all_artifacts["input_image"] = input_art
        timings["ingestion_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        messages.append(f"Loaded image '{filename}' ({image_rgb.shape[1]}x{image_rgb.shape[0]} px, {format_name}).")

        # ---------------------------------------------------------
        # STAGE 2: Metadata Extraction
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        metadata = MetadataExtractor.extract_metadata(img_bytes, filename=filename, format_hint=format_name)
        timings["metadata_extraction_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if metadata.has_exif:
            messages.append("EXIF metadata successfully extracted.")
        else:
            warnings.append("No EXIF metadata found in image.")

        if metadata.has_gps:
            messages.append(f"GPS tags found: Lat {metadata.gps.latitude:.5f}, Lon {metadata.gps.longitude:.5f}, Alt {metadata.gps.altitude}m.")
        else:
            warnings.append("No GPS position tags found in image.")

        if metadata.has_geotiff and metadata.geotiff:
            if metadata.is_dem:
                messages.append(
                    f"GeoTIFF identified as Digital Elevation Model (DEM) with CRS '{metadata.geotiff.crs}' "
                    f"(elevation range: [{metadata.geotiff.min_elevation_m}m, {metadata.geotiff.max_elevation_m}m])."
                )
            else:
                messages.append(f"GeoTIFF identified as Optical Orthophoto with CRS: '{metadata.geotiff.crs}'.")

        # ---------------------------------------------------------
        # STAGE 3: Depth Estimation (DepthAnything V2 for Optical, Passthrough for DEM)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        if metadata.is_dem and raw_info.get("dem_array") is not None:
            # Single-band DEM is an elevation raster product, NOT an optical camera photograph.
            # Do NOT pass elevation rasters to DepthAnything V2 as optical imagery.
            dem_arr = raw_info["dem_array"]
            valid_mask = np.isfinite(dem_arr)
            nodata_val = raw_info.get("nodata")
            if nodata_val is not None:
                valid_mask = valid_mask & (dem_arr != nodata_val)

            v_min = float(np.min(dem_arr[valid_mask])) if np.any(valid_mask) else 0.0
            v_max = float(np.max(dem_arr[valid_mask])) if np.any(valid_mask) else 1.0
            v_mean = float(np.mean(dem_arr[valid_mask])) if np.any(valid_mask) else 0.5
            v_std = float(np.std(dem_arr[valid_mask])) if np.any(valid_mask) else 0.1

            # Map terrain elevation to inverted camera-relative depth proxy: higher terrain = closer/smaller depth
            span = max(1e-6, v_max - v_min)
            rel_arr = ((v_max - dem_arr) / span + 0.1).astype(np.float32)

            relative_depth = RelativeDepthMap(
                array=rel_arr,
                width=dem_arr.shape[1],
                height=dem_arr.shape[0],
                min_val=round(float(np.min(rel_arr[valid_mask])), 3) if np.any(valid_mask) else 0.1,
                max_val=round(float(np.max(rel_arr[valid_mask])), 3) if np.any(valid_mask) else 1.1,
                mean_val=round(float(np.mean(rel_arr[valid_mask])), 3) if np.any(valid_mask) else 0.6,
                std_val=round(float(np.std(rel_arr[valid_mask])), 3) if np.any(valid_mask) else 0.2,
                model_name="dem_elevation_passthrough",
                model_config_name="elevation_raster",
                device="cpu",
                inference_time_ms=0.0,
                is_metric=False,
                units="dimensionless",
            )
            warnings.append(
                "Input is a single-band elevation raster (DEM), not an optical camera photograph. "
                "DepthAnything V2 optical inference was bypassed."
            )
        else:
            relative_depth: RelativeDepthMap = RelativeDepthEstimator.estimate_depth(image_rgb)
            messages.append(
                f"DepthAnything V2 inference completed on device '{relative_depth.device}' "
                f"in {relative_depth.inference_time_ms:.1f}ms. Relative depth range: [{relative_depth.min_val:.3f}, {relative_depth.max_val:.3f}]."
            )

        depth_arts = ArtifactManager.save_raw_relative_depth(req_id, relative_depth)
        all_artifacts.update(depth_arts)
        timings["depth_inference_ms"] = relative_depth.inference_time_ms

        current_state = PipelineState.STATE_A

        # ---------------------------------------------------------
        # STAGE 4: Camera Modeling
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        camera: CameraModel = CameraModelBuilder.build_camera_model(metadata, camera_overrides)
        timings["camera_modeling_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if camera.intrinsics.is_estimated:
            warnings.append(
                f"Camera focal length estimated using method '{camera.intrinsics.estimation_method}' "
                f"(fx={camera.intrinsics.fx:.1f}px, confidence={camera.intrinsics.confidence:.2f})."
            )
        else:
            messages.append(
                f"Camera focal length determined via '{camera.intrinsics.estimation_method}' "
                f"(fx={camera.intrinsics.fx:.1f}px)."
            )

        # ---------------------------------------------------------
        # STAGE 5: Relative 3D Back-projection (State B)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        rel_point_cloud = DepthBackProjector.backproject(
            depth_map=relative_depth,
            intrinsics=camera.intrinsics,
            image_rgb=image_rgb,
            compute_normals=True,
            subsample_step=2,
        )
        pc_arts = ArtifactManager.save_point_cloud(req_id, rel_point_cloud)
        all_artifacts.update(pc_arts)
        timings["3d_reconstruction_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        current_state = PipelineState.STATE_B
        messages.append(f"Generated 3D point cloud with {rel_point_cloud.point_count:,} vertices in camera frame.")

        # ---------------------------------------------------------
        # STAGE 6: Metric Calibration (State C)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        metric_depth: Optional[MetricDepthMap] = None
        calib_result: Optional[CalibrationResult] = None

        # Determine effective calibration reference
        effective_calib_ref = calibration_ref
        if (effective_calib_ref is None or effective_calib_ref.method == CalibrationMethod.NONE) and metadata.is_dem and raw_info.get("dem_array") is not None:
            # Auto-use input DEM GeoTIFF elevation array
            effective_calib_ref = CalibrationReference(
                method=CalibrationMethod.REFERENCE_DEM,
                reference_dem_array=raw_info["dem_array"],
                reference_dem_nodata=raw_info.get("nodata"),
                camera_altitude_m=metadata.gps.altitude if (metadata.has_gps and metadata.gps) else None,
            )

        if effective_calib_ref is None or effective_calib_ref.method == CalibrationMethod.NONE:
            metric_depth, calib_result = MetricCalibrator.calibrate(
                relative_depth=relative_depth,
                reference=None,
                intrinsics=camera.intrinsics,
            )
        else:
            metric_depth, calib_result = MetricCalibrator.calibrate(
                relative_depth=relative_depth,
                reference=effective_calib_ref,
                intrinsics=camera.intrinsics,
            )

        timings["metric_calibration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        point_cloud_metric: Optional[PointCloud3D] = None

        if metric_depth is not None and calib_result is not None and calib_result.success:
            current_state = PipelineState.STATE_C
            metric_arts = ArtifactManager.save_metric_depth(req_id, metric_depth)
            all_artifacts.update(metric_arts)

            # Reconstruct metric point cloud
            point_cloud_metric = DepthBackProjector.backproject(
                depth_map=metric_depth,
                intrinsics=camera.intrinsics,
                image_rgb=image_rgb,
                compute_normals=True,
                subsample_step=2,
            )
            # Update saved point cloud artifact to metric version
            pc_arts = ArtifactManager.save_point_cloud(req_id, point_cloud_metric)
            all_artifacts.update(pc_arts)

            if calib_result.is_provisional:
                warnings.append(
                    "Provisional metric scale applied. Depth is scaled using a fixed engineering scale "
                    "and is NOT scientifically calibrated to physical ground truth."
                )
                messages.append(
                    f"Provisional fixed-scale metric depth applied ({calib_result.scale_factor:.1f}x engineering scale). "
                    f"Depth range=[{metric_depth.min_depth_m:.2f}m, {metric_depth.max_depth_m:.2f}m]."
                )
            else:
                messages.append(
                    f"Metric calibration succeeded ({calib_result.method.value}): Scale factor={calib_result.scale_factor:.4f}, "
                    f"Depth range=[{metric_depth.min_depth_m:.2f}m, {metric_depth.max_depth_m:.2f}m]."
                )
        else:
            reason = calib_result.reason if calib_result else "No metric references supplied."
            warnings.append(f"Metric calibration skipped / unavailable: {reason}")
            messages.append("Pipeline gracefully degraded to Relative Depth mode (State B).")

        # ---------------------------------------------------------
        # STAGE 7: Geospatial Georeferencing & DSM (State D)
        # ---------------------------------------------------------
        # STAGE 7: Geospatial Georeferencing & DSM (State D / Local Metric State C)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        dsm_result: Optional[DSMResult] = None
        dsm_type: Optional[str] = None

        if current_state == PipelineState.STATE_C and point_cloud_metric is not None:
            can_georeference = False

            # Check if GeoTIFF CRS + transform is present, or if camera has position
            if metadata.has_geotiff and metadata.geotiff and metadata.geotiff.crs and metadata.geotiff.transform:
                target_crs = target_crs or metadata.geotiff.crs
                can_georeference = True
            elif camera and camera.has_position:
                can_georeference = True

            if can_georeference:
                try:
                    geo_point_cloud = CoordinateTransformer.transform_camera_to_projected_crs(
                        point_cloud=point_cloud_metric,
                        camera=camera,
                        target_crs=target_crs,
                    )
                    # Update point cloud artifact with georeferenced coords
                    geo_pc_arts = ArtifactManager.save_point_cloud(req_id, geo_point_cloud)
                    all_artifacts.update(geo_pc_arts)

                    # Rasterize Georeferenced DSM
                    dsm_result = DSMRasterizer.rasterize(
                        point_cloud=geo_point_cloud,
                        resolution_m=dsm_resolution_m,
                        is_local=False,
                    )
                    dsm_type = "georeferenced_metric"
                    dsm_arts = ArtifactManager.save_dsm(req_id, dsm_result)
                    all_artifacts.update(dsm_arts)

                    current_state = PipelineState.STATE_D
                    messages.append(
                        f"Georeferenced Metric DSM created in '{dsm_result.crs}' with resolution {dsm_result.resolution_m}m/px "
                        f"({dsm_result.width}x{dsm_result.height} grid, coverage: {dsm_result.valid_coverage_percent}%)."
                    )
                except Exception as e:
                    warnings.append(f"Geospatial transform / Georeferenced DSM failed: {str(e)}. Generating Local Metric DSM instead.")
                    dsm_result = DSMRasterizer.rasterize(
                        point_cloud=point_cloud_metric,
                        resolution_m=dsm_resolution_m,
                        is_local=True,
                    )
                    dsm_type = "local_metric"
                    dsm_arts = ArtifactManager.save_dsm(req_id, dsm_result)
                    all_artifacts.update(dsm_arts)
                    messages.append(
                        f"Local Metric DSM generated in camera frame with resolution {dsm_result.resolution_m}m/px "
                        f"({dsm_result.width}x{dsm_result.height} grid, coverage: {dsm_result.valid_coverage_percent}%)."
                    )
            else:
                # Generate Local Metric DSM when georeferencing is not available
                dsm_result = DSMRasterizer.rasterize(
                    point_cloud=point_cloud_metric,
                    resolution_m=dsm_resolution_m,
                    is_local=True,
                )
                dsm_type = "local_metric"
                dsm_arts = ArtifactManager.save_dsm(req_id, dsm_result)
                all_artifacts.update(dsm_arts)
                messages.append(
                    f"Local Metric DSM generated in camera coordinate frame with resolution {dsm_result.resolution_m}m/px "
                    f"({dsm_result.width}x{dsm_result.height} grid, coverage: {dsm_result.valid_coverage_percent}%)."
                )

        timings["geospatial_and_dsm_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # ---------------------------------------------------------
        # STAGE 8: Validation (if reference data provided)
        # ---------------------------------------------------------
        validation_report: Optional[ValidationReport] = None
        if validation_ref_points:
            t0 = time.perf_counter()
            preds = validation_ref_points.get("predicted", [])
            refs = validation_ref_points.get("reference", [])
            tol = float(validation_ref_points.get("tolerance_m", 1.0))
            source_name = str(validation_ref_points.get("reference_source", "Ground Truth Control Points"))

            if len(preds) > 0 and len(refs) > 0:
                validation_report = ValidationEngine.validate_points(
                    predicted_values=preds,
                    reference_values=refs,
                    tolerance_m=tol,
                    reference_name=source_name,
                )
                val_arts = ArtifactManager.save_validation_report(req_id, validation_report)
                all_artifacts.update(val_arts)
                messages.append(
                    f"Validation completed against {source_name}: MAE={validation_report.metrics.mae:.3f}m, "
                    f"RMSE={validation_report.metrics.rmse:.3f}m, LE90={validation_report.metrics.le90:.3f}m."
                )
            timings["validation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # ---------------------------------------------------------
        # Compile Summary
        # ---------------------------------------------------------
        t_total = (time.perf_counter() - t_start) * 1000.0

        # Georeferencing is available if input has valid GeoTIFF spatial metadata (CRS + transform),
        # or valid GPS coordinates, or if State D georeferenced DSM was produced.
        has_geotiff_georef = bool(
            metadata.has_geotiff
            and metadata.geotiff is not None
            and metadata.geotiff.crs is not None
            and metadata.geotiff.transform is not None
        )
        has_gps_georef = bool(
            metadata.has_gps
            and metadata.gps is not None
            and metadata.gps.latitude is not None
            and metadata.gps.longitude is not None
        )
        has_pos = camera.has_position if camera else False
        has_ori = camera.has_orientation if camera else False
        has_complete_pose = camera.has_complete_pose if camera else False
        georeferencing_available = has_geotiff_georef or has_gps_georef or (current_state == PipelineState.STATE_D)

        summary = ProcessingSummary(
            request_id=req_id,
            state=current_state,
            relative_depth_available=True,
            camera_model_available=True,
            camera_position_available=has_pos,
            camera_orientation_available=has_ori,
            complete_camera_pose_available=has_complete_pose,
            metric_depth_available=(metric_depth is not None),
            georeferencing_available=georeferencing_available,
            dsm_available=(dsm_result is not None),
            dsm_type=dsm_type,
            validation_available=(validation_report is not None),
            model_name=relative_depth.model_name,
            device_used=relative_depth.device,
            total_time_ms=round(t_total, 2),
            timings_ms=timings,
            messages=messages,
            warnings=warnings,
            metadata=metadata,
            camera=camera,
            calibration=calib_result,
            validation=validation_report,
            artifacts=all_artifacts,
        )

        ArtifactManager.save_summary(req_id, summary)
        return summary
