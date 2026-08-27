"""Metric calibration engine coordinating calibration strategies and MetricDepthMap creation."""

from typing import Optional, Tuple, Dict, Type
import numpy as np

from backend.models.calibration import (
    CalibrationMethod,
    CalibrationReference,
    CalibrationResult,
)
from backend.models.depth import RelativeDepthMap, MetricDepthMap
from backend.models.camera import CameraIntrinsics
from backend.metric.strategies import (
    BaseCalibrationStrategy,
    AltitudeGroundStrategy,
    GCPStrategy,
    KnownDistanceStrategy,
    ManualScaleStrategy,
    ReferenceDEMStrategy,
)


class MetricCalibrator:
    """Manages metric depth calibration with strict scientific integrity.

    CRITICAL RULE: Never invent metric scale when sufficient metric references are absent.
    GPS coordinates alone are NOT sufficient to establish scene depth scale.
    """

    _STRATEGIES: Dict[CalibrationMethod, Type[BaseCalibrationStrategy]] = {
        CalibrationMethod.ALTITUDE_GROUND: AltitudeGroundStrategy,
        CalibrationMethod.GCP: GCPStrategy,
        CalibrationMethod.KNOWN_DISTANCE: KnownDistanceStrategy,
        CalibrationMethod.MANUAL_SCALE: ManualScaleStrategy,
        CalibrationMethod.REFERENCE_DEM: ReferenceDEMStrategy,
    }

    @classmethod
    def resolve_effective_method(cls, reference: Optional[CalibrationReference]) -> CalibrationMethod:
        """Capability-driven strategy selector determining the strongest valid calibration method.

        Evaluates available metric references and selects the highest-confidence valid strategy:
            1. Ground Control Points (GCPs): Strongest local geometric anchors.
            2. Reference DEM: Dense spatial elevation reference.
            3. Known Distance: Metric scale from physical measurement.
            4. Altitude / Ground: AGL flight height reference.
            5. Manual Scale: Explicit user override.
            6. NONE: Fallback when no valid metric references exist.
        """
        if reference is None:
            return CalibrationMethod.NONE

        if reference.method != CalibrationMethod.NONE:
            return reference.method

        # Priority 1: Ground Control Points
        if reference.gcps and len(reference.gcps) > 0:
            return CalibrationMethod.GCP

        # Priority 2: Reference DEM
        if reference.reference_dem_array is not None or reference.reference_dem_path is not None:
            return CalibrationMethod.REFERENCE_DEM

        # Priority 3: Known Physical Distance Reference
        if reference.distance_references and len(reference.distance_references) > 0:
            return CalibrationMethod.KNOWN_DISTANCE

        # Priority 4: Flight Height AGL / (Camera Altitude + Ground Elevation)
        if reference.flight_height_agl_m is not None or (reference.camera_altitude_m is not None and reference.ground_elevation_m is not None):
            return CalibrationMethod.ALTITUDE_GROUND

        # Priority 5: Manual Scale
        if reference.manual_scale_factor is not None and reference.manual_scale_factor > 0:
            return CalibrationMethod.MANUAL_SCALE

        return CalibrationMethod.NONE

    @classmethod
    def calibrate(
        cls,
        relative_depth: Optional[RelativeDepthMap],
        reference: Optional[CalibrationReference] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        allow_provisional_fallback: bool = True,
        provisional_scale: Optional[float] = None,
    ) -> Tuple[Optional[MetricDepthMap], CalibrationResult]:
        """Calibrate a relative depth map into physical metric meters.

        Args:
            relative_depth: Raw relative depth map from DepthAnything V2.
            reference: User-provided or metadata-derived calibration measurements.
            intrinsics: Camera intrinsics model.
            allow_provisional_fallback: If True, uses configurable fixed scale when no real calibration succeeds.
            provisional_scale: Override scale factor for provisional fallback (default: settings.provisional_metric_scale).

        Returns:
            Tuple of:
                - Optional[MetricDepthMap] (None if calibration fails and provisional fallback is disabled)
                - CalibrationResult describing the outcome, confidence, and provisional status.
        """
        if relative_depth is None or relative_depth.array is None:
            res = CalibrationResult(
                success=False,
                method=CalibrationMethod.NONE,
                reason="Relative depth map is unavailable. Metric calibration cannot be performed.",
                confidence=0.0,
                is_provisional=False,
            )
            return None, res

        effective_method = cls.resolve_effective_method(reference)

        # 1. Attempt real calibration strategy if reference is provided
        calib_result: Optional[CalibrationResult] = None
        if reference is not None and effective_method != CalibrationMethod.NONE:
            strategy_cls = cls._STRATEGIES.get(effective_method)
            if strategy_cls:
                strategy = strategy_cls()
                calib_result = strategy.calibrate(relative_depth, reference, intrinsics)

        # 2. If real calibration succeeded with positive scale factor
        if (
            calib_result is not None
            and calib_result.success
            and calib_result.scale_factor is not None
            and calib_result.scale_factor > 0
        ):
            s = calib_result.scale_factor
            t = calib_result.shift_offset or 0.0

            metric_array = (relative_depth.array * s + t).astype(np.float32)
            valid_mask = metric_array > 0

            min_depth = float(np.nanmin(metric_array[valid_mask])) if np.any(valid_mask) else 0.0
            max_depth = float(np.nanmax(metric_array[valid_mask])) if np.any(valid_mask) else 0.0
            mean_depth = float(np.nanmean(metric_array[valid_mask])) if np.any(valid_mask) else 0.0
            std_depth = float(np.nanstd(metric_array[valid_mask])) if np.any(valid_mask) else 0.0

            metric_depth = MetricDepthMap(
                array=metric_array,
                width=relative_depth.width,
                height=relative_depth.height,
                min_depth_m=round(min_depth, 3),
                max_depth_m=round(max_depth, 3),
                mean_depth_m=round(mean_depth, 3),
                std_depth_m=round(std_depth, 3),
                calibration_method=effective_method.value,
                scale_factor=round(s, 6),
                shift_offset=round(t, 4),
                confidence_score=calib_result.confidence,
                is_metric=True,
                is_provisional=False,
                units="meters",
                valid_mask=valid_mask,
            )
            return metric_depth, calib_result

        # 3. Fallback: Check if Provisional Fixed-Scale Fallback is enabled
        from backend.config import settings

        fallback_enabled = allow_provisional_fallback and settings.enable_provisional_fallback
        if fallback_enabled:
            scale = float(provisional_scale if provisional_scale is not None else settings.provisional_metric_scale)
            shift = 0.0

            prov_result = CalibrationResult(
                success=True,
                method=CalibrationMethod.PROVISIONAL_FIXED_SCALE,
                scale_factor=scale,
                shift_offset=shift,
                confidence=0.20,
                reference_count=0,
                is_provisional=True,
                reason=(
                    "Provisional metric scale applied. Depth is scaled using a fixed engineering scale "
                    "and is NOT scientifically calibrated to physical ground truth."
                ),
                details={
                    "is_provisional": True,
                    "provisional_scale_factor": scale,
                    "engineering_fallback": True,
                    "real_calibration_attempted": effective_method.value if effective_method != CalibrationMethod.NONE else "none",
                    "real_calibration_failure_reason": calib_result.reason if (calib_result and not calib_result.success) else None,
                },
            )

            metric_array = (relative_depth.array * scale + shift).astype(np.float32)
            valid_mask = metric_array > 0

            min_depth = float(np.nanmin(metric_array[valid_mask])) if np.any(valid_mask) else 0.0
            max_depth = float(np.nanmax(metric_array[valid_mask])) if np.any(valid_mask) else 0.0
            mean_depth = float(np.nanmean(metric_array[valid_mask])) if np.any(valid_mask) else 0.0
            std_depth = float(np.nanstd(metric_array[valid_mask])) if np.any(valid_mask) else 0.0

            metric_depth = MetricDepthMap(
                array=metric_array,
                width=relative_depth.width,
                height=relative_depth.height,
                min_depth_m=round(min_depth, 3),
                max_depth_m=round(max_depth, 3),
                mean_depth_m=round(mean_depth, 3),
                std_depth_m=round(std_depth, 3),
                calibration_method=CalibrationMethod.PROVISIONAL_FIXED_SCALE.value,
                scale_factor=round(scale, 6),
                shift_offset=round(shift, 4),
                confidence_score=0.20,
                is_metric=True,
                is_provisional=True,
                units="meters",
                valid_mask=valid_mask,
            )
            return metric_depth, prov_result

        # 4. If provisional fallback is disabled, return uncalibrated result
        if calib_result is None:
            calib_result = CalibrationResult(
                success=False,
                method=CalibrationMethod.NONE,
                reason="No metric references provided and provisional fallback is disabled.",
                confidence=0.0,
                is_provisional=False,
            )

        return None, calib_result
