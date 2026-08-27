"""Accuracy and error validation engine comparing estimated geometry to ground truth."""

from typing import List, Optional, Union, Tuple
import numpy as np

from backend.models.validation import ValidationMetrics, ValidationReport
from backend.models.calibration import GCPPoint
from backend.models.dsm import DSMResult


class ValidationEngine:
    """Computes photogrammetric and geospatial error metrics against ground truth references."""

    @classmethod
    def validate_points(
        cls,
        predicted_values: Union[List[float], np.ndarray],
        reference_values: Union[List[float], np.ndarray],
        tolerance_m: float = 1.0,
        reference_name: str = "Ground Control Points (GCPs)",
    ) -> ValidationReport:
        """Compute statistical accuracy metrics from arrays of predicted vs ground-truth values."""
        pred = np.asarray(predicted_values, dtype=np.float64)
        ref = np.asarray(reference_values, dtype=np.float64)

        valid_mask = np.isfinite(pred) & np.isfinite(ref)
        pred_valid = pred[valid_mask]
        ref_valid = ref[valid_mask]

        n_samples = len(pred_valid)
        if n_samples == 0:
            empty_metrics = ValidationMetrics(
                mae=0.0,
                rmse=0.0,
                mean_bias_error=0.0,
                max_absolute_error=0.0,
                median_absolute_error=0.0,
                le90=0.0,
                le95=0.0,
                percent_within_tolerance=0.0,
                sample_count=0,
                valid_coverage_percent=0.0,
            )
            return ValidationReport(
                reference_source=reference_name,
                metrics=empty_metrics,
                tolerance_threshold_m=tolerance_m,
                passed=False,
                notes="No overlapping valid points found between prediction and reference.",
            )

        residuals = pred_valid - ref_valid  # Error
        abs_residuals = np.abs(residuals)

        mae = float(np.mean(abs_residuals))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mbe = float(np.mean(residuals))
        max_err = float(np.max(abs_residuals))
        median_err = float(np.median(abs_residuals))
        le90 = float(np.percentile(abs_residuals, 90))
        le95 = float(np.percentile(abs_residuals, 95))

        within_tol_count = int(np.sum(abs_residuals <= tolerance_m))
        pct_within_tol = float(within_tol_count / n_samples * 100.0)

        metrics = ValidationMetrics(
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            mean_bias_error=round(mbe, 4),
            max_absolute_error=round(max_err, 4),
            median_absolute_error=round(median_err, 4),
            le90=round(le90, 4),
            le95=round(le95, 4),
            percent_within_tolerance=round(pct_within_tol, 2),
            sample_count=n_samples,
            valid_coverage_percent=100.0,
            units="meters",
        )

        passed = bool(rmse <= (tolerance_m * 1.5))

        return ValidationReport(
            reference_source=reference_name,
            metrics=metrics,
            tolerance_threshold_m=tolerance_m,
            passed=passed,
            notes=f"Evaluated across {n_samples} ground truth reference samples.",
        )

    @classmethod
    def validate_dsm_against_raster(
        cls,
        predicted_dsm: DSMResult,
        reference_grid: np.ndarray,
        tolerance_m: float = 1.0,
        reference_name: str = "Reference LiDAR DEM",
    ) -> ValidationReport:
        """Compare a predicted DSM grid directly against a reference elevation raster."""
        pred = predicted_dsm.grid.astype(np.float64)
        ref = reference_grid.astype(np.float64)

        if pred.shape != ref.shape:
            raise ValueError(f"Shape mismatch: Predicted {pred.shape} vs Reference {ref.shape}")

        valid_mask = (
            (pred != predicted_dsm.nodata_value) &
            (ref != predicted_dsm.nodata_value) &
            np.isfinite(pred) &
            np.isfinite(ref)
        )

        return cls.validate_points(
            predicted_values=pred[valid_mask],
            reference_values=ref[valid_mask],
            tolerance_m=tolerance_m,
            reference_name=reference_name,
        )
