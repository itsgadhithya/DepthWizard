"""Unit tests for validation engine error metrics."""

import pytest
import numpy as np

from backend.validation.metrics import ValidationEngine
from backend.models.validation import ValidationReport


def test_validation_engine_exact_match():
    """Verify metrics for exact agreement between prediction and ground truth."""
    pred = [10.0, 20.0, 30.0, 40.0]
    ref = [10.0, 20.0, 30.0, 40.0]

    report = ValidationEngine.validate_points(pred, ref, tolerance_m=0.5)
    assert isinstance(report, ValidationReport)
    assert report.metrics.mae == 0.0
    assert report.metrics.rmse == 0.0
    assert report.metrics.mean_bias_error == 0.0
    assert report.metrics.percent_within_tolerance == 100.0
    assert report.passed


def test_validation_engine_known_errors():
    """Verify statistical metrics calculation with known residual offsets."""
    pred = np.array([10.5, 20.0, 31.0, 39.5])
    ref = np.array([10.0, 20.0, 30.0, 40.0])
    # Residuals: [+0.5, 0.0, +1.0, -0.5]
    # Abs: [0.5, 0.0, 1.0, 0.5] -> MAE = 2.0 / 4 = 0.5
    # Sq: [0.25, 0.0, 1.0, 0.25] -> Mean Sq = 1.5 / 4 = 0.375 -> RMSE = sqrt(0.375) ~ 0.6124
    # MBE = (0.5 + 0 + 1.0 - 0.5) / 4 = 0.25

    report = ValidationEngine.validate_points(pred, ref, tolerance_m=0.5)
    assert pytest.approx(report.metrics.mae, rel=1e-3) == 0.5
    assert pytest.approx(report.metrics.rmse, rel=1e-3) == 0.6124
    assert pytest.approx(report.metrics.mean_bias_error, rel=1e-3) == 0.25
    assert report.metrics.max_absolute_error == 1.0
    assert report.metrics.sample_count == 4
    # Points with error <= 0.5: 3 out of 4 = 75%
    assert report.metrics.percent_within_tolerance == 75.0
