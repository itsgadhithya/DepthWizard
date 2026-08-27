"""Data models for geometric and elevation validation against reference data."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ValidationMetrics(BaseModel):
    """Statistical accuracy and error metrics comparing prediction against reference ground truth."""
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    mean_bias_error: float = Field(..., description="Mean Bias Error (pred - ref)")
    max_absolute_error: float = Field(..., description="Maximum Absolute Error")
    median_absolute_error: float = Field(..., description="Median Absolute Error")
    le90: float = Field(..., description="90th percentile linear error (LE90)")
    le95: float = Field(..., description="95th percentile linear error (LE95)")
    percent_within_tolerance: float = Field(..., description="Percentage of samples with error <= tolerance")
    sample_count: int = Field(..., description="Number of evaluated reference points/pixels")
    valid_coverage_percent: float = Field(100.0, description="Percentage of valid overlapping reference region")
    units: str = "meters"


class ValidationReport(BaseModel):
    """Complete validation assessment report."""
    reference_source: str
    metrics: ValidationMetrics
    tolerance_threshold_m: float = 1.0
    passed: bool = True
    notes: Optional[str] = None
