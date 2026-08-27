"""Comprehensive unit tests for DepthAnything V2 relative depth estimation and integration."""

import pytest
import numpy as np
import torch

from backend.depth.estimator import RelativeDepthEstimator
from backend.depth.model_loader import DepthModelManager, model_manager
from backend.models.depth import RelativeDepthMap, DepthMetadata
from backend.visualization.colormaps import DepthColorMapper
from backend.storage.artifact_manager import ArtifactManager


def test_depth_model_manager_singleton_and_reuse():
    """Verify model initialization happens once and the model instance is reused across requests."""
    manager1 = DepthModelManager()
    manager2 = DepthModelManager()
    assert manager1 is manager2, "DepthModelManager must follow singleton pattern"

    model_a = model_manager.get_model()
    model_b = model_manager.get_model()
    assert model_a is model_b, "Model instance must be cached and reused across calls"

    info = model_manager.get_info()
    assert info["is_loaded"] is True
    assert info["parameter_count"] > 0
    assert "device" in info
    assert "cuda_available" in info


def test_depth_device_selection_and_cpu_fallback():
    """Verify GPU acceleration detection and CPU fallback behavior."""
    # Test explicit CPU device selection
    model_manager.set_device("cpu")
    info = model_manager.get_info()
    assert info["device"] == "cpu"

    # Test auto selection behavior
    auto_dev = model_manager._select_device("auto")
    if torch.cuda.is_available():
        assert auto_dev.type == "cuda"
    else:
        assert auto_dev.type == "cpu"

    # Test unavailable device fallback
    fallback_dev = model_manager._select_device("cuda" if not torch.cuda.is_available() else "nonexistent_device")
    assert fallback_dev.type == "cpu"


def test_depth_inference_mode_and_eval():
    """Verify inference runs without gradients and model is in eval mode."""
    model = model_manager.get_model()
    assert not model.training, "DepthAnything V2 model must be in eval mode"

    test_img = np.full((64, 64, 3), 120, dtype=np.uint8)
    rel_depth = RelativeDepthEstimator.estimate_depth(test_img)

    # Invariants
    assert not rel_depth.is_metric
    assert rel_depth.units == "dimensionless"
    assert rel_depth.representation == "relative_depth"


def test_raw_float32_depth_preservation(synthetic_rgb_image):
    """Verify DepthAnything V2 preserves raw floating-point depth without integer quantization."""
    rel_depth = RelativeDepthEstimator.estimate_depth(synthetic_rgb_image)

    assert isinstance(rel_depth, RelativeDepthMap)
    assert isinstance(rel_depth.array, np.ndarray)
    assert rel_depth.array.dtype == np.float32
    assert rel_depth.array.shape == synthetic_rgb_image.shape[:2]

    # Verify values are continuous floats and strictly positive
    assert rel_depth.min_val > 0.0
    assert rel_depth.max_val >= rel_depth.min_val
    assert np.all(np.isfinite(rel_depth.array))

    # Verify not quantized to 8-bit or 16-bit integer values
    unique_vals = len(np.unique(rel_depth.array))
    assert unique_vals > 200, "Depth map must preserve continuous floating-point depth values"


def test_separate_visualization_generation(synthetic_rgb_image):
    """Verify a separate colormapped visualization is generated and distinct from computational depth."""
    rel_depth = RelativeDepthEstimator.estimate_depth(synthetic_rgb_image)

    # Generate visualization
    visual_rgb = DepthColorMapper.apply_colormap(rel_depth.array, colormap_name="turbo")
    assert isinstance(visual_rgb, np.ndarray)
    assert visual_rgb.dtype == np.uint8
    assert visual_rgb.shape == (synthetic_rgb_image.shape[0], synthetic_rgb_image.shape[1], 3)

    # Test artifact separation via ArtifactManager
    req_id = "test_vis_sep"
    arts = ArtifactManager.save_raw_relative_depth(req_id, rel_depth)

    assert "raw_relative_depth_npy" in arts
    assert arts["raw_relative_depth_npy"].is_computational is True
    assert arts["raw_relative_depth_npy"].is_visualization is False

    assert "relative_depth_visual_png" in arts
    assert arts["relative_depth_visual_png"].is_computational is False
    assert arts["relative_depth_visual_png"].is_visualization is True


@pytest.mark.parametrize(
    "h, w",
    [
        (518, 518),  # Standard square
        (450, 800),  # 16:9 Landscape
        (800, 450),  # 9:16 Portrait
        (317, 523),  # Prime / arbitrary dimensions
        (64, 64),    # Small image
    ],
)
def test_arbitrary_image_sizes_and_aspect_ratios(h, w):
    """Verify model safely handles diverse resolutions and aspect ratios, resizing back exactly to input size."""
    test_img = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    rel_depth = RelativeDepthEstimator.estimate_depth(test_img)

    assert rel_depth.width == w
    assert rel_depth.height == h
    assert rel_depth.array.shape == (h, w)
    assert rel_depth.array.dtype == np.float32


def test_high_resolution_exceeding_4096():
    """Verify that images larger than 4096px (e.g. 4500x4500) process without artificial resolution caps."""
    # Create high-res 4500x4500 test array
    high_res_img = np.full((4500, 4500, 3), 128, dtype=np.uint8)
    rel_depth = RelativeDepthEstimator.estimate_depth(high_res_img)

    assert rel_depth.width == 4500
    assert rel_depth.height == 4500
    assert rel_depth.array.shape == (4500, 4500)
    assert rel_depth.array.dtype == np.float32



def test_input_validation_and_format_handling():
    """Verify input dimension validation, channel conversions (grayscale, RGBA), and error guards."""
    # 1. Grayscale (H, W) -> auto RGB
    gray_img = np.full((64, 64), 100, dtype=np.uint8)
    res_gray = RelativeDepthEstimator.estimate_depth(gray_img)
    assert res_gray.array.shape == (64, 64)

    # 2. RGBA (H, W, 4) -> auto RGB
    rgba_img = np.full((64, 64, 4), 150, dtype=np.uint8)
    res_rgba = RelativeDepthEstimator.estimate_depth(rgba_img)
    assert res_rgba.array.shape == (64, 64)

    # 3. Invalid: Image too small (<14x14)
    with pytest.raises(ValueError, match="too small"):
        RelativeDepthEstimator.estimate_depth(np.zeros((10, 10, 3), dtype=np.uint8))

    # 4. Invalid: Empty array
    with pytest.raises(ValueError, match="empty"):
        RelativeDepthEstimator.estimate_depth(np.zeros((0, 0, 3), dtype=np.uint8))

    # 5. Invalid: NaNs or Infs
    nan_img = np.full((64, 64, 3), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        RelativeDepthEstimator.estimate_depth(nan_img)


def test_timing_and_hardware_metadata_exposure(synthetic_rgb_image):
    """Verify inference timing and hardware metadata are accurately computed and exposed."""
    rel_depth = RelativeDepthEstimator.estimate_depth(synthetic_rgb_image)

    assert rel_depth.inference_time_ms > 0.0
    assert rel_depth.device != ""
    assert rel_depth.model_name == "depth_anything_v2_vits"
    assert rel_depth.model_config_name == "vits"

    assert rel_depth.metadata is not None
    assert isinstance(rel_depth.metadata, DepthMetadata)
    assert rel_depth.metadata.input_width == synthetic_rgb_image.shape[1]
    assert rel_depth.metadata.input_height == synthetic_rgb_image.shape[0]
    assert rel_depth.metadata.inference_time_ms > 0.0
    assert rel_depth.metadata.is_metric is False
