"""Relative depth estimation engine using DepthAnything V2."""

import logging
import time
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from backend.config import settings
from backend.depth.model_loader import model_manager
from backend.models.depth import RelativeDepthMap, DepthMetadata

logger = logging.getLogger("depthwizard.estimator")


class RelativeDepthEstimator:
    """Inference runner producing raw floating-point relative depth maps."""

    @classmethod
    def validate_and_format_input(cls, image: np.ndarray) -> np.ndarray:
        """Validate and convert input array to (H, W, 3) uint8/float32 RGB."""
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid non-null numpy ndarray.")

        if image.size == 0:
            raise ValueError("Input image array is empty (size 0).")

        if not np.all(np.isfinite(image)):
            raise ValueError("Input image contains non-finite values (NaN or Inf).")

        # Handle grayscale (H, W) -> (H, W, 3)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=-1)
            elif image.shape[2] == 4:
                # RGBA -> RGB
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}. Expected 1, 3, or 4.")
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}D array. Expected 2D or 3D.")

        h, w = image.shape[:2]
        if h < 14 or w < 14:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small. Minimum supported size is 14x14 pixels.")

        if settings.max_image_dimension is not None:
            if h > settings.max_image_dimension or w > settings.max_image_dimension:
                raise ValueError(
                    f"Image dimensions ({w}x{h}) exceed maximum allowed dimension of {settings.max_image_dimension}px."
                )

        return image

    @classmethod
    def estimate_depth(
        cls,
        image_rgb: np.ndarray,
        input_size: int = settings.input_size,
    ) -> RelativeDepthMap:
        """Run DepthAnything V2 inference on an RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 or float32 numpy array.
            input_size: Target transformer backbone bound (constrained to multiples of 14).

        Returns:
            RelativeDepthMap with raw float32 predictions, timing breakdown, and model metadata.
        """
        t_total_start = time.perf_counter()

        # 1. Input validation & formatting
        image_rgb = cls.validate_and_format_input(image_rgb)
        orig_h, orig_w = image_rgb.shape[:2]

        device = model_manager._device
        model = model_manager.get_model()

        # 2. Preprocessing
        t_prep_start = time.perf_counter()
        if image_rgb.dtype == np.uint8:
            img_float = image_rgb.astype(np.float32) / 255.0
        else:
            img_float = image_rgb.astype(np.float32)
            if img_float.max() > 1.0:
                img_float = img_float / 255.0

        # ImageNet normalization statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std

        # Avoid unnecessary memory copies: ensure contiguous before torch.from_numpy
        if not img_norm.flags.c_contiguous:
            img_norm = np.ascontiguousarray(img_norm)

        # Tensor conversion (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)

        # Multiples of patch size (14) preserving aspect ratio
        scale = input_size / max(orig_h, orig_w)
        target_h = max(14, int(np.round((orig_h * scale) / 14.0) * 14))
        target_w = max(14, int(np.round((orig_w * scale) / 14.0) * 14))

        if (target_h, target_w) != (orig_h, orig_w):
            tensor_input = F.interpolate(tensor, size=(target_h, target_w), mode="bicubic", align_corners=True)
        else:
            tensor_input = tensor

        t_prep_end = time.perf_counter()
        preprocess_time_ms = (t_prep_end - t_prep_start) * 1000.0

        # 3. Model Inference (Proper inference mode, no gradients, eval mode)
        t_infer_start = time.perf_counter()
        with torch.inference_mode():
            if model.training:
                model.eval()

            # Use automatic mixed precision on CUDA for faster inference
            use_amp = device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(tensor_input)

            # Resize predicted depth back to EXACT original image dimensions
            # Use bicubic interpolation for smoother depth boundaries
            if pred.shape[-2:] != (orig_h, orig_w):
                if pred.ndim == 2:
                    pred = pred.unsqueeze(0).unsqueeze(0)
                elif pred.ndim == 3:
                    pred = pred.unsqueeze(1)
                # Ensure float32 before bicubic to avoid half-precision grid_sample issues
                pred = pred.float()
                pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bicubic", align_corners=True)

            raw_depth_np = pred.squeeze().cpu().numpy().astype(np.float32)

        # Release GPU memory after inference to prevent buildup across requests
        if device.type == "cuda":
            del pred, tensor_input, tensor
            torch.cuda.empty_cache()

        t_infer_end = time.perf_counter()
        inference_time_ms = (t_infer_end - t_infer_start) * 1000.0

        # 4. Postprocessing & Output Validation
        t_post_start = time.perf_counter()

        # Validate output shape matches input exactly
        if raw_depth_np.shape != (orig_h, orig_w):
            raise RuntimeError(
                f"Output depth shape {raw_depth_np.shape} does not match input image shape ({orig_h}, {orig_w})."
            )

        # DepthAnything V2 produces positive relative continuous depth
        raw_depth_np = np.clip(raw_depth_np, 1e-4, None)

        min_val = float(np.nanmin(raw_depth_np))
        max_val = float(np.nanmax(raw_depth_np))
        mean_val = float(np.nanmean(raw_depth_np))
        std_val = float(np.nanstd(raw_depth_np))

        t_post_end = time.perf_counter()
        postprocess_time_ms = (t_post_end - t_post_start) * 1000.0
        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0

        metadata = DepthMetadata(
            model_name=settings.default_model_name,
            encoder=settings.model_encoder,
            input_width=orig_w,
            input_height=orig_h,
            device=str(device),
            inference_time_ms=round(inference_time_ms, 2),
            is_metric=False,
            units="dimensionless",
            min_depth_m=None,
            max_depth_m=None,
            mean_depth_m=None,
        )

        return RelativeDepthMap(
            array=raw_depth_np,
            width=orig_w,
            height=orig_h,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
            model_name=settings.default_model_name,
            model_config_name=settings.model_encoder,
            device=str(device),
            inference_time_ms=round(inference_time_ms, 2),
            representation="relative_depth",
            is_metric=False,
            units="dimensionless",
            metadata=metadata,
        )
