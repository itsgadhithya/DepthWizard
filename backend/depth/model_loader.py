"""Singleton Depth Model Manager to load DepthAnything V2 once and reuse across requests."""

import logging
import threading
import time
from typing import Dict, Any, Optional
import torch

from backend.config import settings
from backend.depth.depth_anything import DepthAnythingV2Model, load_depth_anything_model

logger = logging.getLogger("depthwizard.model_manager")


class DepthModelManager:
    """Thread-safe singleton managing the loaded DepthAnything V2 model instance."""

    _instance: Optional["DepthModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DepthModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._model: Optional[DepthAnythingV2Model] = None
        self._device: torch.device = self._select_device()
        self._encoder: str = settings.model_encoder
        self._checkpoint_path: Optional[str] = None
        self._load_lock = threading.Lock()
        self._load_time_ms: float = 0.0
        self._inference_count: int = 0
        self._initialized = True

    def _select_device(self, preferred_device: Optional[str] = None) -> torch.device:
        """Select execution device according to availability and configuration."""
        req = (preferred_device or settings.default_device).lower()

        if req == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        elif req == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("Apple MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        elif req == "auto":
            if torch.cuda.is_available():
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device("cpu")

    def initialize(
        self,
        checkpoint_path: Optional[str] = None,
        encoder: Optional[str] = None,
        device_str: Optional[str] = None,
    ) -> DepthAnythingV2Model:
        """Explicitly initialize or reconfigure the singleton model."""
        with self._load_lock:
            if encoder:
                self._encoder = encoder
            if checkpoint_path:
                self._checkpoint_path = checkpoint_path
            if device_str:
                self._device = self._select_device(device_str)

            t0 = time.perf_counter()
            self._model = load_depth_anything_model(
                checkpoint_path=self._checkpoint_path,
                encoder=self._encoder,
                device=self._device,
            )
            self._load_time_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"DepthAnything V2 model initialized on {self._device} in {self._load_time_ms:.2f}ms.")
            return self._model

    def get_model(self) -> DepthAnythingV2Model:
        """Retrieve or lazily initialize the singleton model instance."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    t0 = time.perf_counter()
                    self._model = load_depth_anything_model(
                        checkpoint_path=self._checkpoint_path,
                        encoder=self._encoder,
                        device=self._device,
                    )
                    self._load_time_ms = (time.perf_counter() - t0) * 1000.0
                    logger.info(f"Lazily loaded DepthAnything V2 on {self._device} in {self._load_time_ms:.2f}ms.")
        return self._model

    def set_device(self, device_str: str) -> None:
        """Dynamically switch execution device."""
        with self._load_lock:
            target_device = self._select_device(device_str)
            self._device = target_device
            if self._model is not None:
                self._model.to(self._device)
                self._model.eval()
            logger.info(f"Model execution device switched to: {self._device}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently resident in memory."""
        return self._model is not None

    def get_parameter_count(self) -> int:
        """Calculate total number of parameters in the active model."""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())

    def get_info(self) -> Dict[str, Any]:
        """Return operational details and hardware status of the depth model."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else None

        return {
            "model_name": settings.default_model_name,
            "encoder": self._encoder,
            "device": str(self._device),
            "cuda_available": cuda_avail,
            "cuda_device_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "is_loaded": self._model is not None,
            "load_time_ms": round(self._load_time_ms, 2),
            "parameter_count": self.get_parameter_count(),
            "inference_mode": "torch.inference_mode",
        }


# Global singleton instance
model_manager = DepthModelManager()
