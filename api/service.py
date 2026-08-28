import io
import json
import math
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import cv2
from PIL import Image, ExifTags
import rasterio
from rasterio.transform import from_origin
import trimesh

from src.config import (
    ARTIFACTS_DIR, INPUT_DIR, SPATIAL_DTYPE,
    ensure_directories_exist, pipeline_logger
)
from src.geotiff_processor import GeoTIFFProcessor, GeoTIFFProcessorError
from src.depth_fusion import PrecisionTerrainFuser, FusionPipelineError
from src.mesh_exporter import MeshExporter
from src.depth_engine import DepthEngine
from api.models import (
    DepthWizardResponse, TimingsBreakdown, ArtifactInfo,
    ImageMetadata, GeospatialMetadata, CameraParameters,
    CalibrationInfo, ValidationMetrics
)

logger = pipeline_logger


class DepthWizardService:
    """
    Main service orchestrating the DepthWizard V2 pipeline for API requests.
    """

    def __init__(self):
        ensure_directories_exist()
        self.depth_engine = DepthEngine()

    def process_request(
        self,
        image_bytes: bytes,
        filename: str,
        depth_bytes: Optional[bytes] = None,
        depth_filename: Optional[str] = None,
        modulation_weight: float = 0.35,
        z_exaggeration: float = 1.0,
        pixel_spacing_m: Optional[float] = None,
        focal_length_px: Optional[float] = None,
    ) -> DepthWizardResponse:
        """
        Executes the end-to-end processing pipeline for an uploaded image.
        """
        t_start = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        req_dir = ARTIFACTS_DIR / request_id
        req_dir.mkdir(parents=True, exist_ok=True)

        timings = TimingsBreakdown()
        messages: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, ArtifactInfo] = {}

        # ── 1. Ingestion ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        ext = Path(filename).suffix.lower() if filename else ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            ext = ".jpg"

        input_path = req_dir / f"input_image{ext}"
        with open(input_path, "wb") as f:
            f.write(image_bytes)

        # Register input image artifact
        artifacts["input_image"] = ArtifactInfo(
            filename=input_path.name,
            artifact_type="input_image",
            download_url=f"/api/v1/artifacts/{request_id}/{input_path.name}",
            size_bytes=len(image_bytes),
            media_type=self._get_media_type(input_path.name),
            description="Original uploaded image",
        )

        is_geotiff = False
        geo_meta: Optional[Dict[str, Any]] = None
        image_rgb: Optional[np.ndarray] = None
        dem_data: Optional[np.ndarray] = None

        # Check if file is GeoTIFF with rasterio
        if ext in [".tif", ".tiff"]:
            try:
                proc = GeoTIFFProcessor(input_path)
                dem_data, image_rgb, geo_meta = proc.read_dem_and_texture()
                is_geotiff = True
                messages.append(f"Loaded GeoTIFF dataset with CRS={geo_meta.get('crs')}")
            except Exception as e:
                logger.warning(f"File {filename} is not a valid GeoTIFF DEM ({e}). Treating as standard image.")
                is_geotiff = False

        if image_rgb is None:
            # Load standard image with PIL/OpenCV
            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                pil_img = pil_img.convert("RGB")
                image_rgb = np.array(pil_img)
            except Exception as e:
                raise ValueError(f"Invalid or corrupted image format: {e}")

        h, w, c = image_rgb.shape
        timings.ingestion_ms = (time.perf_counter() - t0) * 1000.0
        messages.append(f"Image ingested successfully ({w}x{h}, {c} channels).")

        # ── 2. Metadata Extraction ────────────────────────────────────────────
        t0 = time.perf_counter()
        exif_dict: Dict[str, Any] = {}
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            raw_exif = pil_img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Only serialize basic types
                    if isinstance(value, (str, int, float)):
                        exif_dict[tag] = value
                    else:
                        exif_dict[tag] = str(value)
        except Exception:
            pass

        geospatial_model: Optional[GeospatialMetadata] = None
        if is_geotiff and geo_meta:
            bounds_obj = geo_meta.get("bounds")
            bounds_dict = None
            if bounds_obj:
                bounds_dict = {
                    "left": float(bounds_obj.left),
                    "bottom": float(bounds_obj.bottom),
                    "right": float(bounds_obj.right),
                    "top": float(bounds_obj.top),
                }

            transform_list = None
            if geo_meta.get("transform"):
                transform_list = list(geo_meta["transform"])[:6]

            # Calculate pixel spacing if bounds are in geographical/projected units
            calc_pixel_spacing = pixel_spacing_m or 30.0
            if bounds_obj:
                try:
                    mid_lat = math.radians((bounds_obj.top + bounds_obj.bottom) / 2)
                    x_span_m = (bounds_obj.right - bounds_obj.left) * 111_320 * math.cos(mid_lat)
                    y_span_m = (bounds_obj.top - bounds_obj.bottom) * 111_320
                    px_x = abs(x_span_m / w)
                    px_y = abs(y_span_m / h)
                    if px_x > 0.01 and px_y > 0.01:
                        calc_pixel_spacing = (px_x + px_y) / 2.0
                except Exception:
                    calc_pixel_spacing = 30.0

            geospatial_model = GeospatialMetadata(
                crs=str(geo_meta.get("crs")),
                bounds=bounds_dict,
                transform=transform_list,
                pixel_spacing_m=calc_pixel_spacing,
                nodata=geo_meta.get("nodata"),
            )
        else:
            calc_pixel_spacing = pixel_spacing_m or 1.0

        image_metadata = ImageMetadata(
            image_width=w,
            image_height=h,
            channels=c,
            format=ext.replace(".", "").upper(),
            exif=exif_dict,
            geospatial=geospatial_model,
        )
        timings.metadata_extraction_ms = (time.perf_counter() - t0) * 1000.0

        # ── 3. Depth Inference ────────────────────────────────────────────────
        t0 = time.perf_counter()
        if depth_bytes and len(depth_bytes) > 0:
            # User supplied custom depth map
            depth_ext = Path(depth_filename or "depth.npy").suffix.lower()
            upload_depth_path = req_dir / f"uploaded_depth{depth_ext}"
            with open(upload_depth_path, "wb") as f:
                f.write(depth_bytes)

            fuser = PrecisionTerrainFuser(modulation_weight=modulation_weight)
            raw_depth = fuser.load_depth_anything_map(upload_depth_path, target_shape=(h, w))
            depth_vis_16 = (raw_depth * 65535.0).astype(np.uint16)
            depth_vis_colored = cv2.applyColorMap((raw_depth * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
            depth_vis_colored = cv2.cvtColor(depth_vis_colored, cv2.COLOR_BGR2RGB)
            inference_time = (time.perf_counter() - t0) * 1000.0
            messages.append("Using uploaded custom depth map.")
        else:
            raw_depth, depth_vis_colored, depth_vis_16, inference_time = self.depth_engine.estimate_depth(image_rgb)
            messages.append(f"Depth estimation completed via {self.depth_engine.model_name}.")

        # Save relative depth artifacts
        npy_path = req_dir / "relative_depth.npy"
        png_16_path = req_dir / "relative_depth_16bit.png"
        png_color_path = req_dir / "relative_depth_colored.png"

        np.save(npy_path, raw_depth)
        cv2.imwrite(str(png_16_path), depth_vis_16)
        cv2.imwrite(str(png_color_path), cv2.cvtColor(depth_vis_colored, cv2.COLOR_RGB2BGR))

        artifacts["relative_depth"] = ArtifactInfo(
            filename=npy_path.name,
            artifact_type="relative_depth",
            download_url=f"/api/v1/artifacts/{request_id}/{npy_path.name}",
            size_bytes=npy_path.stat().st_size,
            media_type="application/octet-stream",
            description="Raw relative depth array (float64 numpy format)",
        )
        artifacts["relative_depth_16bit"] = ArtifactInfo(
            filename=png_16_path.name,
            artifact_type="visualization",
            download_url=f"/api/v1/artifacts/{request_id}/{png_16_path.name}",
            size_bytes=png_16_path.stat().st_size,
            media_type="image/png",
            description="Normalized 16-bit lossless grayscale depth visualization",
        )
        artifacts["relative_depth_colored"] = ArtifactInfo(
            filename=png_color_path.name,
            artifact_type="visualization",
            download_url=f"/api/v1/artifacts/{request_id}/{png_color_path.name}",
            size_bytes=png_color_path.stat().st_size,
            media_type="image/png",
            description="Inferno colormap relative depth visualization",
        )

        timings.depth_inference_ms = (time.perf_counter() - t0) * 1000.0

        # ── 4. Camera Modeling ────────────────────────────────────────────────
        t0 = time.perf_counter()
        if focal_length_px:
            fx = fy = float(focal_length_px)
            cx, cy = w / 2.0, h / 2.0
            fov = 2.0 * math.atan(w / (2.0 * fx)) * (180.0 / math.pi)
            is_cam_estimated = False
        else:
            # Standard pinhole estimation based on typical 55 degree horizontal FOV
            fov = 55.0
            fx = fy = (w / 2.0) / math.tan(math.radians(fov / 2.0))
            cx, cy = w / 2.0, h / 2.0
            is_cam_estimated = True

        camera_model = CameraParameters(
            fx=round(fx, 2),
            fy=round(fy, 2),
            cx=round(cx, 2),
            cy=round(cy, 2),
            fov_deg=round(fov, 2),
            estimated=is_cam_estimated,
        )
        timings.camera_modeling_ms = (time.perf_counter() - t0) * 1000.0

        # ── 5. 3D Reconstruction & DSM Generation ────────────────────────────
        t0_3d = time.perf_counter()
        t0_dsm = time.perf_counter()

        metric_available = False
        georeferencing_available = False
        dsm_available = False
        state = "STATE_B"
        calibration_info = CalibrationInfo(method="none", calibrated=False)

        if is_geotiff and dem_data is not None:
            # Case 1: GeoTIFF with absolute DEM elevation
            messages.append("Executing high-precision DEM + DepthAnythingV2 fusion.")
            fuser = PrecisionTerrainFuser(modulation_weight=modulation_weight)
            dem_fused = fuser.fuse(dem_data, raw_depth)

            # Export fused DSM as GeoTIFF
            dsm_path = req_dir / "fused_dsm.tif"
            transform = geo_meta.get("transform") if geo_meta else from_origin(0, 0, 1, 1)
            crs_val = geo_meta.get("crs", "EPSG:4326") if geo_meta else "EPSG:4326"

            with rasterio.open(
                dsm_path,
                "w",
                driver="GTiff",
                height=dem_fused.shape[0],
                width=dem_fused.shape[1],
                count=1,
                dtype=rasterio.float64,
                crs=crs_val,
                transform=transform,
            ) as dst:
                dst.write(dem_fused, 1)

            artifacts["dsm"] = ArtifactInfo(
                filename=dsm_path.name,
                artifact_type="dsm",
                download_url=f"/api/v1/artifacts/{request_id}/{dsm_path.name}",
                size_bytes=dsm_path.stat().st_size,
                media_type="image/tiff",
                description="Fused high-resolution Digital Surface Model GeoTIFF",
            )

            # Generate 3D StructuredGrid and PLY mesh
            exporter = MeshExporter(
                dem_fused=dem_fused,
                rgb_texture=image_rgb,
                pixel_spacing_m=calc_pixel_spacing,
                z_exaggeration=z_exaggeration,
            )
            mesh_path = req_dir / "terrain_3d_mesh.ply"
            exporter.export_ply(base_name="terrain_3d_mesh", output_path=mesh_path)

            artifacts["3d_model"] = ArtifactInfo(
                filename="terrain_3d_mesh.ply",
                artifact_type="3d_model",
                download_url=f"/api/v1/artifacts/{request_id}/terrain_3d_mesh.ply",
                size_bytes=mesh_path.stat().st_size if mesh_path.exists() else None,
                media_type="application/x-ply",
                description="3D terrain surface mesh (Binary PLY with baked RGB vertex colors)",
            )

            metric_available = True
            georeferencing_available = (geo_meta is not None and bool(geo_meta.get("crs")))
            dsm_available = True
            state = "STATE_C"
            calibration_info = CalibrationInfo(
                method="geotiff_dem_fusion",
                scale_factor=1.0,
                reference_type="geotiff_elevation_band",
                reference_value=float(dem_data.mean()),
                calibrated=True,
            )
            timings.geospatial_and_dsm_ms = (time.perf_counter() - t0_dsm) * 1000.0
            timings.reconstruction_3d_ms = (time.perf_counter() - t0_3d) * 1000.0

        else:
            # Case 2: Standard optical image (JPEG/PNG) without metric calibration
            messages.append("Standard optical image: Generating relative 3D mesh via pinhole back-projection.")
            # Uncalibrated relative backprojection
            z_rel = (1.0 - raw_depth * 0.7) * 100.0  # relative depth scale

            # Backproject pixels (u, v) -> (X, Y, Z)
            u_coords = np.arange(w, dtype=SPATIAL_DTYPE)
            v_coords = np.arange(h, dtype=SPATIAL_DTYPE)
            uu, vv = np.meshgrid(u_coords, v_coords)

            xx = (uu - cx) * z_rel / fx
            yy = (vv - cy) * z_rel / fy
            zz = z_rel * z_exaggeration

            # Build triangle mesh
            verts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            colors = image_rgb.reshape(-1, 3)

            idx = np.arange(h * w).reshape(h, w)
            f1 = np.column_stack((idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()))
            f2 = np.column_stack((idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()))
            faces = np.vstack((f1, f2))

            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)
            mesh_path = req_dir / "terrain_3d_mesh.ply"
            mesh.export(str(mesh_path), file_type="ply")

            artifacts["3d_model"] = ArtifactInfo(
                filename="terrain_3d_mesh.ply",
                artifact_type="3d_model",
                download_url=f"/api/v1/artifacts/{request_id}/terrain_3d_mesh.ply",
                size_bytes=mesh_path.stat().st_size,
                media_type="application/x-ply",
                description="Relative 3D mesh model (Binary PLY with baked RGB vertex colors)",
            )

            metric_available = False
            georeferencing_available = False
            dsm_available = False
            state = "STATE_B"
            calibration_info = CalibrationInfo(method="none", calibrated=False)
            warnings.append("No metric/geospatial DEM reference found. Outputs are in relative model coordinates.")

            timings.reconstruction_3d_ms = (time.perf_counter() - t0_3d) * 1000.0
            timings.geospatial_and_dsm_ms = 0.0

        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        response = DepthWizardResponse(
            request_id=request_id,
            state=state,
            relative_depth_available=True,
            camera_model_available=True,
            metric_depth_available=metric_available,
            georeferencing_available=georeferencing_available,
            dsm_available=dsm_available,
            validation_available=False,
            model_name=self.depth_engine.model_name,
            device_used=self.depth_engine.device,
            total_time_ms=round(total_time_ms, 2),
            timings_ms=timings,
            messages=messages,
            warnings=warnings,
            metadata=image_metadata,
            camera=camera_model,
            calibration=calibration_info,
            validation=None,
            artifacts=artifacts,
        )

        # Save session response JSON for state persistence
        try:
            resp_json_path = req_dir / "response.json"
            with open(resp_json_path, "w", encoding="utf-8") as f:
                f.write(response.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Could not persist session response JSON: {e}")

        return response

    def get_session(self, request_id: str) -> Optional[DepthWizardResponse]:
        """
        Retrieves stored DepthWizardResponse for a given request ID.
        """
        safe_id = Path(request_id).name
        req_dir = ARTIFACTS_DIR / safe_id
        if not req_dir.exists() or not req_dir.is_dir():
            return None

        resp_json_path = req_dir / "response.json"
        if resp_json_path.exists():
            try:
                with open(resp_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return DepthWizardResponse.model_validate(data)
            except Exception as e:
                logger.warning(f"Failed to parse stored session response.json: {e}")

        # Fallback: Reconstruct response from artifact files in the directory
        artifacts: Dict[str, ArtifactInfo] = {}
        has_dsm = False
        has_mesh = False

        for f in req_dir.iterdir():
            if not f.is_file():
                continue
            name = f.name
            size = f.stat().st_size
            media_type = self._get_media_type(name)
            if name.startswith("input_image"):
                artifacts["input_image"] = ArtifactInfo(
                    filename=name, artifact_type="input_image",
                    download_url=f"/api/v1/artifacts/{safe_id}/{name}",
                    size_bytes=size, media_type=media_type,
                    description="Original uploaded image"
                )
            elif name == "terrain_3d_mesh.ply":
                has_mesh = True
                artifacts["3d_model"] = ArtifactInfo(
                    filename=name, artifact_type="3d_model",
                    download_url=f"/api/v1/artifacts/{safe_id}/{name}",
                    size_bytes=size, media_type=media_type,
                    description="3D terrain surface mesh"
                )
            elif name == "fused_dsm.tif":
                has_dsm = True
                artifacts["dsm"] = ArtifactInfo(
                    filename=name, artifact_type="dsm",
                    download_url=f"/api/v1/artifacts/{safe_id}/{name}",
                    size_bytes=size, media_type=media_type,
                    description="Digital Surface Model GeoTIFF"
                )
            elif name == "relative_depth.npy":
                artifacts["relative_depth"] = ArtifactInfo(
                    filename=name, artifact_type="relative_depth",
                    download_url=f"/api/v1/artifacts/{safe_id}/{name}",
                    size_bytes=size, media_type=media_type,
                    description="Raw relative depth array"
                )
            elif name == "relative_depth_colored.png":
                artifacts["relative_depth_colored"] = ArtifactInfo(
                    filename=name, artifact_type="visualization",
                    download_url=f"/api/v1/artifacts/{safe_id}/{name}",
                    size_bytes=size, media_type=media_type,
                    description="Inferno colormap relative depth visual"
                )

        state = "STATE_C" if has_dsm else "STATE_B"
        return DepthWizardResponse(
            request_id=safe_id,
            state=state,
            relative_depth_available=True,
            camera_model_available=True,
            metric_depth_available=has_dsm,
            georeferencing_available=has_dsm,
            dsm_available=has_dsm,
            validation_available=False,
            model_name=self.depth_engine.model_name,
            device_used=self.depth_engine.device,
            total_time_ms=2500.0,
            timings_ms=TimingsBreakdown(),
            messages=["Session restored from artifact store."],
            warnings=[],
            metadata=ImageMetadata(image_width=512, image_height=512, channels=3, format="TIF" if has_dsm else "JPG"),
            camera=CameraParameters(fx=512.0, fy=512.0, cx=256.0, cy=256.0, fov_deg=55.0, estimated=True),
            calibration=CalibrationInfo(method="geotiff_dem_fusion" if has_dsm else "none", calibrated=has_dsm),
            validation=None,
            artifacts=artifacts,
        )

    def list_samples(self) -> List[Dict[str, Any]]:
        """
        Lists bundled sample datasets available for 1-click execution.
        """
        samples = []
        if not INPUT_DIR.exists():
            return samples

        sample_definitions = {
            "sample_terrain.tif": {
                "id": "himalayas_dem",
                "name": "Himalayas GeoTIFF DEM (Multi-Band)",
                "description": "Calibrated 4-band GeoTIFF with RGB satellite imagery + DEM elevation band (ISRO Bengaluru / Mt Everest).",
                "type": "GeoTIFF DEM",
                "recommended_z": 1.5,
                "recommended_mod": 0.35,
            },
            "test_optical.jpg": {
                "id": "aerial_optical",
                "name": "High-Res Aerial Optical Survey",
                "description": "Uncalibrated optical RGB reconnaissance photo for monocular relative depth estimation.",
                "type": "Optical RGB",
                "recommended_z": 1.0,
                "recommended_mod": 0.40,
            }
        }

        for file_path in INPUT_DIR.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                name = file_path.name
                info = sample_definitions.get(name, {
                    "id": file_path.stem,
                    "name": file_path.name,
                    "description": f"Sample dataset {file_path.name}",
                    "type": file_path.suffix.upper().replace(".", ""),
                    "recommended_z": 1.0,
                    "recommended_mod": 0.35,
                })
                samples.append({
                    "id": info["id"],
                    "filename": name,
                    "name": info["name"],
                    "description": info["description"],
                    "type": info["type"],
                    "size_bytes": file_path.stat().st_size,
                    "recommended_z": info["recommended_z"],
                    "recommended_mod": info["recommended_mod"],
                    "download_url": f"/api/v1/samples/{name}",
                })
        return samples

    def get_sample_path(self, filename: str) -> Optional[Path]:
        """
        Returns safe path to a sample file in INPUT_DIR.
        """
        safe_name = Path(filename).name
        target_path = INPUT_DIR / safe_name
        if target_path.exists() and target_path.is_file():
            return target_path
        return None

    def process_sample(
        self,
        sample_name: str,
        modulation_weight: float = 0.35,
        z_exaggeration: float = 1.0,
        pixel_spacing_m: Optional[float] = None,
        focal_length_px: Optional[float] = None,
    ) -> DepthWizardResponse:
        """
        Loads and executes a sample dataset directly on the server.
        """
        sample_path = self.get_sample_path(sample_name)
        if not sample_path:
            raise ValueError(f"Sample dataset '{sample_name}' not found in {INPUT_DIR}.")

        with open(sample_path, "rb") as f:
            image_bytes = f.read()

        return self.process_request(
            image_bytes=image_bytes,
            filename=sample_path.name,
            modulation_weight=modulation_weight,
            z_exaggeration=z_exaggeration,
            pixel_spacing_m=pixel_spacing_m,
            focal_length_px=focal_length_px,
        )

    def _get_media_type(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".npy": "application/octet-stream",
            ".ply": "application/x-ply",
        }
        return mapping.get(ext, "application/octet-stream")

