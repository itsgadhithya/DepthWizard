"""Artifact storage and output file management."""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import rasterio

from backend.config import settings
from backend.models.results import ArtifactInfo, ProcessingSummary
from backend.models.depth import RelativeDepthMap, MetricDepthMap
from backend.models.geometry import PointCloud3D
from backend.models.dsm import DSMResult
from backend.models.validation import ValidationReport
from backend.geometry.point_cloud import PointCloudIO
from backend.dsm.geotiff_exporter import GeoTIFFExporter
from backend.visualization.exports import VisualizationExporter


class ArtifactManager:
    """Manages storage directories and artifact registration for processing requests."""

    @classmethod
    def get_request_dir(cls, request_id: str) -> Path:
        """Return and create the artifact storage directory for a request."""
        req_dir = settings.artifacts_dir / request_id
        req_dir.mkdir(parents=True, exist_ok=True)
        return req_dir

    @classmethod
    def save_uploaded_image(cls, request_id: str, filename: str, image_bytes: bytes) -> ArtifactInfo:
        """Save uploaded source image."""
        req_dir = cls.get_request_dir(request_id)
        ext = Path(filename).suffix or ".jpg"
        target_path = req_dir / f"input_image{ext}"
        with open(target_path, "wb") as f:
            f.write(image_bytes)

        return ArtifactInfo(
            name="Input Image",
            filename=target_path.name,
            artifact_type="input_image",
            file_path=str(target_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{target_path.name}",
            size_bytes=target_path.stat().st_size,
            is_computational=True,
            is_visualization=True,
        )

    @classmethod
    def save_raw_relative_depth(cls, request_id: str, relative_depth: RelativeDepthMap) -> Dict[str, ArtifactInfo]:
        """Save raw float32 relative depth as .npy and 32-bit float TIFF."""
        req_dir = cls.get_request_dir(request_id)
        artifacts = {}

        # 1. Raw NumPy .npy array
        npy_path = req_dir / "raw_relative_depth.npy"
        np.save(npy_path, relative_depth.array.astype(np.float32))
        artifacts["raw_relative_depth_npy"] = ArtifactInfo(
            name="Raw Relative Depth (NumPy)",
            filename=npy_path.name,
            artifact_type="raw_depth_npy",
            file_path=str(npy_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{npy_path.name}",
            size_bytes=npy_path.stat().st_size,
            is_computational=True,
            is_visualization=False,
        )

        # 2. 32-bit Floating Point TIFF
        tif_path = req_dir / "raw_relative_depth_32f.tif"
        h, w = relative_depth.height, relative_depth.width
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=rasterio.float32,
        ) as dst:
            dst.write(relative_depth.array.astype(np.float32), 1)

        artifacts["raw_relative_depth_tif"] = ArtifactInfo(
            name="Raw Relative Depth (32-bit Float TIFF)",
            filename=tif_path.name,
            artifact_type="raw_depth_tif",
            file_path=str(tif_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{tif_path.name}",
            size_bytes=tif_path.stat().st_size,
            is_computational=True,
            is_visualization=False,
        )

        # 3. Separate 8-bit Colormapped Visual PNG
        png_path = req_dir / "relative_depth_visual.png"
        VisualizationExporter.export_relative_depth_visual(relative_depth, str(png_path), colormap="turbo")
        artifacts["relative_depth_visual_png"] = ArtifactInfo(
            name="Relative Depth Colormap (Visual Only)",
            filename=png_path.name,
            artifact_type="visual_png",
            file_path=str(png_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{png_path.name}",
            size_bytes=png_path.stat().st_size,
            is_computational=False,
            is_visualization=True,
        )

        return artifacts

    @classmethod
    def save_metric_depth(cls, request_id: str, metric_depth: MetricDepthMap) -> Dict[str, ArtifactInfo]:
        """Save calibrated metric depth array and visual preview."""
        req_dir = cls.get_request_dir(request_id)
        artifacts = {}

        # 1. Metric NumPy .npy
        npy_path = req_dir / "metric_depth_meters.npy"
        np.save(npy_path, metric_depth.array.astype(np.float32))
        artifacts["metric_depth_npy"] = ArtifactInfo(
            name="Metric Depth in Meters (NumPy)",
            filename=npy_path.name,
            artifact_type="metric_depth_npy",
            file_path=str(npy_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{npy_path.name}",
            size_bytes=npy_path.stat().st_size,
            is_computational=True,
            is_visualization=False,
        )

        # 2. Metric Visual PNG
        png_path = req_dir / "metric_depth_visual.png"
        VisualizationExporter.export_metric_depth_visual(metric_depth, str(png_path), colormap="turbo")
        artifacts["metric_depth_visual_png"] = ArtifactInfo(
            name="Metric Depth Colormap (Visual Only)",
            filename=png_path.name,
            artifact_type="visual_png",
            file_path=str(png_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{png_path.name}",
            size_bytes=png_path.stat().st_size,
            is_computational=False,
            is_visualization=True,
        )

        return artifacts

    @classmethod
    def save_point_cloud(cls, request_id: str, point_cloud: PointCloud3D) -> Dict[str, ArtifactInfo]:
        """Save 3D point cloud in PLY format and web JSON preview."""
        req_dir = cls.get_request_dir(request_id)
        artifacts = {}

        # 1. Binary PLY file
        ply_path = req_dir / "point_cloud.ply"
        PointCloudIO.save_ply(point_cloud, str(ply_path), binary=True)
        artifacts["point_cloud_ply"] = ArtifactInfo(
            name="3D Point Cloud (Stanford PLY)",
            filename=ply_path.name,
            artifact_type="point_cloud_ply",
            file_path=str(ply_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{ply_path.name}",
            size_bytes=ply_path.stat().st_size,
            is_computational=True,
            is_visualization=False,
        )

        # 2. WebGL JSON preview
        json_path = req_dir / "point_cloud_preview.json"
        web_dict = PointCloudIO.to_web_json(point_cloud, max_points=settings.point_cloud_subsample_limit)
        with open(json_path, "w") as f:
            json.dump(web_dict, f)

        artifacts["point_cloud_json"] = ArtifactInfo(
            name="3D Point Cloud WebGL Preview (JSON)",
            filename=json_path.name,
            artifact_type="point_cloud_json",
            file_path=str(json_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{json_path.name}",
            size_bytes=json_path.stat().st_size,
            is_computational=False,
            is_visualization=True,
        )

        return artifacts

    @classmethod
    def save_dsm(cls, request_id: str, dsm: DSMResult) -> Dict[str, ArtifactInfo]:
        """Save Digital Surface Model as GeoTIFF raster, NumPy float32 array, and visualization PNGs."""
        req_dir = cls.get_request_dir(request_id)
        artifacts = {}

        # 1. GeoTIFF / TIFF raster
        tif_path = req_dir / "dsm.tif"
        GeoTIFFExporter.save_dsm_geotiff(dsm, str(tif_path))
        dsm_label = "Digital Surface Model (GeoTIFF)" if not dsm.is_local else "Local Metric Digital Surface Model (TIFF)"
        artifacts["dsm_geotiff"] = ArtifactInfo(
            name=dsm_label,
            filename=tif_path.name,
            artifact_type="dsm_tif",
            file_path=str(tif_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{tif_path.name}",
            size_bytes=tif_path.stat().st_size,
            is_computational=True,
            is_visualization=False,
        )

        # 2. Raw Float32 DSM NumPy array (in meters, preserving NoData)
        npy_path = req_dir / "dsm_elevation_meters.npy"
        np.save(npy_path, dsm.grid.astype(np.float32))
        artifacts["dsm_npy"] = ArtifactInfo(
            name="DSM Surface Elevation in Meters (NumPy)",
            filename=npy_path.name,
            artifact_type="dsm_npy",
            file_path=str(npy_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{npy_path.name}",
            size_bytes=npy_path.stat().st_size,
            is_computational=True,
            is_visualization=False,
        )

        # 3. Hillshade & Color Relief PNGs
        hill_path = req_dir / "dsm_hillshade.png"
        color_path = req_dir / "dsm_color_relief.png"
        visual_path = req_dir / "dsm_visual.png"
        VisualizationExporter.export_dsm_visuals(
            dsm=dsm,
            hillshade_path=str(hill_path),
            colorized_path=str(color_path),
        )

        # Also write dsm_visual.png as primary visual
        import shutil
        shutil.copyfile(color_path, visual_path)

        artifacts["dsm_visual_png"] = ArtifactInfo(
            name="DSM Surface Colormap (Visual Only)",
            filename=visual_path.name,
            artifact_type="visual_png",
            file_path=str(visual_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{visual_path.name}",
            size_bytes=visual_path.stat().st_size,
            is_computational=False,
            is_visualization=True,
        )

        artifacts["dsm_hillshade_png"] = ArtifactInfo(
            name="DSM Shaded Relief Hillshade (Visual)",
            filename=hill_path.name,
            artifact_type="visual_png",
            file_path=str(hill_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{hill_path.name}",
            size_bytes=hill_path.stat().st_size,
            is_computational=False,
            is_visualization=True,
        )

        artifacts["dsm_color_relief_png"] = ArtifactInfo(
            name="DSM Draped Elevation Relief (Visual)",
            filename=color_path.name,
            artifact_type="visual_png",
            file_path=str(color_path.resolve()),
            download_url=f"/api/v1/artifacts/{request_id}/{color_path.name}",
            size_bytes=color_path.stat().st_size,
            is_computational=False,
            is_visualization=True,
        )

        # 4. DSM Provenance Metadata JSON
        if dsm.metadata:
            meta_path = req_dir / "dsm_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(dsm.metadata.model_dump(), f, indent=2)
            artifacts["dsm_metadata_json"] = ArtifactInfo(
                name="DSM Generation Metadata (JSON)",
                filename=meta_path.name,
                artifact_type="report_json",
                file_path=str(meta_path.resolve()),
                download_url=f"/api/v1/artifacts/{request_id}/{meta_path.name}",
                size_bytes=meta_path.stat().st_size,
                is_computational=True,
                is_visualization=True,
            )

        return artifacts

    @classmethod
    def save_validation_report(cls, request_id: str, report: ValidationReport) -> Dict[str, ArtifactInfo]:
        """Save validation report as JSON."""
        req_dir = cls.get_request_dir(request_id)
        json_path = req_dir / "validation_report.json"
        with open(json_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)

        return {
            "validation_report_json": ArtifactInfo(
                name="Validation Accuracy Report (JSON)",
                filename=json_path.name,
                artifact_type="report_json",
                file_path=str(json_path.resolve()),
                download_url=f"/api/v1/artifacts/{request_id}/{json_path.name}",
                size_bytes=json_path.stat().st_size,
                is_computational=True,
                is_visualization=True,
            )
        }

    @classmethod
    def save_summary(cls, request_id: str, summary: ProcessingSummary) -> str:
        """Save pipeline execution summary JSON."""
        req_dir = cls.get_request_dir(request_id)
        summary_path = req_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary.model_dump(), f, indent=2)
        return str(summary_path.resolve())

    @classmethod
    def get_artifact_file(cls, request_id: str, filename: str) -> Optional[Path]:
        """Retrieve existing artifact path safely."""
        target = (settings.artifacts_dir / request_id / filename).resolve()
        if target.is_file() and str(target).startswith(str(settings.artifacts_dir.resolve())):
            return target
        return None

    @classmethod
    def list_request_artifacts(cls, request_id: str) -> Optional[Dict[str, Any]]:
        """List all artifact files available for a request ID."""
        req_dir = (settings.artifacts_dir / request_id).resolve()
        if not req_dir.is_dir() or not str(req_dir).startswith(str(settings.artifacts_dir.resolve())):
            return None

        files = []
        for item in sorted(req_dir.glob("*")):
            if item.is_file():
                files.append({
                    "filename": item.name,
                    "download_url": f"/api/v1/artifacts/{request_id}/{item.name}",
                    "size_bytes": item.stat().st_size,
                })
        return {
            "request_id": request_id,
            "artifact_count": len(files),
            "artifacts": files,
        }

