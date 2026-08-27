"""Visual export helper coordinating generation of all presentation artifacts."""

from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from backend.models.depth import RelativeDepthMap, MetricDepthMap
from backend.models.dsm import DSMResult
from backend.visualization.colormaps import DepthColorMapper
from backend.visualization.hillshade import HillshadeGenerator


class VisualizationExporter:
    """Generates and persists visual presentation artifacts."""

    @classmethod
    def export_relative_depth_visual(
        cls,
        relative_depth: RelativeDepthMap,
        output_path: str,
        colormap: str = "turbo",
    ) -> str:
        """Generate and save 8-bit colormapped relative depth preview PNG."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rgb = DepthColorMapper.apply_colormap(
            data_2d=relative_depth.array,
            colormap_name=colormap,
            invert=False,
        )
        DepthColorMapper.save_image(rgb, str(path))
        return str(path.resolve())

    @classmethod
    def export_metric_depth_visual(
        cls,
        metric_depth: MetricDepthMap,
        output_path: str,
        colormap: str = "turbo",
    ) -> str:
        """Generate and save 8-bit colormapped metric depth preview PNG."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rgb = DepthColorMapper.apply_colormap(
            data_2d=metric_depth.array,
            colormap_name=colormap,
            valid_mask=metric_depth.valid_mask,
            invert=False,
        )
        DepthColorMapper.save_image(rgb, str(path))
        return str(path.resolve())

    @classmethod
    def export_dsm_visuals(
        cls,
        dsm: DSMResult,
        hillshade_path: str,
        colorized_path: str,
        colormap: str = "terrain",
    ) -> tuple[str, str]:
        """Generate both shaded relief hillshade and colorized relief for a DSM."""
        p_hill = Path(hillshade_path)
        p_color = Path(colorized_path)
        p_hill.parent.mkdir(parents=True, exist_ok=True)
        p_color.parent.mkdir(parents=True, exist_ok=True)

        valid_mask = (dsm.grid != dsm.nodata_value) & np.isfinite(dsm.grid)

        # 1. Grayscale hillshade
        hill_8u = HillshadeGenerator.compute_hillshade(
            elevation_grid=dsm.grid,
            resolution_m=dsm.resolution_m,
            nodata_value=dsm.nodata_value,
        )
        Image.fromarray(hill_8u).save(p_hill, format="PNG", optimize=True)

        # 2. Draped colorized relief
        color_rgb = DepthColorMapper.apply_colormap(
            data_2d=dsm.grid,
            colormap_name=colormap,
            valid_mask=valid_mask,
        )
        draped_rgb = HillshadeGenerator.blend_color_and_hillshade(color_rgb, hill_8u, blend_factor=0.45)
        Image.fromarray(draped_rgb).save(p_color, format="PNG", optimize=True)

        return str(p_hill.resolve()), str(p_color.resolve())
