"""
3D Mesh Exporter — PyVista StructuredGrid with correct RGB point-colour layout.
Persistent .ply export via trimesh.

IMPORTANT - VTK point ordering for StructuredGrid(dims=(W, H, 1)):
  VTK iterates x-fast, y-slow (Fortran-like for structured grids built via
  meshgrid with indexing='xy').  We use np.meshgrid default (indexing='xy')
  which gives gx[row, col] and gy[row, col], then reshape to (H*W, ...).
  The RGB array must also be flattened the same way: rgb.reshape(-1, 3)
  where rgb has shape (H, W, 3) — this is C-order (row-major), matching
  how VTK's StructuredGrid point IDs are assigned.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pyvista as pv
import trimesh

from src.config import pipeline_logger, RESULT_DIR, SPATIAL_DTYPE


class MeshExporterError(Exception):
    pass


class MeshExporter:
    """
    Builds a PyVista StructuredGrid terrain surface and exports it to .ply.
    Z-axis is scaled so horizontal and vertical units match realistically.
    RGB colours are baked as uint8 point data for accurate texture mapping.
    """

    def __init__(
        self,
        dem_fused: np.ndarray,
        rgb_texture: np.ndarray,
        pixel_spacing_m: float = 30.0,
        z_exaggeration: float = 1.0,
    ) -> None:
        if dem_fused.shape[:2] != rgb_texture.shape[:2]:
            raise MeshExporterError(
                f"DEM {dem_fused.shape} != RGB {rgb_texture.shape}"
            )
        self.dem  = dem_fused.astype(SPATIAL_DTYPE)
        self.rgb  = rgb_texture.astype(np.uint8)
        self.px   = float(pixel_spacing_m)
        self.z_ex = float(z_exaggeration)
        self.h, self.w = self.dem.shape

        # Pre-compute the world-space X/Y grids once; reused by both methods
        x = np.arange(self.w, dtype=SPATIAL_DTYPE) * self.px
        y = np.arange(self.h, dtype=SPATIAL_DTYPE) * self.px
        # meshgrid default indexing='xy': gx[row,col] = x[col], gy[row,col] = y[row]
        self._gx, self._gy = np.meshgrid(x, y)

        pipeline_logger.info(
            f"MeshExporter: {self.w}x{self.h} grid, "
            f"px={self.px:.1f}m, z_ex={self.z_ex}, "
            f"elev=[{self.dem.min():.1f}m, {self.dem.max():.1f}m]"
        )

    def create_structured_grid(self) -> pv.StructuredGrid:
        """
        Builds a GPU-ready PyVista StructuredGrid with RGB point colours.

        Colour accuracy guarantee
        -------------------------
        pv.StructuredGrid(gx, gy, gz) with (H,W) arrays assigns point IDs
        in row-major (C) order: point 0 = (gx[0,0], gy[0,0], gz[0,0]),
        point 1 = (gx[0,1], ...), ..., point W = (gx[1,0], ...).
        self.rgb has shape (H,W,3), so rgb.reshape(-1,3) flattens identically.
        Setting rgb=True in add_mesh() tells PyVista to interpret the 3-column
        uint8 array as literal R,G,B values — no colormap is applied.
        """
        gz = self.dem * self.z_ex

        grid = pv.StructuredGrid(self._gx, self._gy, gz)

        # Flatten RGB in C-order to match VTK point ordering
        rgb_flat = self.rgb.reshape(-1, 3)   # (H*W, 3) uint8
        grid.point_data["RGB"] = rgb_flat
        grid.point_data.active_scalars_name = "RGB"

        pipeline_logger.info(
            f"StructuredGrid: {grid.n_points:,} pts / {grid.n_cells:,} cells  "
            f"RGB dtype={rgb_flat.dtype}  range=[{rgb_flat.min()},{rgb_flat.max()}]"
        )
        return grid

    def get_elevation_extremes(self) -> Tuple[Tuple[float, float, float, float],
                                               Tuple[float, float, float, float]]:
        """
        Returns world-space (x, y, z_world, elev_m) for the highest and lowest
        valid terrain points.  z_world = elev_m * z_exaggeration.
        """
        gz = self.dem * self.z_ex

        idx_max = np.unravel_index(np.argmax(self.dem), self.dem.shape)
        idx_min = np.unravel_index(np.argmin(self.dem), self.dem.shape)

        def _pt(idx):
            row, col = idx
            return (
                float(self._gx[row, col]),
                float(self._gy[row, col]),
                float(gz[row, col]),
                float(self.dem[row, col]),
            )

        return _pt(idx_max), _pt(idx_min)

    def export_ply(self, base_name: str = "terrain_fused_model", output_path: Path | None = None) -> Path:
        """Exports a .ply to result/ or custom output_path (fast binary format)."""
        gz     = self.dem * self.z_ex
        verts  = np.column_stack((self._gx.ravel(), self._gy.ravel(), gz.ravel()))
        colors = self.rgb.reshape(-1, 3)

        idx = np.arange(self.h * self.w).reshape(self.h, self.w)
        f1  = np.column_stack((idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()))
        f2  = np.column_stack((idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()))

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=np.vstack((f1, f2)),
            vertex_colors=colors,
            process=False,
        )
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            out = RESULT_DIR / f"{base_name}.ply"

        mesh.export(str(out), file_type="ply")
        pipeline_logger.info(f"Saved PLY -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
        return out
