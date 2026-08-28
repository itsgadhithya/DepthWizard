"""
High-Performance WASD Fly-Through Renderer
Uses PyVista + VTK with dedicated GPU enforcement (NVIDIA via env-var priority).

Features
--------
- Correct RGB texture mapping (rgb=True on uint8 point data)
- Min / Max elevation sphere markers with floating height labels
- Metric scale bar overlay
- Clean dark theme with zero UI clutter
- WASD + Q/E + mouse-look FPS camera
"""

import os

# ── Force Windows to prefer the NVIDIA discrete GPU (Optimus / hybrid graphics)
# These must be set BEFORE any OpenGL/VTK import.
os.environ["CUDA_VISIBLE_DEVICES"]       = "0"   # CUDA picks discrete GPU 0
os.environ["NVAPI_ENABLE_GPU_SELECTION"] = "1"   # NVIDIA Optimus hint
# NOTE: Do NOT set VTK_DEFAULT_OPENGL_WINDOW — it causes VTK to look for OSMesa
#       which is not bundled with PyVista on Windows and will crash the window.

import numpy as np
import pyvista as pv
from pyvista import themes

from src.config import pipeline_logger


# ── PyVista global theme (clean, dark, professional) ─────────────────────────
_theme = themes.DarkTheme()
_theme.show_edges  = False
_theme.lighting    = True
_theme.background  = "black"
_theme.font.color  = "white"
pv.global_theme.load_theme(_theme)


class TerrainFlythroughViewer:
    """
    Mission-critical 3D terrain fly-through viewer.
    Accepts the PyVista StructuredGrid produced by MeshExporter and optional
    elevation-extreme coordinates from MeshExporter.get_elevation_extremes().
    """

    def __init__(
        self,
        grid: pv.StructuredGrid,
        elevation_extremes=None,   # ((x,y,z,elev_m), (x,y,z,elev_m)) max then min
        pixel_spacing_m: float = 100.0,
        move_speed: float | None = None,
        title: str = "ISRO Mission 3D Terrain Fly-Through",
    ) -> None:
        self.grid = grid

        # Auto-compute move speed from terrain diagonal
        bounds = grid.bounds
        diag   = np.sqrt(
            (bounds[1]-bounds[0])**2 +
            (bounds[3]-bounds[2])**2 +
            (bounds[5]-bounds[4])**2
        )
        self.speed = move_speed if move_speed is not None else max(diag * 0.008, 10.0)
        pipeline_logger.info(f"Viewer: auto move_speed={self.speed:.1f}  diagonal={diag:.0f}")

        # ── Plotter ────────────────────────────────────────────────────────
        self.pl = pv.Plotter(title=title, lighting="three lights", off_screen=False)
        self.pl.set_background("black")
        self.pl.hide_axes()
        self.pl.remove_all_lights()

        # Directional sun light for hillshading
        sun = pv.Light(position=(0.5, 0.5, 1.0), light_type="scene light")
        sun.intensity = 1.0
        self.pl.add_light(sun)

        # ── Terrain mesh — RGB colour data ────────────────────────────────
        # rgb=True tells PyVista to interpret the uint8 (N,3) scalar array as
        # literal R,G,B values with NO colormap applied.
        self.pl.add_mesh(
            self.grid,
            scalars="RGB",
            rgb=True,
            show_edges=False,
            smooth_shading=True,
            ambient=0.20,
            diffuse=0.85,
            specular=0.05,
        )

        # ── Topographic annotations ───────────────────────────────────────
        if elevation_extremes is not None:
            self._add_elevation_markers(elevation_extremes)

        self._add_scale_bar(pixel_spacing_m)

        self._reset_camera()
        self._bind_keys()
        pipeline_logger.info("TerrainFlythroughViewer ready.  W/A/S/D | Q/E | Mouse | R=reset")

    # ── Topographic annotation helpers ────────────────────────────────────────

    def _add_elevation_markers(self, extremes) -> None:
        """
        Adds red sphere + floating label at the highest and lowest terrain point.
        extremes = ((x_max, y_max, z_max, elev_max_m), (x_min, y_min, z_min, elev_min_m))
        """
        (xH, yH, zH, eH), (xL, yL, zL, eL) = extremes

        # Sphere radius ~ 0.4% of diagonal for visibility
        bounds = self.grid.bounds
        diag = np.sqrt(
            (bounds[1]-bounds[0])**2 +
            (bounds[3]-bounds[2])**2 +
            (bounds[5]-bounds[4])**2
        )
        r = max(diag * 0.004, 50.0)

        label_offset = r * 4.0   # float text this many units above the sphere

        for (x, y, z, elev_m), tag in [
            ((xH, yH, zH, eH), "MAX"),
            ((xL, yL, zL, eL), "MIN"),
        ]:
            sphere = pv.Sphere(radius=r, center=(x, y, z))
            self.pl.add_mesh(sphere, color="red", lighting=False)

            label_text = f"{tag}: {elev_m:.1f} m"
            # pv.PolyData point label
            pt = pv.PolyData(np.array([[x, y, z + label_offset]]))
            pt["labels"] = [label_text]
            self.pl.add_point_labels(
                pt,
                "labels",
                font_size=14,
                text_color="white",
                bold=True,
                shadow=True,
                show_points=False,
                always_visible=True,
            )
            pipeline_logger.info(f"Elevation marker: {label_text} at ({x:.0f}, {y:.0f}, {z:.0f})")

    def _add_scale_bar(self, pixel_spacing_m: float) -> None:
        """
        Draws a 3D scale bar on the terrain surface in the bottom-left corner.
        Chooses a round-number length that is ~10% of the terrain X-extent.
        """
        b = self.grid.bounds
        x_extent  = b[1] - b[0]
        y_base     = b[2]          # front edge
        z_base     = b[4]          # ground level (min elevation after z_ex)

        # Choose a clean round length: 1, 2, 5, 10, 20, 50 km steps
        raw_len = x_extent * 0.10
        for step in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
            if raw_len <= step:
                bar_len_m = step
                break
        else:
            bar_len_m = round(raw_len / 10000) * 10000

        x0 = b[0] + x_extent * 0.04
        x1 = x0 + bar_len_m
        y0 = y_base + (b[3]-b[2]) * 0.02
        z0 = z_base + (b[5]-b[4]) * 0.01     # slightly above ground

        bar_height = (b[5] - b[4]) * 0.004
        bar_width  = (b[3] - b[2]) * 0.005

        # Build bar as a thin box
        bar = pv.Box(bounds=(x0, x1, y0, y0 + bar_width, z0, z0 + bar_height))
        self.pl.add_mesh(bar, color="white", lighting=False)

        # Label above centre of bar
        cx = (x0 + x1) / 2
        label_z = z0 + bar_height + (b[5]-b[4]) * 0.02

        if bar_len_m >= 1000:
            label = f"{bar_len_m/1000:.0f} km"
        else:
            label = f"{bar_len_m:.0f} m"

        pt = pv.PolyData(np.array([[cx, y0, label_z]]))
        pt["labels"] = [label]
        self.pl.add_point_labels(
            pt, "labels",
            font_size=13,
            text_color="yellow",
            bold=True,
            shadow=True,
            show_points=False,
            always_visible=True,
        )
        pipeline_logger.info(f"Scale bar: {label} ({x0:.0f}m to {x1:.0f}m)")

    # ── Camera ────────────────────────────────────────────────────────────────

    def _reset_camera(self) -> None:
        b = self.grid.bounds
        cx     = (b[0] + b[1]) / 2
        cy     = (b[2] + b[3]) / 2
        zmax   = b[5]
        zdelta = b[5] - b[4]

        pos   = (cx, b[2] - (b[3]-b[2]) * 0.15, zmax + zdelta * 0.6)
        focal = (cx, cy, zmax * 0.4)
        self.pl.camera_position = [pos, focal, (0.0, 0.0, 1.0)]
        self.pl.camera.clipping_range = (1.0, 200000.0)

    # ── Key bindings ─────────────────────────────────────────────────────────

    def _bind_keys(self) -> None:
        s = self.speed

        def _translate(dx, dy, dz):
            cam   = self.pl.camera
            fwd   = np.array(cam.direction, dtype=np.float64)
            up    = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up)
            nr    = np.linalg.norm(right)
            if nr > 0:
                right /= nr

            pos   = np.array(cam.position,    dtype=np.float64)
            foc   = np.array(cam.focal_point, dtype=np.float64)
            delta = fwd * dy + right * dx + up * dz
            cam.position    = tuple(pos + delta)
            cam.focal_point = tuple(foc + delta)
            self.pl.render()

        self.pl.add_key_event("w", lambda: _translate(0,  s, 0))
        self.pl.add_key_event("s", lambda: _translate(0, -s, 0))
        self.pl.add_key_event("a", lambda: _translate(-s, 0, 0))
        self.pl.add_key_event("d", lambda: _translate( s, 0, 0))
        self.pl.add_key_event("q", lambda: _translate(0, 0,  s))
        self.pl.add_key_event("e", lambda: _translate(0, 0, -s))
        self.pl.add_key_event("r", lambda: (self._reset_camera(), self.pl.render()))

        self.pl.enable_terrain_style()

    # ── Launch ────────────────────────────────────────────────────────────────

    def start_flythrough(self) -> None:
        pipeline_logger.info("Launching fly-through window...")
        self.pl.show(interactive=True, full_screen=False)
