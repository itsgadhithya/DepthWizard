"""3D surface mesh generator and binary glTF 2.0 (GLB) exporter for Digital Surface Models."""

import json
import math
import struct
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from backend.models.mesh import Mesh3D, MeshMetadata
from backend.models.dsm import DSMResult
from backend.models.geometry import PointCloud3D, CoordinateFrame


class DSMMeshGenerator:
    """Generates continuous 3D triangular surface meshes from metric DSM elevation grids and point clouds."""

    @classmethod
    def generate_mesh_from_dsm(
        cls,
        dsm: DSMResult,
        image_rgb: Optional[np.ndarray] = None,
        max_grid_size: int = 512,
        max_elevation_step_m: float = 200.0,
    ) -> Mesh3D:
        """Create a 3D triangle surface mesh directly from a DSM elevation grid.

        Args:
            dsm: DSMResult with 2D float32 elevation grid and spatial bounds.
            image_rgb: Optional (H, W, 3) uint8 image for photographic texture reference.
            max_grid_size: Maximum resolution along longest dimension for rendering performance.
            max_elevation_step_m: Maximum elevation discontinuity across a single triangle face.

        Returns:
            Mesh3D containing vertices, triangular faces, normals, colors, and spatial metrics.
        """
        grid = dsm.grid
        h_orig, w_orig = grid.shape
        nodata = dsm.nodata_value

        # Calculate stride to keep vertex count responsive and smooth in real-time WebGL
        step = max(1, max(h_orig, w_orig) // max_grid_size)
        sub_grid = grid[::step, ::step]
        h_sub, w_sub = sub_grid.shape

        # Grid coordinate generation
        # Determine 2D (X, Y) positions in meters
        if dsm.is_local or not dsm.transform:
            # Local metric camera frame: Center-centered coordinates in meters
            res = dsm.resolution_m * step
            x_coords = (np.arange(w_sub) - w_sub / 2.0) * res
            y_coords = (h_sub / 2.0 - np.arange(h_sub)) * res
            xx, yy = np.meshgrid(x_coords, y_coords)
        else:
            # Georeferenced projected CRS: Apply affine transformation
            t = dsm.transform  # [c, a, b, f, d, e]
            cols = np.arange(w_sub) * step
            rows = np.arange(h_sub) * step
            col_mesh, row_mesh = np.meshgrid(cols, rows)
            xx = t[0] + t[1] * col_mesh + t[2] * row_mesh
            yy = t[3] + t[4] * col_mesh + t[5] * row_mesh

        zz = sub_grid.astype(np.float32)

        # Validity mask (exclude NoData cells and non-finite values)
        valid_mask = (zz > -9000.0) & (zz != nodata) & np.isfinite(zz)

        if not np.any(valid_mask):
            # Fallback: empty mesh
            empty_verts = np.zeros((0, 3), dtype=np.float32)
            empty_faces = np.zeros((0, 3), dtype=np.uint32)
            mesh = Mesh3D(
                vertices=empty_verts,
                faces=empty_faces,
                is_local=dsm.is_local,
                dsm_type=dsm.dsm_type,
                crs=dsm.crs,
            )
            mesh.compute_bounds_and_stats()
            return mesh

        # Create mapping from 2D (row, col) to compact 1D vertex index
        vertex_index_map = np.full((h_sub, w_sub), -1, dtype=np.int32)
        valid_indices = np.argwhere(valid_mask)
        n_vertices = len(valid_indices)

        for new_idx, (r, c) in enumerate(valid_indices):
            vertex_index_map[r, c] = new_idx

        # Extract vertex coordinates [X, Y, Z] in meters
        v_x = xx[valid_mask]
        v_y = yy[valid_mask]
        v_z = zz[valid_mask]
        vertices = np.stack([v_x, v_y, v_z], axis=-1).astype(np.float32)

        # UV coordinates [0.0 to 1.0]
        uv_u = valid_indices[:, 1] / max(1, w_sub - 1)
        uv_v = valid_indices[:, 0] / max(1, h_sub - 1)
        uvs = np.stack([uv_u, uv_v], axis=-1).astype(np.float32)

        # Continuous Scientific Terrain Elevation Colors (Low: Green -> Mid: Yellow/Ochre -> High: Brown -> Summit: White)
        z_min = float(np.min(v_z))
        z_max = float(np.max(v_z))
        z_range = max(1e-4, z_max - z_min)
        z_norm = np.clip((v_z - z_min) / z_range, 0.0, 1.0)
        elevation_colors = cls._elevation_to_rgb(z_norm)

        # Extract real photographic RGB aerial vertex colors if image_rgb provided
        rgb_colors = None
        if image_rgb is not None and len(image_rgb.shape) == 3:
            h_img, w_img = image_rgb.shape[:2]
            r_img = np.clip((valid_indices[:, 0] * step * h_img) // max(1, h_orig), 0, h_img - 1)
            c_img = np.clip((valid_indices[:, 1] * step * w_img) // max(1, w_orig), 0, w_img - 1)
            rgb_colors = image_rgb[r_img, c_img].astype(np.uint8)

        # By default, use RGB aerial colors if available for photorealistic landscape, else elevation colors
        active_colors = rgb_colors if rgb_colors is not None else elevation_colors

        # Build triangular faces (connecting neighbouring valid grid cells)
        faces_list = []
        for r in range(h_sub - 1):
            for c in range(w_sub - 1):
                i00 = vertex_index_map[r, c]
                i10 = vertex_index_map[r, c + 1]
                i01 = vertex_index_map[r + 1, c]
                i11 = vertex_index_map[r + 1, c + 1]

                # First triangle: (i00, i01, i10)
                if i00 >= 0 and i01 >= 0 and i10 >= 0:
                    z00, z01, z10 = zz[r, c], zz[r + 1, c], zz[r, c + 1]
                    if max(abs(z00 - z01), abs(z00 - z10), abs(z01 - z10)) <= max_elevation_step_m:
                        faces_list.append((i00, i01, i10))

                # Second triangle: (i10, i01, i11)
                if i10 >= 0 and i01 >= 0 and i11 >= 0:
                    z10, z01, z11 = zz[r, c + 1], zz[r + 1, c], zz[r + 1, c + 1]
                    if max(abs(z10 - z01), abs(z10 - z11), abs(z01 - z11)) <= max_elevation_step_m:
                        faces_list.append((i10, i01, i11))

        if len(faces_list) > 0:
            faces = np.array(faces_list, dtype=np.uint32)
        else:
            faces = np.zeros((0, 3), dtype=np.uint32)

        # Compute surface normal vectors for high-definition directional terrain lighting
        normals = cls._compute_vertex_normals(vertices, faces)

        mesh = Mesh3D(
            vertices=vertices,
            faces=faces,
            normals=normals,
            colors=active_colors,
            rgb_colors=rgb_colors,
            elevation_colors=elevation_colors,
            uvs=uvs,
            is_local=dsm.is_local,
            dsm_type=dsm.dsm_type,
            crs=dsm.crs,
            units="meters",
        )
        mesh.compute_bounds_and_stats()
        return mesh

    @classmethod
    def _compute_vertex_normals(cls, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute area-weighted smooth surface normal vectors for all vertices."""
        n_verts = len(vertices)
        if n_verts == 0 or len(faces) == 0:
            return np.zeros((n_verts, 3), dtype=np.float32)

        normals = np.zeros((n_verts, 3), dtype=np.float32)

        # Vectors for face edges
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = np.cross(e1, e2)

        # Accumulate onto vertices
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)

        # Normalize to unit length
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        normals = (normals / lengths).astype(np.float32)
        return normals

    @classmethod
    def _elevation_to_rgb(cls, z_norm: np.ndarray) -> np.ndarray:
        """Map normalized elevation [0, 1] to natural continuous topographical terrain colors.

        Color Palette:
            0.00 -> Deep valley / forest green
            0.18 -> Meadow emerald green
            0.40 -> Warm golden ochre / yellow plateau
            0.65 -> Terracotta / mountain ridge brown
            0.85 -> Alpine slate rock
            1.00 -> Mountain summit / snow white
        """
        colors = np.zeros((len(z_norm), 3), dtype=np.uint8)

        stops = [
            (0.00, np.array([27, 67, 50])),    # Deep Forest Green (#1b4332)
            (0.18, np.array([64, 145, 108])),  # Meadow Emerald (#40916c)
            (0.40, np.array([212, 163, 115])), # Warm Golden Ochre (#d4a373)
            (0.65, np.array([156, 102, 68])),  # Mountain Ridge Brown (#9c6644)
            (0.85, np.array([120, 130, 140])), # Alpine Slate Rock (#78828c)
            (1.00, np.array([248, 250, 252])), # Summit Snow White (#f8fafc)
        ]

        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            mask = (z_norm >= t0) & (z_norm <= t1)
            if np.any(mask):
                factor = (z_norm[mask] - t0) / (t1 - t0)
                factor = factor[:, np.newaxis]
                interpolated = c0 * (1.0 - factor) + c1 * factor
                colors[mask] = np.clip(interpolated, 0, 255).astype(np.uint8)

        return colors


class GLBExporter:
    """Pure-Python binary glTF 2.0 (GLB) file generator with zero external C/Rust dependencies."""

    @classmethod
    def save_glb(
        cls,
        mesh: Mesh3D,
        file_path: str,
    ) -> str:
        """Export Mesh3D to a standard binary glTF 2.0 (.glb) container.

        Args:
            mesh: Mesh3D instance containing vertices, faces, normals, colors.
            file_path: Target destination path on filesystem.

        Returns:
            Absolute path to saved .glb file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not mesh.bounds_min:
            mesh.compute_bounds_and_stats()

        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        normals = mesh.normals.astype(np.float32) if mesh.normals is not None else np.zeros_like(vertices)
        
        # Colors: Convert to float32 [0.0, 1.0] RGBA or RGB
        if mesh.colors is not None:
            if mesh.colors.dtype == np.uint8:
                colors = (mesh.colors.astype(np.float32) / 255.0).astype(np.float32)
            else:
                colors = mesh.colors.astype(np.float32)
        else:
            colors = np.full((len(vertices), 3), 0.8, dtype=np.float32)

        # Ensure vertices and faces are C-contiguous
        v_bytes = vertices.tobytes()
        n_bytes = normals.tobytes()
        c_bytes = colors.tobytes()
        f_bytes = faces.tobytes()

        # Build binary buffer with 4-byte alignments
        bin_parts = []
        buffer_views = []
        offset = 0

        # BufferView 0: POSITION (FLOAT, VEC3)
        bin_parts.append(v_bytes)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(v_bytes),
            "target": 34962,  # ARRAY_BUFFER
        })
        offset += len(v_bytes)

        # BufferView 1: NORMAL (FLOAT, VEC3)
        bin_parts.append(n_bytes)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(n_bytes),
            "target": 34962,  # ARRAY_BUFFER
        })
        offset += len(n_bytes)

        # BufferView 2: COLOR_0 (FLOAT, VEC3)
        bin_parts.append(c_bytes)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(c_bytes),
            "target": 34962,  # ARRAY_BUFFER
        })
        offset += len(c_bytes)

        # BufferView 3: INDICES (UNSIGNED_INT, SCALAR)
        bin_parts.append(f_bytes)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(f_bytes),
            "target": 34963,  # ELEMENT_ARRAY_BUFFER
        })
        offset += len(f_bytes)

        # Accessors
        min_pos = mesh.bounds_min if mesh.bounds_min else [0.0, 0.0, 0.0]
        max_pos = mesh.bounds_max if mesh.bounds_max else [0.0, 0.0, 0.0]

        accessors = [
            # 0: POSITION
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": min_pos,
                "max": max_pos,
            },
            # 1: NORMAL
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(normals),
                "type": "VEC3",
            },
            # 2: COLOR_0
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(colors),
                "type": "VEC3",
            },
            # 3: INDICES
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(faces) * 3,
                "type": "SCALAR",
                "min": [0],
                "max": [max(0, len(vertices) - 1)],
            },
        ]

        # Materials
        materials = [
            {
                "name": "DSM_Terrain_Material",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.05,
                    "roughnessFactor": 0.85,
                },
                "doubleSided": True,
            }
        ]

        # Meshes & Primitives
        meshes = [
            {
                "name": "DigitalSurfaceModel",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "COLOR_0": 2,
                        },
                        "indices": 3,
                        "material": 0,
                        "mode": 4,  # TRIANGLES
                    }
                ],
            }
        ]

        # Nodes, Scene, Asset
        nodes = [{"mesh": 0, "name": "DSM_Root_Node"}]
        scenes = [{"nodes": [0], "name": "DSM_Scene"}]

        gltf_dict: Dict[str, Any] = {
            "asset": {
                "version": "2.0",
                "generator": "DepthWizard 3D Metric Geometry Engine",
                "extras": {
                    "units": "meters",
                    "dsm_type": mesh.dsm_type,
                    "is_local": mesh.is_local,
                    "crs": mesh.crs,
                    "width_m": mesh.width_m,
                    "length_m": mesh.length_m,
                    "height_range_m": mesh.height_range_m,
                }
            },
            "scene": 0,
            "scenes": scenes,
            "nodes": nodes,
            "meshes": meshes,
            "materials": materials,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": offset}],
        }

        # JSON chunk encoding (padded to 4-byte boundary with 0x20 spaces)
        json_str = json.dumps(gltf_dict, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")
        json_pad_len = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b" " * json_pad_len

        # BIN chunk encoding (padded to 4-byte boundary with 0x00)
        bin_data = b"".join(bin_parts)
        bin_pad_len = (4 - (len(bin_data) % 4)) % 4
        bin_data += b"\x00" * bin_pad_len

        # GLB Header (12 bytes)
        # magic (0x46546C67), version (2), total_length
        total_glb_len = 12 + (8 + len(json_bytes)) + (8 + len(bin_data))
        header = struct.pack("<4sII", b"glTF", 2, total_glb_len)

        # Chunk 0 Header: length, type=0x4E4F534A ("JSON")
        chunk0_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)

        # Chunk 1 Header: length, type=0x004E4942 ("BIN\0")
        chunk1_header = struct.pack("<II", len(bin_data), 0x004E4942)

        with open(path, "wb") as f:
            f.write(header)
            f.write(chunk0_header)
            f.write(json_bytes)
            f.write(chunk1_header)
            f.write(bin_data)

        return str(path.resolve())

    @classmethod
    def save_obj(cls, mesh: Mesh3D, file_path: str) -> str:
        """Export Mesh3D to a standard Wavefront OBJ file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        vertices = mesh.vertices
        faces = mesh.faces
        colors = mesh.colors

        with open(path, "w", encoding="utf-8") as f:
            f.write("# DepthWizard Metric Digital Surface Model (OBJ)\n")
            f.write(f"# Units: {mesh.units}\n")
            f.write(f"# DSM Type: {mesh.dsm_type}\n")

            has_colors = colors is not None and len(colors) == len(vertices)
            for i, v in enumerate(vertices):
                if has_colors:
                    c = colors[i] / 255.0 if colors[i].dtype == np.uint8 else colors[i]
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
                else:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            # 1-indexed faces in OBJ
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        return str(path.resolve())
