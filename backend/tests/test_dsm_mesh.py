"""Unit tests for 3D Surface Mesh generation, NoData handling, and binary GLB / OBJ export."""

import struct
import json
import pytest
import numpy as np
from pathlib import Path

from backend.models.dsm import DSMResult
from backend.models.geospatial import GeoBounds
from backend.models.mesh import Mesh3D, MeshMetadata
from backend.dsm.mesh_generator import DSMMeshGenerator, GLBExporter


def test_dsm_mesh_generation_synthetic_grid():
    """Verify generating 3D surface mesh from a regular metric elevation grid."""
    w, h = 30, 30
    x = np.linspace(0.0, 60.0, w)  # 60m width
    y = np.linspace(0.0, 60.0, h)  # 60m length
    xx, yy = np.meshgrid(x, y)
    zz = 20.0 + 5.0 * np.sin(xx / 10.0) + 2.0 * np.cos(yy / 10.0)  # undulating terrain (Z in meters)

    transform = [0.0, 2.0, 0.0, 60.0, 0.0, -2.0]
    bounds = GeoBounds(min_x=0.0, min_y=0.0, max_x=60.0, max_y=60.0, crs="LOCAL")

    dsm = DSMResult(
        grid=zz.astype(np.float32),
        width=w,
        height=h,
        crs=None,
        dsm_type="local_metric",
        is_local=True,
        transform=transform,
        bounds=bounds,
        resolution_m=2.0,
        min_elevation_m=float(np.min(zz)),
        max_elevation_m=float(np.max(zz)),
        mean_elevation_m=float(np.mean(zz)),
        nodata_value=-9999.0,
        valid_pixel_count=w * h,
        valid_coverage_percent=100.0,
        units="meters",
    )

    mesh = DSMMeshGenerator.generate_mesh_from_dsm(dsm=dsm, max_grid_size=100)

    assert isinstance(mesh, Mesh3D)
    assert mesh.vertex_count == w * h
    assert mesh.triangle_count == (w - 1) * (h - 1) * 2
    assert mesh.vertices.shape == (w * h, 3)
    assert mesh.faces.shape == ((w - 1) * (h - 1) * 2, 3)
    assert mesh.normals.shape == (w * h, 3)
    assert mesh.colors.shape == (w * h, 3)
    assert mesh.units == "meters"

    # Verify physical dimensions match terrain in meters
    assert pytest.approx(mesh.width_m, rel=0.05) == 60.0
    assert pytest.approx(mesh.length_m, rel=0.05) == 60.0
    assert mesh.height_range_m > 0.0
    assert mesh.bounds_min[2] >= 12.0
    assert mesh.bounds_max[2] <= 28.0

    # Verify surface normal vectors are unit length
    normal_lengths = np.linalg.norm(mesh.normals, axis=1)
    np.testing.assert_allclose(normal_lengths, 1.0, atol=1e-3)


def test_dsm_mesh_nodata_boundary_handling():
    """Verify that NoData cells (-9999.0) are excluded and do not create invalid geometry."""
    w, h = 20, 20
    grid = np.full((h, w), 50.0, dtype=np.float32)

    # Cut a central void (e.g. unobserved area / water body)
    grid[5:15, 5:15] = -9999.0

    transform = [0.0, 1.0, 0.0, 20.0, 0.0, -1.0]
    bounds = GeoBounds(min_x=0.0, min_y=0.0, max_x=20.0, max_y=20.0, crs="LOCAL")

    dsm = DSMResult(
        grid=grid,
        width=w,
        height=h,
        crs=None,
        dsm_type="local_metric",
        is_local=True,
        transform=transform,
        bounds=bounds,
        resolution_m=1.0,
        min_elevation_m=50.0,
        max_elevation_m=50.0,
        mean_elevation_m=50.0,
        nodata_value=-9999.0,
        valid_pixel_count=w * h - 100,
        valid_coverage_percent=75.0,
        units="meters",
    )

    mesh = DSMMeshGenerator.generate_mesh_from_dsm(dsm=dsm)

    # Valid vertex count must equal valid cell count (400 - 100 = 300)
    assert mesh.vertex_count == 300
    assert np.all(mesh.vertices[:, 2] == 50.0)
    assert not np.any(mesh.vertices[:, 2] == -9999.0)

    # Triangles in the void region should not exist
    assert mesh.triangle_count < (w - 1) * (h - 1) * 2


def test_glb_exporter_binary_container(tmp_path):
    """Verify GLBExporter writes a valid, self-contained binary glTF 2.0 container."""
    # Synthetic tetrahedron mesh
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [5.0, 10.0, 0.0],
        [5.0, 5.0, 8.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ], dtype=np.uint32)
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
    ], dtype=np.uint8)

    mesh = Mesh3D(
        vertices=vertices,
        faces=faces,
        colors=colors,
        is_local=True,
        dsm_type="local_metric",
        units="meters",
    )
    mesh.compute_bounds_and_stats()

    out_glb = tmp_path / "test_model.glb"
    saved_path = GLBExporter.save_glb(mesh, str(out_glb))

    assert Path(saved_path).exists()
    file_bytes = Path(saved_path).read_bytes()

    # 1. Validate 12-byte GLB Header
    assert len(file_bytes) >= 12
    magic, version, total_len = struct.unpack("<4sII", file_bytes[:12])
    assert magic == b"glTF"
    assert version == 2
    assert total_len == len(file_bytes)

    # 2. Validate Chunk 0 (JSON)
    chunk0_len, chunk0_type = struct.unpack("<II", file_bytes[12:20])
    assert chunk0_type == 0x4E4F534A  # "JSON"
    json_bytes = file_bytes[20: 20 + chunk0_len]
    gltf_json = json.loads(json_bytes.decode("utf-8"))

    assert "asset" in gltf_json
    assert gltf_json["asset"]["version"] == "2.0"
    assert "meshes" in gltf_json
    assert "accessors" in gltf_json
    assert "bufferViews" in gltf_json

    # 3. Validate Chunk 1 (BIN)
    bin_start = 20 + chunk0_len
    chunk1_len, chunk1_type = struct.unpack("<II", file_bytes[bin_start: bin_start + 8])
    assert chunk1_type == 0x004E4942  # "BIN\0"
    assert bin_start + 8 + chunk1_len == total_len


def test_obj_exporter(tmp_path):
    """Verify OBJ exporter outputs readable Wavefront OBJ file."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    mesh = Mesh3D(vertices=vertices, faces=faces, units="meters")

    out_obj = tmp_path / "test_model.obj"
    saved = GLBExporter.save_obj(mesh, str(out_obj))

    assert Path(saved).exists()
    content = Path(saved).read_text(encoding="utf-8")
    assert "v 0.0000 0.0000 0.0000" in content
    assert "f 1 2 3" in content
