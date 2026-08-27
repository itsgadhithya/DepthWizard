"""Unit tests for 3D back-projection and point cloud serialization."""

import pytest
import numpy as np
from pathlib import Path

from backend.models.depth import RelativeDepthMap
from backend.models.camera import CameraIntrinsics
from backend.models.geometry import PointCloud3D, CoordinateFrame, Units
from backend.geometry.backprojector import DepthBackProjector
from backend.geometry.point_cloud import PointCloudIO


def test_depth_backprojector_math():
    """Verify pinhole backprojection arithmetic matches theoretical geometry."""
    # 3x3 depth map with known depths
    z_arr = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float32)

    depth_map = RelativeDepthMap(
        array=z_arr,
        width=3,
        height=3,
        min_val=10.0,
        max_val=10.0,
        mean_val=10.0,
        std_val=0.0,
        model_name="test",
        model_config_name="test",
        device="cpu",
        inference_time_ms=1.0,
    )

    intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=1.0, cy=1.0, width=3, height=3)

    pc = DepthBackProjector.backproject(depth_map, intrinsics, compute_normals=False)

    assert isinstance(pc, PointCloud3D)
    assert pc.point_count == 9

    # Check center pixel (1, 1): u=1, v=1 -> X = (1-1)*10/10 = 0, Y = (1-1)*10/10 = 0, Z = 10
    center_pt = pc.points[4]
    assert pytest.approx(center_pt[0]) == 0.0
    assert pytest.approx(center_pt[1]) == 0.0
    assert pytest.approx(center_pt[2]) == 10.0

    # Check top-left pixel (0, 0): u=0, v=0 -> X = (0-1)*10/10 = -1.0, Y = (0-1)*10/10 = -1.0, Z = 10
    tl_pt = pc.points[0]
    assert pytest.approx(tl_pt[0]) == -1.0
    assert pytest.approx(tl_pt[1]) == -1.0
    assert pytest.approx(tl_pt[2]) == 10.0


def test_point_cloud_ply_export(tmp_path):
    """Verify PLY file export creates valid Stanford PLY header and data."""
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    pc = PointCloud3D(points=pts, colors=colors)
    pc.compute_bounds()

    out_file = tmp_path / "test.ply"
    saved_path = PointCloudIO.save_ply(pc, str(out_file), binary=True)

    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 0

    with open(saved_path, "rb") as f:
        header = f.read(400).decode("ascii", errors="ignore")
        assert "ply" in header
        assert "element vertex 2" in header
        assert "end_header" in header


def test_point_cloud_web_json():
    """Verify WebGL JSON export."""
    pts = np.random.rand(100, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
    pc = PointCloud3D(points=pts, colors=colors)
    pc.compute_bounds()

    web_dict = PointCloudIO.to_web_json(pc, max_points=50)
    assert web_dict["point_count"] <= 50
    assert len(web_dict["positions"]) == web_dict["point_count"] * 3
