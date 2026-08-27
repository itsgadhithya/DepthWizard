"""Data models for 3D surface meshes, triangles, and GLB/glTF export."""

from typing import List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MeshMetadata(BaseModel):
    """Metadata describing a generated 3D surface mesh."""
    format: str = "GLB"
    units: str = "meters"
    dsm_type: str = "local_metric"  # "local_metric" or "georeferenced_metric"
    is_local: bool = True
    crs: Optional[str] = None
    width_m: float = 0.0
    length_m: float = 0.0
    height_min_m: float = 0.0
    height_max_m: float = 0.0
    height_range_m: float = 0.0
    vertex_count: int = 0
    triangle_count: int = 0
    bounds_min: List[float] = Field(default_factory=list)
    bounds_max: List[float] = Field(default_factory=list)
    center: List[float] = Field(default_factory=list)


class Mesh3D(BaseModel):
    """Dense 3D triangular surface mesh representation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vertices: np.ndarray = Field(..., description="(N, 3) float32 coordinates in meters [X, Y, Z]")
    faces: np.ndarray = Field(..., description="(M, 3) uint32 triangle indices referencing vertices")
    normals: Optional[np.ndarray] = Field(default=None, description="Optional (N, 3) float32 surface normal vectors")
    colors: Optional[np.ndarray] = Field(default=None, description="Optional (N, 3) uint8 or float32 RGB colors")
    uvs: Optional[np.ndarray] = Field(default=None, description="Optional (N, 2) float32 texture coordinates [U, V]")
    
    is_local: bool = True
    dsm_type: str = "local_metric"
    crs: Optional[str] = None
    units: str = "meters"

    # Computed metrics
    vertex_count: int = 0
    triangle_count: int = 0
    bounds_min: List[float] = Field(default_factory=list)
    bounds_max: List[float] = Field(default_factory=list)
    center: List[float] = Field(default_factory=list)
    width_m: float = 0.0
    length_m: float = 0.0
    height_range_m: float = 0.0

    def compute_bounds_and_stats(self) -> None:
        """Calculate spatial extents, center, and bounding metrics in meters."""
        self.vertex_count = len(self.vertices) if self.vertices is not None else 0
        self.triangle_count = len(self.faces) if self.faces is not None else 0

        if self.vertex_count > 0:
            min_xyz = np.min(self.vertices, axis=0).astype(float)
            max_xyz = np.max(self.vertices, axis=0).astype(float)
            self.bounds_min = min_xyz.tolist()
            self.bounds_max = max_xyz.tolist()
            self.center = ((min_xyz + max_xyz) / 2.0).tolist()
            self.width_m = float(max_xyz[0] - min_xyz[0])
            self.length_m = float(max_xyz[1] - min_xyz[1])
            self.height_range_m = float(max_xyz[2] - min_xyz[2])
        else:
            self.bounds_min = [0.0, 0.0, 0.0]
            self.bounds_max = [0.0, 0.0, 0.0]
            self.center = [0.0, 0.0, 0.0]
            self.width_m = 0.0
            self.length_m = 0.0
            self.height_range_m = 0.0

    def get_metadata(self) -> MeshMetadata:
        """Get summary metadata object."""
        if not self.bounds_min:
            self.compute_bounds_and_stats()

        return MeshMetadata(
            format="GLB",
            units=self.units,
            dsm_type=self.dsm_type,
            is_local=self.is_local,
            crs=self.crs,
            width_m=round(self.width_m, 2),
            length_m=round(self.length_m, 2),
            height_min_m=round(self.bounds_min[2], 2) if self.bounds_min else 0.0,
            height_max_m=round(self.bounds_max[2], 2) if self.bounds_max else 0.0,
            height_range_m=round(self.height_range_m, 2),
            vertex_count=self.vertex_count,
            triangle_count=self.triangle_count,
            bounds_min=[round(v, 2) for v in self.bounds_min],
            bounds_max=[round(v, 2) for v in self.bounds_max],
            center=[round(v, 2) for v in self.center],
        )
