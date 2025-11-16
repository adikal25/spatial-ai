"""
Service responsible for running TripoSG and exposing lightweight mesh stats.
"""
from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh
from PIL import Image
from loguru import logger

from src.models.schemas import BoundingBox
from src.tripo_reconstruct import TripoReconstructor


@dataclass
class Object3DStats:
    """Simple geometric summary for a subset of mesh vertices."""

    centroid: List[float]
    extent: List[float]
    point_count: int


@dataclass
class ReconstructionResult:
    """Holds reconstruction artifacts for downstream consumers."""

    mesh: Optional[trimesh.Trimesh]
    mesh_path: Optional[str]
    preview_base64: Optional[str]
    metadata: Dict[str, Any]


def compute_mesh_object_stats(
    mesh: Optional[trimesh.Trimesh],
    bbox: BoundingBox,
    min_points: int = 64
) -> Optional[Object3DStats]:
    """
    Approximate a 3D subset for a 2D bounding box by slicing the mesh bounds.
    """
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return None

    vertices = mesh.vertices
    bounds = mesh.bounds  # (min, max)
    if bounds is None or len(bounds) != 2:
        return None

    min_bound = bounds[0]
    max_bound = bounds[1]
    span = np.maximum(max_bound - min_bound, 1e-6)

    x_min = min_bound[0] + span[0] * float(bbox.x1)
    x_max = min_bound[0] + span[0] * float(bbox.x2)
    y_min = min_bound[1] + span[1] * float(bbox.y1)
    y_max = min_bound[1] + span[1] * float(bbox.y2)

    mask = np.where(
        (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
        (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)
    )[0]

    if mask.size < min_points:
        return None

    subset = vertices[mask]
    centroid = subset.mean(axis=0).tolist()
    extent = (subset.max(axis=0) - subset.min(axis=0)).tolist()

    return Object3DStats(
        centroid=centroid,
        extent=extent,
        point_count=int(mask.size)
    )


class ReconstructionService:
    """Async-friendly wrapper that orchestrates TripoSG inference."""

    def __init__(
        self,
        device: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        self.enabled = (
            enabled if enabled is not None
            else os.getenv("ENABLE_TRIPO_RECONSTRUCTION", "false").lower() == "true"
        )
        self.device = device or os.getenv("TRIPOSG_DEVICE") or "cuda"

        self._reconstructor: Optional[TripoReconstructor] = None
        self._latest_result: Optional[ReconstructionResult] = None

    async def load(self) -> None:
        """Lazily load TripoSG weights."""
        if not self.enabled:
            logger.info("TripoSG reconstruction disabled via configuration")
            return

        if self._reconstructor is None:
            self._reconstructor = TripoReconstructor(device=self.device)
            logger.info("TripoSG model loaded")

    async def reconstruct(self, image_base64: str) -> Optional[ReconstructionResult]:
        """Run TripoSG and cache the latest mesh + preview."""
        if not self.enabled:
            return None

        await self.load()
        if self._reconstructor is None:
            raise RuntimeError("Reconstruction requested before model was ready")

        image = self._decode_image(image_base64)
        mesh: trimesh.Trimesh = await asyncio.to_thread(
            self._reconstructor.reconstruct_mesh,
            image
        )

        preview = await asyncio.to_thread(
            self._reconstructor.render_preview,
            mesh
        )

        metadata = self._summarize_mesh(mesh)
        tripo_metadata = mesh.metadata.get("tripo", {}) if mesh.metadata else {}
        metadata["tripo"] = tripo_metadata

        mesh_path = tripo_metadata.get("mesh_path")
        result = ReconstructionResult(
            mesh=mesh,
            mesh_path=mesh_path,
            preview_base64=preview,
            metadata=metadata
        )

        self._latest_result = result
        return result

    def latest(self) -> Optional[ReconstructionResult]:
        """Return cached reconstruction result, if any."""
        return self._latest_result

    def is_ready(self) -> bool:
        """Return True if the underlying TripoSG weights are loaded."""
        return self.enabled and self._reconstructor is not None

    @staticmethod
    def _decode_image(image_base64: str) -> Image.Image:
        """Decode base64 image payloads or data URLs."""
        if image_base64.startswith("data:image"):
            _, encoded = image_base64.split(",", maxsplit=1)
        else:
            encoded = image_base64

        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")

    @staticmethod
    def _summarize_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Collect lightweight stats for inclusion in API responses."""
        vertices = len(mesh.vertices) if mesh.vertices is not None else 0
        faces = len(mesh.faces) if mesh.faces is not None else 0
        bounds = mesh.bounds.tolist() if mesh.bounds is not None else None

        return {
            "vertices": vertices,
            "faces": faces,
            "bounds": bounds
        }
