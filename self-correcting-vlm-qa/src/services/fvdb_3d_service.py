# src/services/fvdb_3d_service.py
"""
3D reconstruction and object geometry utilities using NVIDIA fVDB.

This service:
- Backprojects a depth map into 3D points with a simple pinhole camera model.
- Voxelizes those points into an fVDB GridBatch.
- Computes approximate 3D centroids for detected objects given their 2D bounding boxes.
- Computes pairwise distances between object centroids in 3D.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from src.models.schemas import BoundingBox

try:
    import fvdb  # type: ignore
except ImportError:  # pragma: no cover - only on machines without fVDB
    fvdb = None


@dataclass
class Scene3D:
    """Light-weight container for an fVDB 3D scene."""

    grid: Any  # fvdb.GridBatch
    ijk: torch.Tensor  # (N, 3) voxel indices
    origin: torch.Tensor  # (3,) world-space origin
    voxel_size: float
    image_size: Tuple[int, int]
    points: Optional[torch.Tensor] = None  # (N, 3) backprojected points (cpu)
    pixel_coords: Optional[torch.Tensor] = None  # (N, 2) [u, v] pixels for each point

    @property
    def voxel_count(self) -> int:
        return int(self.ijk.shape[0])


class Fvdb3DReconstructionService:
    """
    Service for building a sparse 3D voxel representation of a scene using fVDB.

    It is intentionally independent of the VLM pipeline and only depends on:
    - a depth map (H x W)
    - optional bounding boxes (normalized 0-1)

    Enable/disable via env vars:

    ENABLE_FVDB_3D=true|false         # default: false
    FVDB_VOXEL_SIZE=0.05              # meters (or arbitrary units)
    FVDB_DEVICE=cuda|cpu|auto         # default: auto
    """

    def __init__(self) -> None:
        enable_env = os.getenv("ENABLE_FVDB_3D", "false").lower() == "true"
        self.voxel_size: float = float(os.getenv("FVDB_VOXEL_SIZE", "0.05"))
        device_pref = os.getenv("FVDB_DEVICE", "auto").lower()

        if device_pref == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif device_pref == "cpu":
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.enabled: bool = fvdb is not None and enable_env
        if not fvdb:
            logger.warning("fvdb package not available; 3D reconstruction disabled.")
        elif not enable_env:
            logger.info("ENABLE_FVDB_3D is false; 3D reconstruction disabled.")
        else:
            logger.info(
                f"fVDB 3D reconstruction enabled "
                f"(voxel_size={self.voxel_size}, device={self.device})"
            )

    @staticmethod
    def _get_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Return simple synthetic pinhole intrinsics.

        This is a heuristic; replace with calibrated intrinsics if available.
        """
        fx = fy = float(max(width, height))
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    def build_scene(self, depth_map: np.ndarray) -> Optional[Scene3D]:
        """
        Build an fVDB GridBatch from a depth map.

        Args:
            depth_map: (H, W) depth array. Values <=0 or non-finite are ignored.

        Returns:
            Scene3D or None if fVDB is disabled or depth is invalid.
        """
        if not self.enabled:
            return None

        if depth_map.ndim != 2:
            raise ValueError("depth_map must be 2D (H, W)")

        h, w = depth_map.shape
        device = self.device

        depth = torch.from_numpy(depth_map).to(device=device, dtype=torch.float32)

        # Pixel coordinates (v, u)
        v_coords, u_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        valid_mask = torch.isfinite(depth) & (depth > 0)
        if not bool(valid_mask.any()):
            logger.warning("No valid depth values; skipping fVDB reconstruction.")
            return None

        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        z_valid = depth[valid_mask]

        fx, fy, cx, cy = self._get_intrinsics(w, h)
        fx_t = torch.tensor(fx, device=device, dtype=torch.float32)
        fy_t = torch.tensor(fy, device=device, dtype=torch.float32)
        cx_t = torch.tensor(cx, device=device, dtype=torch.float32)
        cy_t = torch.tensor(cy, device=device, dtype=torch.float32)

        # Back-project to 3D
        x = (u_valid - cx_t) * z_valid / fx_t
        y = (v_valid - cy_t) * z_valid / fy_t
        points = torch.stack([x, y, z_valid], dim=-1)  # (N, 3)
        pixel_coords = torch.stack([u_valid, v_valid], dim=-1)  # (N, 2)

        # Define origin and voxel indices
        origin = points.min(dim=0).values
        voxel_size = float(self.voxel_size)
        coords = (points - origin) / voxel_size
        ijk = torch.floor(coords).to(torch.int64)  # (N, 3)

        # Build fVDB GridBatch
        # NOTE: API may vary slightly by fvdb version; adjust if needed.
        jt = fvdb.JaggedTensor([ijk])
        voxel_sizes = torch.tensor(
            [[voxel_size, voxel_size, voxel_size]],
            device=device,
            dtype=torch.float32,
        )
        origins = origin.unsqueeze(0)

        grid = fvdb.GridBatch.from_ijk(
            ijk=jt,
            voxel_sizes=voxel_sizes,
            origins=origins,
        )

        logger.info(
            f"Built fVDB GridBatch: "
            f"{ijk.shape[0]} voxels, image_size=({h}, {w}), voxel_size={voxel_size}"
        )

        return Scene3D(
            grid=grid,
            ijk=ijk,
            origin=origin,
            voxel_size=voxel_size,
            image_size=(h, w),
            points=points.detach().to("cpu"),
            pixel_coords=pixel_coords.detach().to("cpu"),
        )

    def compute_object_geometry(
        self,
        depth_map: np.ndarray,
        bounding_boxes: Sequence[BoundingBox],
    ) -> Dict[str, Any]:
        """
        Compute 3D centroids for objects and pairwise distances.

        Args:
            depth_map: (H, W) depth array.
            bounding_boxes: sequence of BoundingBox models (normalized coords).

        Returns:
            Dict with:
              - 'scene': Scene3D or None
              - 'centroids_3d': List[Optional[np.ndarray]] of shape (3,)
              - 'pairwise_distances': Dict[(i, j), float] for i<j
        """
        if not self.enabled:
            return {
                "scene": None,
                "centroids_3d": [None] * len(bounding_boxes),
                "pairwise_distances": {},
            }

        scene = self.build_scene(depth_map)
        h, w = depth_map.shape
        fx, fy, cx, cy = self._get_intrinsics(w, h)

        centroids: List[Optional[np.ndarray]] = []
        for bbox in bounding_boxes:
            c = self._bbox_to_centroid(
                depth_map,
                bbox,
                fx,
                fy,
                cx,
                cy,
                scene=scene,
            )
            centroids.append(c)

        pairwise: Dict[Tuple[int, int], float] = {}
        for i in range(len(centroids)):
            ci = centroids[i]
            if ci is None:
                continue
            for j in range(i + 1, len(centroids)):
                cj = centroids[j]
                if cj is None:
                    continue
                d = float(np.linalg.norm(ci - cj))
                pairwise[(i, j)] = d

        return {
            "scene": scene,
            "centroids_3d": centroids,
            "pairwise_distances": pairwise,
        }

    @staticmethod
    def _bbox_to_centroid(
        depth_map: np.ndarray,
        bbox: BoundingBox,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        scene: Optional[Scene3D] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute approximate 3D centroid for a bounding box region.

        Uses mean (u, v, z) in the box and back-projects to 3D.
        """
        h, w = depth_map.shape

        # Convert normalized [0,1] coords to pixels
        x1_px = int(max(0, min(bbox.x1 * w, w - 1)))
        y1_px = int(max(0, min(bbox.y1 * h, h - 1)))
        x2_px = int(max(0, min(bbox.x2 * w, w)))
        y2_px = int(max(0, min(bbox.y2 * h, h)))

        if x2_px <= x1_px or y2_px <= y1_px:
            return None

        if (
            scene is not None
            and scene.points is not None
            and scene.pixel_coords is not None
        ):
            uv = scene.pixel_coords
            pts = scene.points
            mask = (
                (uv[:, 0] >= float(x1_px))
                & (uv[:, 0] < float(x2_px))
                & (uv[:, 1] >= float(y1_px))
                & (uv[:, 1] < float(y2_px))
            )
            if mask.any():
                pts_in_box = pts[mask]
                if pts_in_box.shape[0] > 0:
                    centroid = pts_in_box.mean(dim=0)
                    return centroid.detach().cpu().numpy()

        region = depth_map[y1_px:y2_px, x1_px:x2_px]
        if region.size == 0:
            return None

        # Use only positive, finite depth values
        mask = np.isfinite(region) & (region > 0)
        if not mask.any():
            return None

        zs = region[mask]

        # Pixel coordinates within region
        ys, xs = np.where(mask)
        ys = ys + y1_px
        xs = xs + x1_px

        u_mean = float(xs.mean())
        v_mean = float(ys.mean())
        z_mean = float(zs.mean())

        # Back-project to 3D
        X = (u_mean - cx) * z_mean / fx
        Y = (v_mean - cy) * z_mean / fy
        Z = z_mean

        return np.array([X, Y, Z], dtype=np.float32)
