# tests/test_fvdb_3d_service.py
import os

import numpy as np
import pytest

fvdb = pytest.importorskip("fvdb")

from src.models.schemas import BoundingBox
from src.services.fvdb_3d_service import Fvdb3DReconstructionService


def test_fvdb_3d_compute_object_geometry_smoke():
    # Ensure 3D is enabled for this test
    os.environ["ENABLE_FVDB_3D"] = "true"

    service = Fvdb3DReconstructionService()
    assert service.enabled

    # Synthetic depth: simple plane
    depth_map = np.ones((4, 4), dtype=np.float32)

    # Single bounding box covering entire image
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, label="object")

    geom = service.compute_object_geometry(depth_map, [bbox])

    assert "centroids_3d" in geom
    assert len(geom["centroids_3d"]) == 1
    centroid = geom["centroids_3d"][0]
    assert centroid is not None
    assert len(centroid) == 3

    scene = geom["scene"]
    assert scene is not None
    assert scene.voxel_count > 0
