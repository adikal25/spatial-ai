"""Pydantic schemas used by the API and services.

These models define:
- Request/response contracts for the FastAPI endpoints
- Internal representations of bounding boxes, spatial metrics, and contradictions
"""

# src/models/schemas.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Normalized bounding box for a detected object (0-1 coordinates)."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: Optional[str] = None
    confidence: Optional[float] = None


class SpatialMetrics(BaseModel):
    """Depth, size, and (optionally) 3D geometry for a single object."""

    object_id: str
    depth_mean: float
    depth_std: float
    estimated_distance: Optional[float] = None
    estimated_size: Optional[Dict[str, float]] = None
    bounding_box: BoundingBox
    # New: 3D centroid from fVDB-based reconstruction
    centroid_3d: Optional[Dict[str, float]] = None


class Contradiction(BaseModel):
    """Detected contradiction between VLM answer and geometric evidence."""

    type: str
    claim: str
    evidence: str
    severity: float


class QuestionRequest(BaseModel):
    """Payload for the /ask endpoint."""

    image: str
    question: str
    use_fallback: bool = False


class QuestionResponse(BaseModel):
    """Main response returned by the /ask endpoint."""

    answer: str
    revised_answer: Optional[str] = None
    self_reflection: Optional[str] = None
    confidence: float
    proof_overlay: Optional[str] = None
    detected_objects: List[BoundingBox] = []
    spatial_metrics: List[SpatialMetrics] = []
    contradictions: List[Contradiction] = []
    latency_ms: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Response model for /health."""

    status: str
    version: str
    models_loaded: Dict[str, bool]
