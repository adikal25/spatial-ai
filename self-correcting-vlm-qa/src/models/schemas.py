"""
Data models and schemas for the Self-Correcting VLM QA API.
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    object_id: Optional[str] = Field(None, description="Stable identifier used in structured claims")
    label: Optional[str] = Field(None, description="Object label")
    confidence: Optional[float] = Field(None, description="Detection confidence")


class SpatialMetrics(BaseModel):
    """Geometric metrics computed from depth estimation."""
    object_id: str
    depth_mean: float = Field(..., description="Mean depth value (higher = further)")
    depth_std: float = Field(..., description="Depth standard deviation")
    estimated_distance: Optional[float] = Field(None, description="Relative distance metric (not absolute meters)")
    estimated_size: Optional[Dict[str, float]] = Field(None, description="Estimated size (normalized width, height)")
    bounding_box: BoundingBox


class Contradiction(BaseModel):
    """Detected contradiction in VLM response."""
    type: str = Field(..., description="Type of contradiction (size/distance/count)")
    claim: str = Field(..., description="VLM's original claim")
    evidence: str = Field(..., description="Geometric evidence")
    severity: float = Field(..., description="Severity score 0-1")


class ComparisonClaim(BaseModel):
    """Structured comparison claim between two objects."""
    subject_id: str = Field(..., description="ID of the first object")
    object_id: str = Field(..., description="ID of the second object")
    attribute: Literal["distance", "size"] = Field(..., description="Attribute being compared")
    relation: Literal["closer", "further", "similar", "larger", "smaller", "same_size"] = Field(
        ..., description="Claimed relation between the objects"
    )


class CountClaim(BaseModel):
    """Structured count claim for a given object type."""
    object_type: str = Field(..., description="Label/type being counted")
    count: int = Field(..., description="Claimed count")
    object_ids: Optional[List[str]] = Field(None, description="IDs included in the count (if enumerated)")


class QuestionRequest(BaseModel):
    """Request payload for /ask endpoint."""
    image: str = Field(..., description="Base64-encoded image data")
    question: str = Field(..., description="Spatial question about the image")
    use_fallback: bool = Field(False, description="Use fallback VLM if primary fails")


class QuestionResponse(BaseModel):
    """Response payload for /ask endpoint."""
    answer: str = Field(..., description="Initial VLM response")
    revised_answer: Optional[str] = Field(None, description="Self-corrected response (if contradictions found)")
    self_reflection: Optional[str] = Field(None, description="Claude's self-reflection on its errors")
    confidence: float = Field(..., description="Confidence score 0-1")
    proof_overlay: Optional[str] = Field(None, description="Base64-encoded proof image with depth visualization")
    detected_objects: List[BoundingBox] = Field(default_factory=list)
    spatial_metrics: List[SpatialMetrics] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list, description="Detected contradictions")
    latency_ms: Dict[str, float] = Field(default_factory=dict, description="Stage latencies")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
