"""
Data models and schemas for the Self-Correcting VLM QA API.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    label: Optional[str] = Field(None, description="Object label")
    confidence: Optional[float] = Field(None, description="Detection confidence")


class SpatialMetrics(BaseModel):
    """Geometric metrics computed from depth estimation."""
    object_id: str
    depth_mean: float = Field(..., description="Mean depth value")
    depth_std: float = Field(..., description="Depth standard deviation")
    estimated_distance: Optional[float] = Field(None, description="Estimated distance in meters")
    estimated_size: Optional[Dict[str, float]] = Field(None, description="Estimated size (width, height)")
    bounding_box: BoundingBox


class Contradiction(BaseModel):
    """Detected contradiction in VLM response."""
    type: str = Field(..., description="Type of contradiction (size/distance/count)")
    claim: str = Field(..., description="VLM's original claim")
    evidence: str = Field(..., description="Geometric evidence")
    severity: float = Field(..., description="Severity score 0-1")


class IterationHistory(BaseModel):
    """History of a single iteration in the self-correction loop."""
    iteration: int = Field(..., description="Iteration number (0-indexed)")
    answer: str = Field(..., description="Answer for this iteration")
    reasoning: Optional[str] = Field(None, description="Initial reasoning (iteration 0) or self-reflection (later iterations)")
    contradiction_count: int = Field(..., description="Number of contradictions detected")
    avg_severity: float = Field(..., description="Average severity of contradictions (0-1)")
    confidence: float = Field(..., description="Confidence score for this iteration")
    self_reflection: Optional[str] = Field(None, description="Self-reflection text (if corrected)")
    contradictions: List[Contradiction] = Field(default_factory=list)
    improved: bool = Field(..., description="Whether this iteration improved over previous")


class ConvergenceMetrics(BaseModel):
    """Convergence metrics for the iterative loop."""
    total_iterations: int = Field(..., description="Total iterations run")
    converged: bool = Field(..., description="Whether the loop converged successfully")
    convergence_reason: str = Field(..., description="Reason for convergence/stopping")
    final_contradiction_count: int = Field(..., description="Contradictions in final iteration")
    initial_contradiction_count: int = Field(..., description="Contradictions in first iteration")
    contradiction_reduction: float = Field(..., description="Percentage reduction in contradictions")


class QuestionRequest(BaseModel):
    """Request payload for /ask endpoint."""
    image: str = Field(..., description="Base64-encoded image data")
    question: str = Field(..., description="Spatial question about the image")
    use_fallback: bool = Field(False, description="Use fallback VLM if primary fails")


class QuestionResponse(BaseModel):
    """Response payload for /ask endpoint."""
    answer: str = Field(..., description="Initial VLM response")
    revised_answer: Optional[str] = Field(None, description="Final self-corrected response")
    self_reflection: Optional[str] = Field(None, description="Final self-reflection")
    confidence: float = Field(..., description="Final confidence score 0-1")
    proof_overlay: Optional[str] = Field(None, description="Base64-encoded proof image")
    detected_objects: List[BoundingBox] = Field(default_factory=list)
    spatial_metrics: List[SpatialMetrics] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list, description="Final iteration contradictions")
    iteration_history: List[IterationHistory] = Field(default_factory=list, description="History of all iterations")
    convergence_metrics: Optional[ConvergenceMetrics] = Field(None, description="Loop convergence metrics")
    latency_ms: Dict[str, float] = Field(default_factory=dict, description="Stage latencies")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
