"""
Main FastAPI application for Self-Correcting Vision-Language QA.
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.models.schemas import QuestionRequest, QuestionResponse, HealthResponse
from src.services.vlm_service import VLMService
from src.services.depth_service import DepthService
from src.services.verifier_service import VerifierService
from src.services.correction_service import CorrectionService

# Load environment variables from .env (check multiple locations)
project_root = Path(__file__).parent.parent.parent
env_paths = [
    project_root / ".env",  # Project root
    project_root / "config" / ".env",  # Config directory
]

# Try loading from each location
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded .env from {env_path}")
        break
else:
    # Also try loading from environment variables directly (useful for deployment)
    load_dotenv()
    logger.info("Attempted to load from environment variables")


# Global service instances
services: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services."""
    logger.info("Initializing services...")

    # Initialize services
    services["vlm"] = VLMService()
    services["depth"] = DepthService()
    services["verifier"] = VerifierService(services["depth"])
    services["correction"] = CorrectionService(services["vlm"])

    # Load models
    await services["depth"].load_model()

    logger.info("Services initialized successfully")
    yield

    # Cleanup
    logger.info("Shutting down services...")
    services.clear()


# Create FastAPI app
app = FastAPI(
    title="Self-Correcting Vision-Language QA",
    description="Automated verification and self-correction pipeline for VLM spatial reasoning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "vlm": services.get("vlm") is not None,
            "depth": services.get("depth") is not None and services["depth"].model is not None,
        }
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for self-correcting VLM question answering.

    Three-stage pipeline:
    1. Ask: VLM generates initial response with bounding boxes (1-3s)
    2. Verify: Depth estimation + geometric contradiction detection (1-4s)
    3. Correct: VLM self-corrects using evidence (1-4s)
    """
    latency = {}

    try:
        # Stage 1: Ask VLM
        logger.info(f"Stage 1: Asking VLM - '{request.question}'")
        start_time = time.time()

        initial_response = await services["vlm"].ask_with_boxes(
            image_base64=request.image,
            question=request.question,
            use_fallback=request.use_fallback
        )

        latency["ask_ms"] = (time.time() - start_time) * 1000
        logger.info(f"Stage 1 completed in {latency['ask_ms']:.0f}ms")

        # Stage 2: Verify with depth geometry
        logger.info("Stage 2: Verifying with depth geometry")
        start_time = time.time()

        verification_result = await services["verifier"].verify(
            image_base64=request.image,
            vlm_response=initial_response,
            question=request.question
        )

        latency["verify_ms"] = (time.time() - start_time) * 1000
        logger.info(f"Stage 2 completed in {latency['verify_ms']:.0f}ms")

        # Stage 3: Self-correct if contradictions found
        revised_answer = None
        self_reflection = None
        confidence = 0.9

        if verification_result.contradictions:
            logger.info(f"Stage 3: Self-correcting ({len(verification_result.contradictions)} contradictions)")
            start_time = time.time()

            correction_result = await services["correction"].correct(
                image_base64=request.image,
                original_answer=initial_response["answer"],
                original_reasoning=initial_response.get("reasoning", ""),
                contradictions=verification_result.contradictions,
                proof_overlay=verification_result.proof_overlay,
                question=request.question
            )

            revised_answer = correction_result["revised_answer"]
            self_reflection = correction_result.get("self_reflection", "")
            confidence = correction_result["confidence"]

            latency["correct_ms"] = (time.time() - start_time) * 1000
            logger.info(f"Stage 3 completed in {latency['correct_ms']:.0f}ms")
        else:
            logger.info("No contradictions found, skipping Stage 3")
            latency["correct_ms"] = 0

        # Calculate total latency
        total_latency = sum(latency.values())
        latency["total_ms"] = total_latency

        logger.info(f"Request completed in {total_latency:.0f}ms")

        return QuestionResponse(
            answer=initial_response["answer"],
            revised_answer=revised_answer,
            self_reflection=self_reflection,
            confidence=confidence,
            proof_overlay=verification_result.proof_overlay,
            detected_objects=initial_response.get("bounding_boxes", []),
            spatial_metrics=verification_result.spatial_metrics,
            contradictions=verification_result.contradictions,
            latency_ms=latency,
            metadata={
                "model_used": initial_response.get("model", "unknown"),
                "contradictions_found": len(verification_result.contradictions),
                "original_reasoning": initial_response.get("reasoning", "")
            }
        )

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
