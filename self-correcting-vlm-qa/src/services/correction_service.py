"""
Self-correction service that makes the VLM refine answers using geometric evidence.
Implements explicit self-reasoning loop.
"""
from typing import Dict, Any, List, Optional

from loguru import logger

from src.models.schemas import Contradiction


class CorrectionService:
    """Service for Claude-based self-correction using geometric evidence."""

    def __init__(self, vlm_service):
        """Initialize correction service with VLM service."""
        self.vlm_service = vlm_service

    async def correct(
        self,
        image_base64: str,
        original_answer: str,
        original_reasoning: str,
        contradictions: List[Contradiction],
        proof_overlay: str,
        question: str,
        geometry_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make the VLM self-correct its answer using geometric evidence.
        Uses explicit self-reasoning loop.

        Args:
            image_base64: Original image
            original_answer: Model's initial answer
            original_reasoning: Model's initial reasoning
            contradictions: List of detected contradictions
            proof_overlay: Proof image with depth visualization
            question: Original question
            geometry_summary: Optional textual summary of 3D voxel metrics

        Returns:
            Dictionary with revised answer, self-reflection, and confidence score
        """
        try:
            # Convert Contradiction objects to dicts
            contradiction_dicts = [
                {
                    "type": c.type,
                    "claim": c.claim,
                    "evidence": c.evidence,
                    "severity": c.severity
                }
                for c in contradictions
            ]

            # Use the self_correct_with_reasoning method from VLMService
            result = await self.vlm_service.self_correct_with_reasoning(
                image_base64=image_base64,
                original_question=question,
                original_answer=original_answer,
                original_reasoning=original_reasoning,
                contradictions=contradiction_dicts,
                proof_overlay_base64=proof_overlay,
                geometry_summary=geometry_summary,
            )

            logger.info(f"Self-correction complete, confidence: {result['confidence']}")

            return {
                "revised_answer": result["revised_answer"],
                "self_reflection": result.get("self_reflection", ""),
                "confidence": result["confidence"],
                "full_reasoning": result.get("full_reasoning", "")
            }

        except Exception as e:
            logger.error(f"Error during self-correction: {str(e)}")
            # Return original answer if correction fails
            return {
                "revised_answer": original_answer,
                "self_reflection": "Error during self-correction",
                "confidence": 0.5,
                "full_reasoning": ""
            }

    def _build_evidence_prompt(
        self,
        original_answer: str,
        contradictions: List[Contradiction]
    ) -> str:
        """
        Build evidence text from contradictions.

        Args:
            original_answer: Original answer
            contradictions: List of contradictions

        Returns:
            Formatted evidence text
        """
        evidence_lines = []

        for i, contradiction in enumerate(contradictions, 1):
            evidence_lines.append(
                f"{i}. **{contradiction.type.upper()} CONTRADICTION** (severity: {contradiction.severity:.1%})\n"
                f"   - Your claim: {contradiction.claim}\n"
                f"   - Geometric evidence: {contradiction.evidence}"
            )

        return "\n\n".join(evidence_lines)

    def _extract_confidence(self, text: str) -> float:
        """
        Extract confidence score from VLM response.

        Args:
            text: VLM response text

        Returns:
            Confidence score 0-1
        """
        import re

        # Look for patterns like "confidence: 0.8" or "80% confident"
        patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9]+)%\s+confident',
            r'confidence[:\s]+([0-9]+)%'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                value = float(match.group(1))
                # Convert percentage to decimal if needed
                if value > 1:
                    value = value / 100.0
                return min(1.0, max(0.0, value))

        # Default confidence based on whether answer changed significantly
        return 0.7
