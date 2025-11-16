"""
Vision-Language Model service using Anthropic Claude models.
Handles spatial question answering plus self-correction.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from loguru import logger

from src.models.schemas import BoundingBox
from src.utils.image_utils import resize_image_if_needed


class VLMService:
    """Service for interacting with Anthropic's multimodal Claude models."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = Anthropic(api_key=api_key)
        self.model = os.getenv("CLAUDE_VLM_MODEL", "claude-sonnet-4-20250514")
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))
        self.max_output_tokens = int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "2048"))

    async def ask_with_boxes(
        self,
        image_base64: str,
        question: str,
        use_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Ask Claude a spatial question and request structured bounding boxes."""
        return await self._ask_claude(image_base64, question)

    async def _ask_claude(self, image_base64: str, question: str) -> Dict[str, Any]:
        try:
            logger.info("Processing image for Claude vision API...")
            processed_image = resize_image_if_needed(
                image_base64,
                max_size_mb=4.7,
                preserve_quality=True,
            )
            image_block = self._build_claude_image_content(processed_image)

            prompt = (
                "Analyze this image and answer the spatial question below. "
                "Respond strictly with JSON containing: answer (string), reasoning (string), "
                "and objects (array of {label, x1, y1, x2, y2, confidence}). "
                "Bounding box coordinates must be normalized between 0 and 1. "
                "Do not include any text outside the JSON response.\n\n"
                f"Question: {question}"
            )

            response = self.client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            image_block,
                        ],
                    }
                ],
            )

            response_text = self._extract_text_from_response(response)
            payload = self._parse_structured_response(response_text)

            answer = payload.get("answer", "").strip()
            reasoning = payload.get("reasoning", "").strip()
            objects = payload.get("objects", []) or []

            bounding_boxes: List[BoundingBox] = []
            for obj in objects:
                try:
                    bbox = BoundingBox(
                        x1=float(obj["x1"]),
                        y1=float(obj["y1"]),
                        x2=float(obj["x2"]),
                        y2=float(obj["y2"]),
                        label=obj.get("label"),
                        confidence=obj.get("confidence"),
                    )
                    bounding_boxes.append(bbox)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning(f"Skipping malformed bounding box: {exc}")

            logger.info(
                "Claude response processed: %d objects detected",
                len(bounding_boxes),
            )
            logger.debug("Reasoning: %s", reasoning)

            return {
                "answer": answer,
                "reasoning": reasoning,
                "bounding_boxes": bounding_boxes,
                "model": self.model,
            }

        except Exception as exc:
            logger.error(f"Error querying Claude: {exc}")
            raise

    async def self_correct_with_reasoning(
        self,
        image_base64: str,
        original_question: str,
        original_answer: str,
        original_reasoning: str,
        contradictions: List[Dict[str, Any]],
        proof_overlay_base64: str,
        geometry_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Self-correction with explicit reasoning loop using Claude."""
        try:
            processed_image = resize_image_if_needed(
                image_base64,
                max_size_mb=4.7,
                preserve_quality=True,
            )
            processed_proof = resize_image_if_needed(
                proof_overlay_base64,
                max_size_mb=4.7,
                preserve_quality=True,
            )

            evidence_text = "\n\n".join(
                [
                    (
                        f"**Contradiction {i + 1}: {c['type'].upper()}**\n"
                        f"- Your claim: {c['claim']}\n"
                        f"- Geometric evidence: {c['evidence']}\n"
                        f"- Severity: {c['severity']:.1%}"
                    )
                    for i, c in enumerate(contradictions)
                ]
            ) or "No contradictions provided."

            geometry_section = ""
            if geometry_summary:
                geometry_section = f"\n## 3D Geometry Summary\n{geometry_summary}\n"

            prompt = f"""# Self-Correction Task

You previously answered a spatial question about an image. Review your answer using geometric evidence.

## Original Question
{original_question}

## Your Original Answer
{original_answer}

## Your Original Reasoning
{original_reasoning}

## Geometric Contradictions Found
{evidence_text}
{geometry_section}

Follow this process:
1. Review the original image
2. Analyze the depth/verification overlay
3. Compare the measurements with your reasoning
4. Reflect on potential mistakes
5. Provide a corrected answer (or defend the original)

Respond with this format:
**Self-Reflection:**
...

**Revised Answer:**
...

**Confidence:**
[number between 0 and 1]
"""

            response = self.client.messages.create(
                model=self.model,
                temperature=max(self.temperature, 0.2),
                max_tokens=self.max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            self._build_claude_image_content(processed_image),
                            self._build_claude_image_content(processed_proof),
                        ],
                    }
                ],
            )

            full_response = self._extract_text_from_response(response)
            logger.info(
                "Self-correction response received (%d chars)",
                len(full_response),
            )
            logger.debug("Full response sample: %s", full_response[:500])

            revised_answer = full_response
            self_reflection = ""
            confidence = 0.7

            reflection_match = re.search(
                r"\*\*Self-Reflection:\*\*\s*(.+?)(?=\*\*Revised Answer:\*\*|\*\*Confidence:\*\*|$)",
                full_response,
                re.DOTALL,
            )
            answer_match = re.search(
                r"\*\*Revised Answer:\*\*\s*(.+?)(?=\*\*Confidence:\*\*|$)",
                full_response,
                re.DOTALL,
            )
            confidence_match = re.search(
                r"\*\*Confidence:\*\*\s*([0-9.]+)",
                full_response,
            )

            if not reflection_match:
                reflection_match = re.search(
                    r"Self-Reflection:\s*(.+?)(?=Revised Answer:|Confidence:|$)",
                    full_response,
                    re.DOTALL,
                )
            if not answer_match:
                answer_match = re.search(
                    r"Revised Answer:\s*(.+?)(?=Confidence:|$)",
                    full_response,
                    re.DOTALL,
                )
            if not confidence_match:
                confidence_match = re.search(
                    r"Confidence:\s*([0-9.]+)",
                    full_response,
                )

            if reflection_match:
                self_reflection = reflection_match.group(1).strip()
            if answer_match:
                revised_answer = answer_match.group(1).strip()
            if confidence_match:
                try:
                    value = float(confidence_match.group(1))
                    confidence = value if value <= 1 else value / 100.0
                except ValueError:
                    logger.debug("Could not parse confidence value, keeping default")

            return {
                "revised_answer": revised_answer,
                "self_reflection": self_reflection,
                "confidence": min(max(confidence, 0.0), 1.0),
                "full_reasoning": full_response,
            }

        except Exception as exc:
            logger.error(f"Error during self-correction: {exc}")
            raise

    @staticmethod
    def _build_claude_image_content(image_base64: str) -> Dict[str, Any]:
        media_type, data = VLMService._split_data_url(image_base64)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    @staticmethod
    def _split_data_url(image_base64: str) -> Tuple[str, str]:
        if image_base64.startswith("data:image") and "," in image_base64:
            header, data = image_base64.split(",", 1)
            media_type = header.split(";")[0].split(":", 1)[1]
            return media_type, data
        return "image/jpeg", image_base64

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        blocks = getattr(response, "content", []) or []
        texts: List[str] = []
        for block in blocks:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "text":
                text_value = getattr(block, "text", None)
                if text_value is None and isinstance(block, dict):
                    text_value = block.get("text")
                if text_value:
                    texts.append(text_value)
        return "\n".join(texts).strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9_+-]*", "", stripped).strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()
        return stripped

    def _parse_structured_response(self, text: str) -> Dict[str, Any]:
        cleaned = self._strip_code_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.warning("Falling back to default payload parsing")
            return {
                "answer": cleaned.strip(),
                "reasoning": "",
                "objects": [],
            }
