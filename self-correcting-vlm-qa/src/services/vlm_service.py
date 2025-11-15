"""
Vision-Language Model service using a Claude Sonnet vision model.
Handles spatial question answering plus self-correction.
"""
import json
import os
from typing import Any, Dict, List, Tuple

import anthropic
from loguru import logger

from src.models.schemas import BoundingBox
from src.utils.image_utils import resize_image_if_needed


class VLMService:
    """Service for interacting with multimodal VLM models via the Claude (Anthropic) client."""

    def __init__(self):
        # Prefer Anthropic-specific env vars but keep backward compatibility with previous OPENAI_* names
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY (or legacy OPENAI_API_KEY) not found in environment variables")

        self.client = anthropic.Anthropic(api_key=api_key)

        # Default to a Claude Sonnet vision-capable model
        self.model = os.getenv("CLAUDE_VLM_MODEL") or os.getenv("OPENAI_VLM_MODEL", "claude-3.5-sonnet")
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE") or os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.max_output_tokens = int(
            os.getenv("CLAUDE_MAX_OUTPUT_TOKENS") or os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2048")
        )

    async def ask_with_boxes(
        self,
        image_base64: str,
        question: str,
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """Ask the VLM a spatial question and request bounding boxes."""
        return await self._ask_claude(image_base64, question)

    async def _ask_claude(self, image_base64: str, question: str) -> Dict[str, Any]:
        try:
            logger.info("Processing image for VLM vision API (Claude)...")
            processed_image = resize_image_if_needed(image_base64, max_size_mb=4.7, preserve_quality=True)
            image_media_type, image_data = self._extract_media_type_and_data(processed_image)

            tools = [
                {
                    "name": "provide_spatial_answer",
                    "description": (
                        "Provide an answer to a spatial question with bounding boxes for detected objects."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "objects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "x1": {"type": "number"},
                                        "y1": {"type": "number"},
                                        "x2": {"type": "number"},
                                        "y2": {"type": "number"},
                                        "confidence": {"type": "number"},
                                    },
                                    "required": ["label", "x1", "y1", "x2", "y2"],
                                },
                            },
                        },
                        "required": ["answer", "reasoning", "objects"],
                    },
                }
            ]

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                tools=tools,
                system="You are a precise spatial reasoner. Always return bounding boxes via the provided tool.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this image and answer the following spatial question.\n\n"
                                    f"Question: {question}\n\n"
                                    "Please detect relevant objects, describe spatial relationships, "
                                    "and use the provided tool to return your answer."
                                ),
                            },
                        ],
                    }
                ],
            )

            answer = ""
            reasoning = ""
            bounding_boxes: List[BoundingBox] = []

            for block in response.content:
                if block["type"] == "tool_use" and block.get("name") == "provide_spatial_answer":
                    payload = block.get("input", {}) or {}
                    answer = payload.get("answer", "") or answer
                    reasoning = payload.get("reasoning", "") or reasoning
                    objects = payload.get("objects", []) or []

                    bounding_boxes = [
                        BoundingBox(
                            x1=obj["x1"],
                            y1=obj["y1"],
                            x2=obj["x2"],
                            y2=obj["y2"],
                            label=obj.get("label"),
                            confidence=obj.get("confidence", 0.9),
                        )
                        for obj in objects
                    ]
                elif block["type"] == "text":
                    text = block.get("text", "")
                    if text:
                        reasoning = (reasoning + "\n" + text).strip() if reasoning else text.strip()

            if not answer:
                # Fall back to using concatenated text as the answer
                answer = reasoning

            logger.info(f"Claude response: {len(bounding_boxes)} objects detected")
            logger.debug(f"Reasoning: {reasoning}")

            return {
                "answer": answer,
                "reasoning": reasoning,
                "bounding_boxes": bounding_boxes,
                "model": self.model
            }

        except Exception as e:
            logger.error(f"Error querying Claude: {str(e)}")
            raise

    async def self_correct_with_reasoning(
        self,
        image_base64: str,
        original_question: str,
        original_answer: str,
        original_reasoning: str,
        contradictions: List[Dict[str, Any]],
        proof_overlay_base64: str
    ) -> Dict[str, Any]:
        """Self-correction with explicit reasoning loop using the configured VLM."""
        try:
            processed_image = resize_image_if_needed(image_base64, max_size_mb=4.7, preserve_quality=True)
            processed_proof = resize_image_if_needed(proof_overlay_base64, max_size_mb=4.7, preserve_quality=True)

            image_media_type, image_data = self._extract_media_type_and_data(processed_image)
            proof_media_type, proof_data = self._extract_media_type_and_data(processed_proof)

            evidence_text = "\n\n".join([
                (
                    f"**Contradiction {i + 1}: {c['type'].upper()}**\n"
                    f"- Your claim: {c['claim']}\n"
                    f"- Geometric evidence: {c['evidence']}\n"
                    f"- Severity: {c['severity']:.1%}"
                )
                for i, c in enumerate(contradictions)
            ]) or "No contradictions provided."

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
                max_tokens=self.max_output_tokens,
                temperature=max(self.temperature, 0.2),
                system="You are a meticulous assistant that double-checks spatial claims against evidence.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": proof_media_type,
                                    "data": proof_data,
                                },
                            },
                        ],
                    }
                ],
            )

            # Concatenate all text blocks into a single response string
            full_response_parts: List[str] = []
            for block in response.content:
                if block["type"] == "text":
                    text = block.get("text", "")
                    if text:
                        full_response_parts.append(text)
            full_response = "\n".join(full_response_parts).strip()
            logger.info(f"Self-correction response received ({len(full_response)} chars)")
            logger.debug(f"Full response sample: {full_response[:500]}")

            revised_answer = full_response
            self_reflection = ""
            confidence = 0.7

            import re

            reflection_match = re.search(r"\*\*Self-Reflection:\*\*\s*(.+?)(?=\*\*Revised Answer:\*\*|\*\*Confidence:\*\*|$)", full_response, re.DOTALL)
            answer_match = re.search(r"\*\*Revised Answer:\*\*\s*(.+?)(?=\*\*Confidence:\*\*|$)", full_response, re.DOTALL)
            confidence_match = re.search(r"\*\*Confidence:\*\*\s*([0-9.]+)", full_response)

            if not reflection_match:
                reflection_match = re.search(r"Self-Reflection:\s*(.+?)(?=Revised Answer:|Confidence:|$)", full_response, re.DOTALL)
            if not answer_match:
                answer_match = re.search(r"Revised Answer:\s*(.+?)(?=Confidence:|$)", full_response, re.DOTALL)
            if not confidence_match:
                confidence_match = re.search(r"Confidence:\s*([0-9.]+)", full_response)

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
                "full_reasoning": full_response
            }

        except Exception as e:
            logger.error(f"Error during self-correction: {str(e)}")
            raise

    @staticmethod
    def _ensure_data_url(image_base64: str) -> str:
        """Ensure the image string has a proper data URL prefix."""
        if image_base64.startswith("data:image"):
            return image_base64
        return f"data:image/jpeg;base64,{image_base64}"

    @staticmethod
    def _extract_media_type_and_data(image_base64: str) -> Tuple[str, str]:
        """
        Extract media type and raw base64 data from a (possibly) data URL formatted image.

        Anthropic's Claude API expects raw base64 data plus an explicit media type.
        """
        if image_base64.startswith("data:") and "," in image_base64:
            header, data = image_base64.split(",", 1)
            # Example header: "data:image/jpeg;base64"
            try:
                mime_part = header.split(":", 1)[1]
                media_type = mime_part.split(";", 1)[0]
            except (IndexError, ValueError):
                media_type = "image/jpeg"
            return media_type, data

        # Fallback: assume jpeg if no explicit media type is present
        return "image/jpeg", (image_base64.split(",", 1)[1] if "," in image_base64 else image_base64)
