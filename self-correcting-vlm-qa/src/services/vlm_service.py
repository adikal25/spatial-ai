"""
Vision-Language Model service using OpenAI GPT-5-nano.
Handles spatial question answering plus self-correction.
"""
import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from src.models.schemas import BoundingBox
from src.utils.image_utils import resize_image_if_needed


class VLMService:
    """Service for interacting with OpenAI's multimodal GPT models."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_VLM_MODEL", "gpt-5-nano")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2048"))

    async def ask_with_boxes(
        self,
        image_base64: str,
        question: str,
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """Ask GPT-5-nano a spatial question and request bounding boxes."""
        return await self._ask_openai(image_base64, question)

    async def _ask_openai(self, image_base64: str, question: str) -> Dict[str, Any]:
        try:
            logger.info("Processing image for OpenAI GPT-5-nano API...")
            processed_image = resize_image_if_needed(image_base64, max_size_mb=4.7, preserve_quality=True)
            data_url = self._ensure_data_url(processed_image)

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "provide_spatial_answer",
                        "description": "Provide an answer to a spatial question with bounding boxes for detected objects.",
                        "parameters": {
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
                                            "confidence": {"type": "number"}
                                        },
                                        "required": ["label", "x1", "y1", "x2", "y2"]
                                    }
                                }
                            },
                            "required": ["answer", "reasoning", "objects"]
                        }
                    }
                }
            ]

            user_content = [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image and answer the following spatial question.\n\n"
                        f"Question: {question}\n\n"
                        "Please detect relevant objects, describe spatial relationships, and use the provided function to return your answer."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
               # temperature=self.temperature,
             #   max_completion_tokens=self.max_output_tokens, # to run with gpt nano
                tools=tools,
                tool_choice="auto",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise spatial reasoner. Always return bounding boxes via the provided tool."
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
            )

            choice = response.choices[0]
            answer = ""
            reasoning = ""
            bounding_boxes: List[BoundingBox] = []

            if choice.message.tool_calls:
                for call in choice.message.tool_calls:
                    if call.function.name != "provide_spatial_answer":
                        continue
                    try:
                        payload = json.loads(call.function.arguments)
                    except json.JSONDecodeError as exc:
                        logger.error(f"Failed to decode tool arguments: {exc}")
                        continue

                    answer = payload.get("answer", "")
                    reasoning = payload.get("reasoning", "")
                    objects = payload.get("objects", [])

                    bounding_boxes = [
                        BoundingBox(
                            x1=obj["x1"],
                            y1=obj["y1"],
                            x2=obj["x2"],
                            y2=obj["y2"],
                            label=obj.get("label"),
                            confidence=obj.get("confidence", 0.9)
                        )
                        for obj in objects
                    ]
            else:
                message_content = choice.message.content
                if isinstance(message_content, list):
                    answer = "\n".join([part.get("text", "") for part in message_content if isinstance(part, dict)]).strip()
                else:
                    answer = message_content or ""

            logger.info(f"OpenAI response: {len(bounding_boxes)} objects detected")
            logger.debug(f"Reasoning: {reasoning}")

            return {
                "answer": answer,
                "reasoning": reasoning,
                "bounding_boxes": bounding_boxes,
                "model": self.model
            }

        except Exception as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
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
        """Self-correction with explicit reasoning loop using GPT-5-nano."""
        try:
            processed_image = resize_image_if_needed(image_base64, max_size_mb=4.7, preserve_quality=True)
            processed_proof = resize_image_if_needed(proof_overlay_base64, max_size_mb=4.7, preserve_quality=True)

            image_data_url = self._ensure_data_url(processed_image)
            proof_data_url = self._ensure_data_url(processed_proof)

            evidence_text = "\n\n".join([
                (
                    f"**Contradiction {i + 1}: {c['type'].upper()}**\n"
                    f"- Your claim: {c['claim']}\n"
                    f"- Geometric evidence: {c['evidence']}\n"
                    f"- Severity: {c['severity']:.1%}"
                )
                for i, c in enumerate(contradictions)
            ]) or "No contradictions provided."

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

            response = self.client.chat.completions.create(
                model=self.model,
               # temperature=max(self.temperature, 0.2),
               # max_completion_tokens=self.max_output_tokens, # to run with gpt nano max comletion tokens param changed from max_tokens
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meticulous assistant that double-checks spatial claims against evidence."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "image_url", "image_url": {"url": proof_data_url}}
                        ]
                    }
                ]
            )

            full_response = response.choices[0].message.content or ""
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
