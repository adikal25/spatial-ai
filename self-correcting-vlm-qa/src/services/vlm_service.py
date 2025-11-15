"""
Vision-Language Model service using Claude (Anthropic).
Implements self-reasoning loop for spatial question answering.
"""
import os
import base64
import json
from typing import Dict, Any, List, Optional
from io import BytesIO

from anthropic import Anthropic
from loguru import logger

from src.models.schemas import BoundingBox
from src.utils.image_utils import resize_image_if_needed


class VLMService:
    """Service for interacting with Claude Vision."""

    def __init__(self):
        """Initialize VLM service with Anthropic client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = Anthropic(api_key=api_key)
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    async def ask_with_boxes(
        self,
        image_base64: str,
        question: str,
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Ask Claude a spatial question and request bounding boxes for detected objects.

        Args:
            image_base64: Base64-encoded image
            question: Spatial question about the image
            use_fallback: Not used (kept for compatibility)

        Returns:
            Dictionary with answer, bounding boxes, and metadata
        """
        return await self._ask_claude(image_base64, question)

    async def _ask_claude(self, image_base64: str, question: str) -> Dict[str, Any]:
        """
        Query Claude with tool use for structured bounding box output.

        Args:
            image_base64: Base64-encoded image
            question: Spatial question

        Returns:
            Response with answer and bounding boxes
        """
        try:
            logger.info("Processing image for Claude API...")

            # Resize image if needed (Claude has 5MB base64 limit)
            # Tries compression first, only resizes if necessary to preserve quality
            # Using 4.7MB to have safety margin
            image_base64 = resize_image_if_needed(image_base64, max_size_mb=4.7, preserve_quality=True)

            logger.info("Image processing complete, sending to Claude...")

            # Detect and clean base64 string
            if image_base64.startswith("data:image"):
                # Extract media type from data URL
                data_url_parts = image_base64.split(",")
                if len(data_url_parts) == 2:
                    header = data_url_parts[0]
                    if "image/png" in header:
                        media_type = "image/png"
                    elif "image/webp" in header:
                        media_type = "image/webp"
                    elif "image/gif" in header:
                        media_type = "image/gif"
                    else:
                        media_type = "image/jpeg"
                    image_base64 = data_url_parts[1]
                else:
                    media_type = "image/jpeg"
            else:
                # No data URL, assume JPEG (our compression outputs JPEG)
                media_type = "image/jpeg"

            # Define tool for bounding box extraction
            tools = [
                {
                    "name": "provide_spatial_answer",
                    "description": "Provide an answer to a spatial question with bounding boxes for detected objects. Use this tool to structure your response with object locations.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Natural language answer to the spatial question"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Your internal reasoning about the spatial relationships in the image"
                            },
                            "objects": {
                                "type": "array",
                                "description": "List of detected objects with bounding boxes (normalized coordinates 0-1)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Object label/name"
                                        },
                                        "x1": {
                                            "type": "number",
                                            "description": "Top-left x coordinate (0-1)"
                                        },
                                        "y1": {
                                            "type": "number",
                                            "description": "Top-left y coordinate (0-1)"
                                        },
                                        "x2": {
                                            "type": "number",
                                            "description": "Bottom-right x coordinate (0-1)"
                                        },
                                        "y2": {
                                            "type": "number",
                                            "description": "Bottom-right y coordinate (0-1)"
                                        },
                                        "confidence": {
                                            "type": "number",
                                            "description": "Confidence in detection (0-1)"
                                        }
                                    },
                                    "required": ["label", "x1", "y1", "x2", "y2"]
                                }
                            }
                        },
                        "required": ["answer", "reasoning", "objects"]
                    }
                }
            ]

            # Create message with image
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=tools,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": f"""Analyze this image and answer the following spatial question with careful reasoning:

**Question:** {question}

**Instructions:**
1. First, identify ALL relevant objects in the image
2. For EACH object, provide:
   - A clear label/name
   - Accurate bounding box (normalized coordinates 0-1)
   - Your estimate of its relative position (foreground/midground/background)

3. When making comparisons between objects, be EXPLICIT:
   - Use clear comparative language: "Object A is closer/further than Object B"
   - Use clear size language: "Object A is larger/smaller than Object B"
   - Avoid vague terms like "similar" or "about the same" unless they truly are

4. In your reasoning, explain your spatial judgment:
   - What visual cues indicate depth? (occlusion, size perspective, position in frame)
   - How do you estimate relative sizes?
   - What makes you confident in your assessment?

5. Use the provide_spatial_answer tool with:
   - A direct, clear answer to the question
   - Detailed reasoning explaining your spatial analysis
   - All detected objects with accurate bounding boxes

**Be specific and decisive** - avoid hedging unless genuinely uncertain."""
                            }
                        ]
                    }
                ]
            )

            # Parse tool use response
            answer = ""
            reasoning = ""
            bounding_boxes = []

            for block in response.content:
                if block.type == "tool_use" and block.name == "provide_spatial_answer":
                    tool_input = block.input
                    answer = tool_input.get("answer", "")
                    reasoning = tool_input.get("reasoning", "")
                    objects = tool_input.get("objects", [])

                    # Convert to BoundingBox objects
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
                elif block.type == "text":
                    # Fallback to text response if no tool use
                    if not answer:
                        answer = block.text

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
        """
        Self-correction with explicit reasoning loop.
        Claude reviews its original answer against geometric evidence.

        Args:
            image_base64: Original image
            original_question: Original question asked
            original_answer: Claude's initial answer
            original_reasoning: Claude's initial reasoning
            contradictions: List of detected contradictions
            proof_overlay_base64: Proof image with depth visualization

        Returns:
            Dictionary with revised answer, reasoning, and confidence
        """
        try:
            # Resize images if needed (Claude has 5MB base64 limit)
            # Preserves quality - tries compression first before resizing
            image_base64 = resize_image_if_needed(image_base64, max_size_mb=4.7, preserve_quality=True)
            proof_overlay_base64 = resize_image_if_needed(proof_overlay_base64, max_size_mb=4.7, preserve_quality=True)

            # Detect media types and clean base64 strings
            if image_base64.startswith("data:image"):
                parts = image_base64.split(",")
                header = parts[0]
                if "image/png" in header:
                    image_media_type = "image/png"
                elif "image/jpeg" in header or "image/jpg" in header:
                    image_media_type = "image/jpeg"
                else:
                    image_media_type = "image/jpeg"
                image_base64 = parts[1]
            else:
                image_media_type = "image/jpeg"

            if proof_overlay_base64.startswith("data:image"):
                parts = proof_overlay_base64.split(",")
                header = parts[0]
                if "image/png" in header:
                    proof_media_type = "image/png"
                elif "image/jpeg" in header or "image/jpg" in header:
                    proof_media_type = "image/jpeg"
                else:
                    proof_media_type = "image/jpeg"
                proof_overlay_base64 = parts[1]
            else:
                proof_media_type = "image/jpeg"

            logger.info(f"Image formats - Original: {image_media_type}, Proof: {proof_media_type}")

            # Build contradiction evidence
            evidence_text = "\n\n".join([
                f"**Contradiction {i+1}: {c['type'].upper()}**\n"
                f"- Your claim: {c['claim']}\n"
                f"- Geometric evidence: {c['evidence']}\n"
                f"- Severity: {c['severity']:.1%}"
                for i, c in enumerate(contradictions)
            ])

            # Self-correction prompt with reasoning loop
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""# Self-Correction with Multi-Signal Geometric Evidence

You previously answered a spatial question. Geometric analysis using **multiple visual cues** has found potential contradictions. The system only flags contradictions when multiple signals agree, so this evidence is reliable.

## Original Question
{original_question}

## Your Original Answer
{original_answer}

## Your Original Reasoning
{original_reasoning}

## ⚠️ Contradictions Detected
{evidence_text}

**Note**: These contradictions were detected using multiple validation signals (depth analysis, occlusion detection, vertical position), not just depth alone. The system only flags issues when there's strong supporting evidence.

## Evidence Provided
I'm showing you two images:
1. **Left**: The original image
2. **Right**: Depth map visualization
   - **Warmer colors (red/yellow) = CLOSER to camera** (lower depth values)
   - **Cooler colors (blue/purple) = FURTHER from camera** (higher depth values)
   - Bounding boxes show detected objects with depth values

## Your Task: Critical Self-Evaluation

Follow this reasoning process carefully:

### Step 1: Re-examine the Original Image
- Look again at the spatial relationships
- Check for occlusion (what's in front of what?)
- Check vertical position (lower objects often closer)
- Consider size perspective
- Were there any ambiguities or assumptions you made?

### Step 2: Analyze the Multi-Signal Evidence
- Study the depth map on the right
- Check if occlusion patterns support the depth readings
- Verify if vertical positions align with distance claims
- Do multiple cues point to the same conclusion?

### Step 3: Identify Your Error (if any)
- **If you were wrong**: What caused the error? (perspective illusion, wrong assumption, missed occlusion)
- **If evidence conflicts**: Note that the system requires multiple signals to agree, so contradictions are likely valid
- **If you're still correct**: Explain why - but consider that multiple independent cues are agreeing

### Step 4: Provide Corrected Answer
- Give a clear, direct answer to the original question
- Reference the specific visual cues that support your answer
- Be honest about uncertainty if it exists

## Response Format

**Self-Reflection:**
[Detailed analysis: What visual cues did you observe? Did you make an error? If so, what caused it? Do the multiple signals (depth + occlusion + position) agree?]

**Revised Answer:**
[Your final answer to: "{original_question}" - Be clear and specific. If you were wrong, give the corrected answer. If you were right, reaffirm with explanation.]

**Confidence:**
[A single number between 0.0 and 1.0]

Remember: The system uses multiple independent visual cues and only flags contradictions with strong evidence. Take this feedback seriously."""
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": proof_media_type,
                                    "data": proof_overlay_base64
                                }
                            }
                        ]
                    }
                ]
            )

            # Parse response
            full_response = ""
            for block in response.content:
                if block.type == "text":
                    full_response = block.text

            logger.info(f"Self-correction response received ({len(full_response)} chars)")
            logger.debug(f"Full response: {full_response[:500]}...")

            # Extract components from response
            revised_answer = full_response
            self_reflection = ""
            confidence = 0.7  # Default

            # Try to parse structured response (be flexible with formatting)
            import re

            # Try with bold markers first
            reflection_match = re.search(r'\*\*Self-Reflection:\*\*\s*(.+?)(?=\*\*Revised Answer:\*\*|\*\*Confidence:\*\*|$)', full_response, re.DOTALL)
            answer_match = re.search(r'\*\*Revised Answer:\*\*\s*(.+?)(?=\*\*Confidence:\*\*|$)', full_response, re.DOTALL)
            confidence_match = re.search(r'\*\*Confidence:\*\*\s*([0-9.]+)', full_response)

            # Fallback to non-bold markers
            if not reflection_match:
                reflection_match = re.search(r'Self-Reflection:\s*(.+?)(?=Revised Answer:|Confidence:|$)', full_response, re.DOTALL)
            if not answer_match:
                answer_match = re.search(r'Revised Answer:\s*(.+?)(?=Confidence:|$)', full_response, re.DOTALL)
            if not confidence_match:
                confidence_match = re.search(r'Confidence:\s*([0-9.]+)', full_response)

            if reflection_match:
                self_reflection = reflection_match.group(1).strip()
                logger.info(f"Extracted self-reflection ({len(self_reflection)} chars)")

            if answer_match:
                revised_answer = answer_match.group(1).strip()
                logger.info(f"Extracted revised answer ({len(revised_answer)} chars)")
            else:
                # If no structured answer found, use full response
                logger.warning("Could not extract structured revised answer, using full response")
                revised_answer = full_response

            if confidence_match:
                conf_val = float(confidence_match.group(1))
                confidence = conf_val if conf_val <= 1 else conf_val / 100
                logger.info(f"Extracted confidence: {confidence}")

            logger.info(f"Self-correction parsing complete - Confidence: {confidence}")

            return {
                "revised_answer": revised_answer,
                "self_reflection": self_reflection,
                "confidence": confidence,
                "full_reasoning": full_response
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
