"""
Geometric verification service that detects contradictions in VLM responses.
Uses depth estimation and geometric analysis.
"""
import base64
import numpy as np
from typing import Dict, Any, List
from io import BytesIO

import cv2
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from src.models.schemas import BoundingBox, SpatialMetrics, Contradiction


class VerificationResult:
    """Result of geometric verification."""

    def __init__(
        self,
        spatial_metrics: List[SpatialMetrics],
        contradictions: List[Contradiction],
        proof_overlay: str
    ):
        self.spatial_metrics = spatial_metrics
        self.contradictions = contradictions
        self.proof_overlay = proof_overlay


class VerifierService:
    """Service for verifying VLM responses with geometric reasoning."""

    def __init__(self, depth_service):
        """Initialize verifier with depth service."""
        self.depth_service = depth_service

        # Thresholds for contradiction detection
        self.relative_size_threshold = 0.3  # 30% difference
        self.relative_distance_threshold = 0.3  # Increased to 30% for more confidence

        # Depth reliability settings
        self.depth_confidence_threshold = 0.4  # Require 40% depth difference for high confidence
        self.require_multiple_signals = True  # Require multiple cues to agree

    async def verify(
        self,
        image_base64: str,
        vlm_response: Dict[str, Any],
        question: str
    ) -> VerificationResult:
        """
        Verify VLM response using depth geometry.

        Args:
            image_base64: Original image
            vlm_response: VLM response with answer and bounding boxes
            question: Original question

        Returns:
            VerificationResult with metrics, contradictions, and proof overlay
        """
        try:
            # Estimate depth map
            depth_map = await self.depth_service.estimate_depth(image_base64)

            # Extract spatial metrics for each detected object
            spatial_metrics = self._compute_spatial_metrics(
                depth_map,
                vlm_response.get("bounding_boxes", [])
            )

            # Detect contradictions
            contradictions = self._detect_contradictions(
                vlm_response["answer"],
                spatial_metrics,
                question
            )

            # Create proof overlay
            proof_overlay = self._create_proof_overlay(
                image_base64,
                depth_map,
                spatial_metrics,
                contradictions
            )

            logger.info(f"Verification complete: {len(contradictions)} contradictions found")

            return VerificationResult(
                spatial_metrics=spatial_metrics,
                contradictions=contradictions,
                proof_overlay=proof_overlay
            )

        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            raise

    def _compute_spatial_metrics(
        self,
        depth_map: np.ndarray,
        bounding_boxes: List[BoundingBox]
    ) -> List[SpatialMetrics]:
        """
        Compute spatial metrics for detected objects.

        Args:
            depth_map: Estimated depth map
            bounding_boxes: List of bounding boxes

        Returns:
            List of spatial metrics for each object
        """
        metrics = []

        for i, bbox in enumerate(bounding_boxes):
            # Extract depth for this object
            mean_depth, std_depth = self.depth_service.extract_object_depth(
                depth_map,
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
                normalized=True
            )

            # Estimate size (normalized units)
            width = bbox.x2 - bbox.x1
            height = bbox.y2 - bbox.y1

            metric = SpatialMetrics(
                object_id=f"obj_{i}_{bbox.label or 'unknown'}",
                depth_mean=mean_depth,
                depth_std=std_depth,
                estimated_distance=self._depth_to_distance(mean_depth),
                estimated_size={"width": width, "height": height},
                bounding_box=bbox
            )

            metrics.append(metric)

        return metrics

    def _check_occlusion(self, bbox1: BoundingBox, bbox2: BoundingBox) -> str:
        """
        Check if one bounding box occludes another.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            "bbox1_occludes_bbox2", "bbox2_occludes_bbox1", or "no_occlusion"
        """
        # Calculate intersection
        x_left = max(bbox1.x1, bbox2.x1)
        y_top = max(bbox1.y1, bbox2.y1)
        x_right = min(bbox1.x2, bbox2.x2)
        y_bottom = min(bbox1.y2, bbox2.y2)

        if x_right <= x_left or y_bottom <= y_top:
            return "no_occlusion"

        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas of each box
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)

        # If intersection is significant (>30% of smaller box), check which occludes
        overlap_ratio1 = intersection_area / area1 if area1 > 0 else 0
        overlap_ratio2 = intersection_area / area2 if area2 > 0 else 0

        # Significant overlap threshold
        if max(overlap_ratio1, overlap_ratio2) > 0.3:
            # The larger box is likely behind the smaller one if they overlap significantly
            # Or if one contains the other
            if area1 > area2 * 1.5:
                return "bbox2_occludes_bbox1"  # Smaller box likely in front
            elif area2 > area1 * 1.5:
                return "bbox1_occludes_bbox2"

        return "no_occlusion"

    def _get_vertical_position_cue(self, bbox1: BoundingBox, bbox2: BoundingBox) -> str:
        """
        Get vertical position cue (objects lower in frame are often closer).

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            "bbox1_lower", "bbox2_lower", or "similar_height"
        """
        # Use bottom of bounding box for comparison
        bottom1 = bbox1.y2
        bottom2 = bbox2.y2

        # Significant difference threshold (10% of image height)
        threshold = 0.1

        if bottom1 > bottom2 + threshold:
            return "bbox1_lower"  # bbox1 is lower in frame (often closer)
        elif bottom2 > bottom1 + threshold:
            return "bbox2_lower"
        else:
            return "similar_height"

    def _detect_contradictions(
        self,
        answer: str,
        spatial_metrics: List[SpatialMetrics],
        question: str
    ) -> List[Contradiction]:
        """
        Detect contradictions between VLM answer and geometric measurements.

        Args:
            answer: VLM's answer
            spatial_metrics: Computed spatial metrics
            question: Original question

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Check relative distances
        if len(spatial_metrics) >= 2:
            contradictions.extend(
                self._check_relative_distances(answer, spatial_metrics)
            )

        # Check relative sizes
        if len(spatial_metrics) >= 2:
            contradictions.extend(
                self._check_relative_sizes(answer, spatial_metrics)
            )

        # Check object counts
        count_contradictions = self._check_object_counts(answer, spatial_metrics, question)
        contradictions.extend(count_contradictions)

        return contradictions

    def _check_relative_distances(
        self,
        answer: str,
        metrics: List[SpatialMetrics]
    ) -> List[Contradiction]:
        """Check if claimed relative distances match depth measurements using multiple cues."""
        contradictions = []

        answer_lower = answer.lower()

        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                obj1 = metrics[i]
                obj2 = metrics[j]

                # Extract object labels for matching
                label1 = (obj1.bounding_box.label or f"object{i}").lower()
                label2 = (obj2.bounding_box.label or f"object{j}").lower()

                # SIGNAL 1: Depth analysis
                depth_diff = abs(obj1.depth_mean - obj2.depth_mean)
                avg_depth = (obj1.depth_mean + obj2.depth_mean) / 2

                if avg_depth > 0:
                    relative_diff = depth_diff / avg_depth

                    # Determine relationship from depth
                    closer_by_depth = obj1 if obj1.depth_mean < obj2.depth_mean else obj2
                    further_by_depth = obj2 if obj1.depth_mean < obj2.depth_mean else obj1
                    closer_label_depth = label1 if obj1.depth_mean < obj2.depth_mean else label2
                    further_label_depth = label2 if obj1.depth_mean < obj2.depth_mean else label1

                    # SIGNAL 2: Occlusion
                    occlusion = self._check_occlusion(obj1.bounding_box, obj2.bounding_box)

                    # SIGNAL 3: Vertical position
                    vertical_cue = self._get_vertical_position_cue(obj1.bounding_box, obj2.bounding_box)

                    # Count how many signals agree
                    signals_agreeing = 0
                    supporting_evidence = []

                    # Check if depth is reliable (high confidence)
                    depth_is_reliable = relative_diff > self.depth_confidence_threshold

                    if depth_is_reliable:
                        signals_agreeing += 1
                        supporting_evidence.append(f"depth ({closer_by_depth.object_id} depth={closer_by_depth.depth_mean:.1f} vs {further_by_depth.object_id} depth={further_by_depth.depth_mean:.1f}, {relative_diff:.1%} difference)")

                    # Check occlusion signal
                    if occlusion == "bbox1_occludes_bbox2" and closer_by_depth == obj1:
                        signals_agreeing += 1
                        supporting_evidence.append(f"occlusion ({label1} occludes {label2})")
                    elif occlusion == "bbox2_occludes_bbox1" and closer_by_depth == obj2:
                        signals_agreeing += 1
                        supporting_evidence.append(f"occlusion ({label2} occludes {label1})")

                    # Check vertical position signal
                    if vertical_cue == "bbox1_lower" and closer_by_depth == obj1:
                        signals_agreeing += 1
                        supporting_evidence.append(f"vertical position ({label1} lower in frame)")
                    elif vertical_cue == "bbox2_lower" and closer_by_depth == obj2:
                        signals_agreeing += 1
                        supporting_evidence.append(f"vertical position ({label2} lower in frame)")

                    # Only flag contradiction if we have strong evidence (multiple signals OR very high depth confidence)
                    has_strong_evidence = (signals_agreeing >= 2) or (depth_is_reliable and relative_diff > 0.5)

                    if has_strong_evidence and relative_diff > self.relative_distance_threshold:
                        # Significant depth difference - objects are NOT at same distance
                        # Use the depth-based determination since we have strong evidence
                        closer_obj = closer_by_depth
                        further_obj = further_by_depth
                        closer_label = closer_label_depth
                        further_label = further_label_depth

                        # Format multi-signal evidence
                        evidence_str = f"Multiple visual cues agree: {', '.join(supporting_evidence)}"

                        # Check if answer claims they're at same/similar distance
                        same_distance_phrases = [
                            "same distance", "equal distance", "equidistant",
                            "similar distance", "about the same distance",
                            "roughly equal distance", "comparable distance"
                        ]

                        for phrase in same_distance_phrases:
                            if phrase in answer_lower:
                                contradictions.append(Contradiction(
                                    type="distance",
                                    claim=f"Objects at {phrase}",
                                    evidence=evidence_str,
                                    severity=min(0.9, 0.5 + (signals_agreeing * 0.15))
                                ))
                                break

                        # Check if answer claims WRONG order (closer/further reversed)
                        # Pattern: "[label1] is closer/nearer" when label1 should be further
                        closer_patterns = [
                            f"{further_label} is closer",
                            f"{further_label} is nearer",
                            f"{further_label} appears closer",
                            f"{further_label} seems closer",
                            f"the {further_label} is closer",
                            f"the {further_label} is nearer",
                        ]

                        for pattern in closer_patterns:
                            if pattern in answer_lower:
                                contradictions.append(Contradiction(
                                    type="distance",
                                    claim=f"{further_obj.object_id} claimed closer to camera",
                                    evidence=evidence_str,
                                    severity=min(0.95, 0.6 + (signals_agreeing * 0.15))
                                ))
                                break

                        # Pattern: "[label2] is further/farther" when label2 should be closer
                        further_patterns = [
                            f"{closer_label} is further",
                            f"{closer_label} is farther",
                            f"{closer_label} is more distant",
                            f"{closer_label} appears further",
                            f"the {closer_label} is further",
                            f"the {closer_label} is farther",
                        ]

                        for pattern in further_patterns:
                            if pattern in answer_lower:
                                contradictions.append(Contradiction(
                                    type="distance",
                                    claim=f"{closer_obj.object_id} claimed further from camera",
                                    evidence=evidence_str,
                                    severity=min(0.95, 0.6 + (signals_agreeing * 0.15))
                                ))
                                break

                    else:
                        # Depth values are similar - objects ARE at similar distance
                        # Check if answer incorrectly claims one is significantly closer

                        strong_distance_phrases = [
                            f"{label1} is much closer",
                            f"{label1} is significantly closer",
                            f"{label1} is far closer",
                            f"{label2} is much closer",
                            f"{label2} is significantly closer",
                            f"{label2} is far closer",
                            f"{label1} is much further",
                            f"{label1} is much farther",
                            f"{label2} is much further",
                            f"{label2} is much farther",
                        ]

                        for phrase in strong_distance_phrases:
                            if phrase in answer_lower:
                                contradictions.append(Contradiction(
                                    type="distance",
                                    claim=f"Claims significant distance difference",
                                    evidence=f"Objects have similar depths: {obj1.object_id}={obj1.depth_mean:.1f}, {obj2.object_id}={obj2.depth_mean:.1f} (only {relative_diff:.1%} difference)",
                                    severity=0.6
                                ))
                                break

        return contradictions

    def _check_relative_sizes(
        self,
        answer: str,
        metrics: List[SpatialMetrics]
    ) -> List[Contradiction]:
        """Check if claimed relative sizes match measurements."""
        contradictions = []
        answer_lower = answer.lower()

        # Compare object sizes
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                obj1 = metrics[i]
                obj2 = metrics[j]

                # Extract object labels for matching
                label1 = (obj1.bounding_box.label or f"object{i}").lower()
                label2 = (obj2.bounding_box.label or f"object{j}").lower()

                size1 = obj1.estimated_size["width"] * obj1.estimated_size["height"]
                size2 = obj2.estimated_size["width"] * obj2.estimated_size["height"]

                if size1 > 0 and size2 > 0:
                    size_ratio = max(size1, size2) / min(size1, size2)
                    larger_obj = obj1 if size1 > size2 else obj2
                    smaller_obj = obj2 if size1 > size2 else obj1
                    larger_label = label1 if size1 > size2 else label2
                    smaller_label = label2 if size1 > size2 else label1

                    if size_ratio > (1 + self.relative_size_threshold):
                        # Significant size difference exists

                        # Check if answer claims they're similar/same size
                        similar_size_phrases = [
                            "same size", "similar size", "equal size",
                            "about the same size", "roughly the same size",
                            "comparable size", "similar in size",
                            "equally sized", "same dimensions"
                        ]

                        for phrase in similar_size_phrases:
                            if phrase in answer_lower:
                                contradictions.append(Contradiction(
                                    type="size",
                                    claim=f"Objects claimed to be {phrase}",
                                    evidence=f"{larger_obj.object_id} is {size_ratio:.1f}x larger in bounding box area than {smaller_obj.object_id}",
                                    severity=min(0.9, 0.4 + (size_ratio - 1) * 0.2)
                                ))
                                break

                        # Check if answer claims WRONG order (larger/smaller reversed)
                        larger_patterns = [
                            f"{smaller_label} is larger",
                            f"{smaller_label} is bigger",
                            f"{smaller_label} is much larger",
                            f"{smaller_label} appears larger",
                            f"the {smaller_label} is larger",
                            f"the {smaller_label} is bigger",
                        ]

                        for pattern in larger_patterns:
                            if pattern in answer_lower:
                                contradictions.append(Contradiction(
                                    type="size",
                                    claim=f"{smaller_obj.object_id} claimed to be larger",
                                    evidence=f"Bounding box analysis shows {larger_obj.object_id} is actually {size_ratio:.1f}x larger than {smaller_obj.object_id}",
                                    severity=min(0.95, 0.5 + (size_ratio - 1) * 0.2)
                                ))
                                break

                        smaller_patterns = [
                            f"{larger_label} is smaller",
                            f"{larger_label} is much smaller",
                            f"{larger_label} appears smaller",
                            f"the {larger_label} is smaller",
                        ]

                        for pattern in smaller_patterns:
                            if pattern in answer_lower:
                                contradictions.append(Contradiction(
                                    type="size",
                                    claim=f"{larger_obj.object_id} claimed to be smaller",
                                    evidence=f"Bounding box analysis shows {larger_obj.object_id} is actually {size_ratio:.1f}x larger than {smaller_obj.object_id}",
                                    severity=min(0.95, 0.5 + (size_ratio - 1) * 0.2)
                                ))
                                break

                    else:
                        # Sizes are similar - check if answer claims significant difference
                        strong_size_phrases = [
                            f"{label1} is much larger",
                            f"{label1} is much bigger",
                            f"{label1} is significantly larger",
                            f"{label2} is much larger",
                            f"{label2} is much bigger",
                            f"{label2} is significantly larger",
                            f"{label1} is much smaller",
                            f"{label2} is much smaller",
                        ]

                        for phrase in strong_size_phrases:
                            if phrase in answer_lower:
                                contradictions.append(Contradiction(
                                    type="size",
                                    claim=f"Claims significant size difference",
                                    evidence=f"Objects have similar bounding box sizes (ratio: {size_ratio:.2f}x)",
                                    severity=0.6
                                ))
                                break

        return contradictions

    def _check_object_counts(
        self,
        answer: str,
        metrics: List[SpatialMetrics],
        question: str
    ) -> List[Contradiction]:
        """Check if counted objects match detected objects."""
        contradictions = []
        import re

        # Look for count-related patterns in the question to determine what's being counted
        question_lower = question.lower()
        answer_lower = answer.lower()

        # Extract what is being counted from the question
        count_patterns = [
            r'how many (\w+)',
            r'count (?:the )?(\w+)',
            r'number of (\w+)',
        ]

        object_type = None
        for pattern in count_patterns:
            match = re.search(pattern, question_lower)
            if match:
                object_type = match.group(1)
                break

        # Extract numbers with context from answer
        # Pattern: "number word objects" or "word number" or "number"
        number_word_map = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'no': 0, 'single': 1, 'couple': 2, 'pair': 2, 'several': 3,
            'few': 3, 'multiple': 2, 'many': 5
        }

        claimed_count = None

        # Try to find digit-based counts first
        digit_matches = re.findall(r'\b(\d+)\s*(\w+)?', answer_lower)
        for match in digit_matches:
            number = int(match[0])
            context = match[1] if len(match) > 1 else ""

            # Skip if it looks like a measurement or year
            if any(unit in context for unit in ['px', 'cm', 'mm', 'inches', 'feet', 'meter', '%']):
                continue

            # If we know what object type we're counting, check if it matches
            if object_type and object_type in answer_lower:
                # Try to find number near the object type
                pattern = rf'\b(\d+)\s*{object_type}|\b{object_type}\s*(\d+)'
                specific_match = re.search(pattern, answer_lower)
                if specific_match:
                    claimed_count = int(specific_match.group(1) or specific_match.group(2))
                    break

            # Otherwise use first number that seems like a count
            if claimed_count is None:
                claimed_count = number

        # Try word-based counts if no digits found
        if claimed_count is None:
            for word, value in number_word_map.items():
                # Look for patterns like "three cars", "no objects", "a single item"
                pattern = rf'\b{word}\b'
                if re.search(pattern, answer_lower):
                    claimed_count = value
                    break

        # Compare claimed count with detected count
        if claimed_count is not None and metrics:
            detected_count = len(metrics)

            # Allow tolerance of 1 for edge detection issues
            if abs(claimed_count - detected_count) > 1:
                # Calculate severity based on discrepancy
                discrepancy = abs(claimed_count - detected_count)
                severity = min(0.95, 0.6 + (discrepancy * 0.1))

                contradictions.append(Contradiction(
                    type="count",
                    claim=f"Answer claims {claimed_count} {object_type or 'objects'}",
                    evidence=f"Detected {detected_count} distinct objects with bounding boxes",
                    severity=severity
                ))
            elif claimed_count == 0 and detected_count > 0:
                # Special case: claimed none but found some
                contradictions.append(Contradiction(
                    type="count",
                    claim=f"Answer claims no {object_type or 'objects'} present",
                    evidence=f"Detected {detected_count} objects in the image",
                    severity=0.9
                ))

        return contradictions

    def _create_proof_overlay(
        self,
        image_base64: str,
        depth_map: np.ndarray,
        spatial_metrics: List[SpatialMetrics],
        contradictions: List[Contradiction]
    ) -> str:
        """
        Create proof overlay image with depth visualization and annotations.

        Args:
            image_base64: Original image
            depth_map: Depth map
            spatial_metrics: Spatial metrics
            contradictions: Detected contradictions

        Returns:
            Base64-encoded proof overlay image
        """
        # Decode original image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))

        # Create depth visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
        depth_img = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))

        # Resize to match original
        depth_img = depth_img.resize(img.size)

        # Create side-by-side comparison
        combined_width = img.width * 2
        combined = Image.new("RGB", (combined_width, img.height))
        combined.paste(img, (0, 0))
        combined.paste(depth_img, (img.width, 0))

        # Draw annotations
        draw = ImageDraw.Draw(combined)

        # Draw bounding boxes and metrics
        for metric in spatial_metrics:
            bbox = metric.bounding_box
            x1 = int(bbox.x1 * img.width)
            y1 = int(bbox.y1 * img.height)
            x2 = int(bbox.x2 * img.width)
            y2 = int(bbox.y2 * img.height)

            # Draw on original image
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 15), f"{bbox.label or 'obj'}", fill="green")

            # Draw on depth map
            draw.rectangle([x1 + img.width, y1, x2 + img.width, y2], outline="yellow", width=2)
            draw.text((x1 + img.width, y1 - 15), f"d={metric.depth_mean:.1f}", fill="yellow")

        # Encode to base64
        buffered = BytesIO()
        combined.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    @staticmethod
    def _depth_to_distance(depth_value: float) -> float:
        """
        Convert depth map value to estimated distance (rough approximation).

        Args:
            depth_value: Depth value from model

        Returns:
            Estimated distance in arbitrary units
        """
        # This is a rough approximation - would need calibration for real distances
        return depth_value / 10.0
