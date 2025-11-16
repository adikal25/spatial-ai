"""
Geometric verification service that detects contradictions in VLM responses.
Uses depth estimation and geometric analysis.
"""
import os
import base64
import numpy as np
from typing import Dict, Any, List, Optional
from io import BytesIO

import cv2
from PIL import Image, ImageDraw
from loguru import logger

from src.models.schemas import (
    BoundingBox,
    SpatialMetrics,
    Contradiction,
    ComparisonClaim,
    CountClaim,
)


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

        # Thresholds for contradiction detection (configurable via environment)
        self.relative_size_threshold = float(os.getenv("RELATIVE_SIZE_THRESHOLD", "0.3"))
        self.relative_distance_threshold = float(os.getenv("RELATIVE_DISTANCE_THRESHOLD", "0.3"))

        # Depth reliability settings
        self.depth_confidence_threshold = float(os.getenv("DEPTH_CONFIDENCE_THRESHOLD", "0.4"))
        self.require_multiple_signals = True  # Require multiple cues to agree

        logger.info(f"Verifier initialized with thresholds - size: {self.relative_size_threshold}, "
                   f"distance: {self.relative_distance_threshold}, depth_confidence: {self.depth_confidence_threshold}")

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

            # Build structured claims from the VLM output
            structured_claims = vlm_response.get("structured_claims", {}) or {}
            comparison_claims = self._build_comparison_claims(structured_claims.get("comparisons"))
            count_claims = self._build_count_claims(structured_claims.get("counts"))

            # Detect contradictions
            contradictions = self._detect_contradictions(
                spatial_metrics=spatial_metrics,
                comparisons=comparison_claims,
                count_claims=count_claims
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

            object_id = bbox.object_id or f"obj_{i}_{bbox.label or 'unknown'}"

            metric = SpatialMetrics(
                object_id=object_id,
                depth_mean=mean_depth,
                depth_std=std_depth,
                estimated_distance=self._depth_to_distance(mean_depth),
                estimated_size={"width": width, "height": height},
                bounding_box=bbox
            )

            # Ensure downstream lookups can rely on bounding_box.object_id
            if not bbox.object_id:
                bbox.object_id = object_id

            metrics.append(metric)

        return metrics

    @staticmethod
    def _build_comparison_claims(raw_claims: Optional[List[Dict[str, Any]]]) -> List[ComparisonClaim]:
        """Validate and convert raw comparison claims to pydantic models."""
        claims: List[ComparisonClaim] = []
        if not raw_claims:
            return claims

        for claim in raw_claims:
            try:
                claims.append(ComparisonClaim(**claim))
            except Exception as err:
                logger.warning(f"Skipping malformed comparison claim from VLM: {err}")

        return claims

    @staticmethod
    def _build_count_claims(raw_counts: Optional[List[Dict[str, Any]]]) -> List[CountClaim]:
        """Validate and convert raw count claims to pydantic models."""
        claims: List[CountClaim] = []
        if not raw_counts:
            return claims

        for claim in raw_counts:
            try:
                claims.append(CountClaim(**claim))
            except Exception as err:
                logger.warning(f"Skipping malformed count claim from VLM: {err}")

        return claims

    def _check_occlusion(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        depth1: Optional[float] = None,
        depth2: Optional[float] = None
    ) -> str:
        """
        Check if one bounding box occludes another using geometric overlap and depth cues.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            depth1: Mean depth of bbox1 (lower = closer)
            depth2: Mean depth of bbox2 (lower = closer)

        Returns:
            "bbox1_occludes_bbox2", "bbox2_occludes_bbox1", or "no_occlusion"
        """
        label1 = bbox1.label or "obj1"
        label2 = bbox2.label or "obj2"

        # Calculate intersection
        x_left = max(bbox1.x1, bbox2.x1)
        y_top = max(bbox1.y1, bbox2.y1)
        x_right = min(bbox1.x2, bbox2.x2)
        y_bottom = min(bbox1.y2, bbox2.y2)

        # No overlap at all
        if x_right <= x_left or y_bottom <= y_top:
            logger.debug(f"Occlusion check [{label1} vs {label2}]: No overlap detected")
            return "no_occlusion"

        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas of each box
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)

        # Calculate overlap ratios from both perspectives
        overlap_ratio1 = intersection_area / area1 if area1 > 0 else 0
        overlap_ratio2 = intersection_area / area2 if area2 > 0 else 0

        # Need significant overlap to consider occlusion (>15% threshold)
        overlap_threshold = 0.15
        if max(overlap_ratio1, overlap_ratio2) < overlap_threshold:
            logger.debug(f"Occlusion check [{label1} vs {label2}]: Overlap too small "
                        f"({overlap_ratio1:.1%}/{overlap_ratio2:.1%} < {overlap_threshold:.1%})")
            return "no_occlusion"

        logger.debug(f"Occlusion check [{label1} vs {label2}]: Overlap detected - "
                    f"{label1} covered {overlap_ratio1:.1%}, {label2} covered {overlap_ratio2:.1%}")

        # Strategy 1: Use depth information if available (most reliable)
        if depth1 is not None and depth2 is not None:
            depth_diff = abs(depth1 - depth2)
            avg_depth = (depth1 + depth2) / 2

            # If depth difference is significant (>20% relative difference)
            if avg_depth > 0 and (depth_diff / avg_depth) > 0.2:
                # Object with lower depth value is closer (in front)
                if depth1 < depth2:
                    logger.info(f"✓ Occlusion [Strategy: Depth-Aware]: {label1} occludes {label2} "
                               f"(depth: {depth1:.1f} < {depth2:.1f}, diff: {depth_diff/avg_depth:.1%})")
                    return "bbox1_occludes_bbox2"
                else:
                    logger.info(f"✓ Occlusion [Strategy: Depth-Aware]: {label2} occludes {label1} "
                               f"(depth: {depth2:.1f} < {depth1:.1f}, diff: {depth_diff/avg_depth:.1%})")
                    return "bbox2_occludes_bbox1"
            else:
                logger.debug(f"Occlusion check [{label1} vs {label2}]: Depth difference too small "
                            f"({depth_diff/avg_depth:.1%} < 20%), trying geometric strategies")

        # Strategy 2: Geometric heuristics when depth is inconclusive

        # Check for containment - if one box mostly contains the other
        if overlap_ratio2 > 0.8:  # bbox2 is mostly inside bbox1
            # The contained box is likely in front
            logger.info(f"✓ Occlusion [Strategy: Containment]: {label2} occludes {label1} "
                       f"({label2} {overlap_ratio2:.1%} contained in {label1})")
            return "bbox2_occludes_bbox1"
        elif overlap_ratio1 > 0.8:  # bbox1 is mostly inside bbox2
            logger.info(f"✓ Occlusion [Strategy: Containment]: {label1} occludes {label2} "
                       f"({label1} {overlap_ratio1:.1%} contained in {label2})")
            return "bbox1_occludes_bbox2"

        # Check size heuristic - when similarly sized objects overlap,
        # the one that's smaller in screen space is often further away
        # (size constancy principle)
        size_ratio = area1 / area2 if area2 > 0 else 1.0

        if size_ratio > 2.0:
            # bbox1 is much larger - likely behind
            logger.info(f"✓ Occlusion [Strategy: Size Heuristic]: {label2} occludes {label1} "
                       f"({label1} is {size_ratio:.1f}x larger, likely behind)")
            return "bbox2_occludes_bbox1"
        elif size_ratio < 0.5:
            # bbox2 is much larger - likely behind
            logger.info(f"✓ Occlusion [Strategy: Size Heuristic]: {label1} occludes {label2} "
                       f"({label2} is {1/size_ratio:.1f}x larger, likely behind)")
            return "bbox1_occludes_bbox2"

        # Overlap exists but can't determine occlusion order confidently
        logger.debug(f"Occlusion check [{label1} vs {label2}]: Overlap exists but cannot determine order "
                    f"(size_ratio: {size_ratio:.2f}, depth_available: {depth1 is not None})")
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
        spatial_metrics: List[SpatialMetrics],
        comparisons: List[ComparisonClaim],
        count_claims: List[CountClaim]
    ) -> List[Contradiction]:
        """Detect contradictions between structured VLM claims and geometric measurements."""
        contradictions: List[Contradiction] = []

        if not spatial_metrics:
            return contradictions

        metric_lookup = {
            metric.bounding_box.object_id or metric.object_id: metric
            for metric in spatial_metrics
        }
        label_index = self._build_label_index(spatial_metrics)

        for claim in comparisons:
            subject_metric = metric_lookup.get(claim.subject_id)
            object_metric = metric_lookup.get(claim.object_id)

            if not subject_metric or not object_metric:
                logger.warning(
                    "Skipping comparison claim referencing unknown IDs: %s vs %s",
                    claim.subject_id,
                    claim.object_id
                )
                continue

            if claim.attribute == "distance":
                contradiction = self._evaluate_distance_claim(claim, subject_metric, object_metric)
            else:
                contradiction = self._evaluate_size_claim(claim, subject_metric, object_metric)

            if contradiction:
                contradictions.append(contradiction)

        contradictions.extend(self._evaluate_count_claims(count_claims, label_index, metric_lookup))

        return contradictions

    @staticmethod
    def _build_label_index(spatial_metrics: List[SpatialMetrics]) -> Dict[str, List[SpatialMetrics]]:
        """Map normalized labels to their corresponding metrics."""
        label_index: Dict[str, List[SpatialMetrics]] = {}
        for metric in spatial_metrics:
            label = (metric.bounding_box.label or '').strip().lower()
            if not label:
                continue
            label_index.setdefault(label, []).append(metric)
        return label_index

    def _evaluate_distance_claim(
        self,
        claim: ComparisonClaim,
        subject_metric: SpatialMetrics,
        object_metric: SpatialMetrics
    ) -> Optional[Contradiction]:
        """Validate a distance relationship claim using multi-signal evidence."""
        evidence = self._gather_distance_evidence(subject_metric, object_metric)
        relation = claim.relation

        if relation == "similar":
            if evidence["relative_diff"] > self.relative_distance_threshold:
                return Contradiction(
                    type="distance",
                    claim=f"{claim.subject_id} and {claim.object_id} claimed similar distance",
                    evidence=evidence["description"],
                    severity=min(0.9, 0.5 + (evidence["signals"] * 0.15))
                )
            return None

        if evidence["relative_diff"] < (self.relative_distance_threshold / 2):
            return Contradiction(
                type="distance",
                claim=f"{claim.subject_id} claimed {relation} than {claim.object_id}",
                evidence=(
                    f"Depth difference is minimal ({evidence['relative_diff']:.1%}); "
                    "objects measure at nearly the same distance."
                ),
                severity=0.55
            )

        if not evidence["has_strong_support"]:
            logger.debug(
                "Insufficient agreement between signals for distance claim %s vs %s",
                claim.subject_id,
                claim.object_id
            )
            return None

        expected_closer = subject_metric if relation == "closer" else object_metric
        actual_closer = evidence["closer"]

        if expected_closer.object_id != actual_closer.object_id:
            severity = min(0.95, 0.6 + (evidence["signals"] * 0.15))
            return Contradiction(
                type="distance",
                claim=f"{claim.subject_id} claimed {relation} than {claim.object_id}",
                evidence=evidence["description"],
                severity=severity
            )

        return None

    def _evaluate_size_claim(
        self,
        claim: ComparisonClaim,
        subject_metric: SpatialMetrics,
        object_metric: SpatialMetrics
    ) -> Optional[Contradiction]:
        """Validate size relationship claims using bounding-box areas."""
        size1 = subject_metric.estimated_size["width"] * subject_metric.estimated_size["height"]
        size2 = object_metric.estimated_size["width"] * object_metric.estimated_size["height"]

        if size1 <= 0 or size2 <= 0:
            return None

        if size1 >= size2:
            larger_metric, smaller_metric = subject_metric, object_metric
            ratio = size1 / size2 if size2 > 0 else float("inf")
        else:
            larger_metric, smaller_metric = object_metric, subject_metric
            ratio = size2 / size1 if size1 > 0 else float("inf")

        relation = claim.relation
        significant_difference = ratio > (1 + self.relative_size_threshold)

        if relation in {"same_size", "similar"}:
            if significant_difference:
                return Contradiction(
                    type="size",
                    claim=f"{claim.subject_id} and {claim.object_id} claimed similar size",
                    evidence=f"{larger_metric.object_id} is {ratio:.1f}× larger than {smaller_metric.object_id}",
                    severity=min(0.9, 0.4 + (ratio - 1.0) * 0.2)
                )
            return None

        if not significant_difference:
            return Contradiction(
                type="size",
                claim=f"{claim.subject_id} claimed {relation} than {claim.object_id}",
                evidence="Bounding boxes are nearly equal in area; no measurable dominance.",
                severity=0.55
            )

        expected_larger = subject_metric if relation == "larger" else object_metric
        actual_larger = larger_metric

        if expected_larger.object_id != actual_larger.object_id:
            severity = min(0.95, 0.5 + (ratio - 1.0) * 0.2)
            return Contradiction(
                type="size",
                claim=f"{claim.subject_id} claimed {relation} than {claim.object_id}",
                evidence=f"Bounding box analysis shows {actual_larger.object_id} is {ratio:.1f}× larger.",
                severity=severity
            )

        return None

    def _evaluate_count_claims(
        self,
        count_claims: List[CountClaim],
        label_index: Dict[str, List[SpatialMetrics]],
        metric_lookup: Dict[str, SpatialMetrics]
    ) -> List[Contradiction]:
        contradictions: List[Contradiction] = []

        for claim in count_claims:
            if claim.object_ids:
                detected = [metric_lookup.get(obj_id) for obj_id in claim.object_ids]
                detected_count = len([m for m in detected if m is not None])
                tolerance = 0
            else:
                detected_objects = label_index.get(claim.object_type.lower(), [])
                detected_count = len(detected_objects)
                tolerance = 1

            difference = abs(claim.count - detected_count)

            if difference > tolerance:
                severity = min(0.95, 0.6 + difference * 0.1)
                contradictions.append(Contradiction(
                    type="count",
                    claim=f"Answer claims {claim.count} {claim.object_type}",
                    evidence=f"Detected {detected_count} matching objects (tolerance {tolerance}).",
                    severity=severity
                ))

        return contradictions

    def _gather_distance_evidence(
        self,
        metric_a: SpatialMetrics,
        metric_b: SpatialMetrics
    ) -> Dict[str, Any]:
        """Aggregate signals that describe which object is closer."""
        depth_diff = abs(metric_a.depth_mean - metric_b.depth_mean)
        avg_depth = (metric_a.depth_mean + metric_b.depth_mean) / 2 or 1e-6
        relative_diff = depth_diff / avg_depth

        depth_closer = metric_a if metric_a.depth_mean < metric_b.depth_mean else metric_b
        depth_further = metric_b if depth_closer == metric_a else metric_a

        closer = depth_closer
        further = depth_further

        signals_agreeing = 0
        supporting_evidence: List[str] = []
        occlusion_support = False
        vertical_support = False

        depth_is_reliable = relative_diff > self.depth_confidence_threshold
        if depth_is_reliable:
            signals_agreeing += 1
            supporting_evidence.append(
                f"depth ({depth_closer.object_id}={depth_closer.depth_mean:.2f} vs {depth_further.object_id}={depth_further.depth_mean:.2f})"
            )

        occlusion = self._check_occlusion(
            metric_a.bounding_box,
            metric_b.bounding_box,
            depth1=metric_a.depth_mean,
            depth2=metric_b.depth_mean
        )
        occlusion_front = None
        occlusion_back = None
        if occlusion == "bbox1_occludes_bbox2":
            occlusion_front, occlusion_back = metric_a, metric_b
        elif occlusion == "bbox2_occludes_bbox1":
            occlusion_front, occlusion_back = metric_b, metric_a

        if occlusion_front is not None:
            occlusion_support = True
            supporting_evidence.append(
                f"occlusion ({occlusion_front.bounding_box.label} in front of {occlusion_back.bounding_box.label})"
            )
            signals_agreeing += 1

            # If depth is unreliable or disagrees, let occlusion decide ordering
            if closer.object_id != occlusion_front.object_id and (
                not depth_is_reliable or relative_diff < self.relative_distance_threshold
            ):
                closer = occlusion_front
                further = occlusion_back

        vertical_cue = self._get_vertical_position_cue(metric_a.bounding_box, metric_b.bounding_box)
        if vertical_cue == "bbox1_lower" and closer == metric_a:
            signals_agreeing += 1
            vertical_support = True
            supporting_evidence.append(f"vertical cue ({metric_a.bounding_box.label} lower in frame)")
        elif vertical_cue == "bbox2_lower" and closer == metric_b:
            signals_agreeing += 1
            vertical_support = True
            supporting_evidence.append(f"vertical cue ({metric_b.bounding_box.label} lower in frame)")

        has_strong_support = (
            signals_agreeing >= 2
            or (depth_is_reliable and relative_diff > 0.5)
            or occlusion_support
            or (vertical_support and depth_is_reliable)
        )
        description = "Multiple cues agree: " + ", ".join(supporting_evidence) if supporting_evidence else "Depth cues are weak"

        return {
            "closer": closer,
            "further": further,
            "relative_diff": relative_diff,
            "signals": signals_agreeing,
            "has_strong_support": has_strong_support,
            "description": description
        }

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

        # Create depth visualization (warmer colors = closer objects)
        depth_viz = np.clip(1.0 - depth_map, 0.0, 1.0)
        depth_normalized = (depth_viz * 255).astype(np.uint8)
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
        Convert depth map value to relative distance metric.

        Note: This returns a RELATIVE distance metric, not absolute distance in meters.
        Monocular depth estimation provides relative depth only. Without camera calibration
        and known object sizes, we cannot compute absolute distances.

        The returned value is useful for comparing relative distances between objects,
        but should not be interpreted as actual meters/feet.

        Args:
            depth_value: Normalized depth value (0 = closer, 1 = further)

        Returns:
            Relative distance metric (0-100) for readability
        """
        return depth_value * 100.0
