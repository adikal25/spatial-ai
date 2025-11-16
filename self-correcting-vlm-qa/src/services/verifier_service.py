"""
Geometric verification service that detects contradictions in VLM responses.
Uses depth estimation and geometric analysis.
"""
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from src.models.schemas import BoundingBox, SpatialMetrics, Contradiction
from src.services.reconstruction_service import (
    ReconstructionResult,
    compute_mesh_object_stats,
)


class VerificationResult:
    """Result of geometric verification."""

    def __init__(
        self,
        spatial_metrics: List[SpatialMetrics],
        contradictions: List[Contradiction],
        proof_overlay: str,
        reconstruction_preview: Optional[str] = None,
        reconstruction_metadata: Optional[Dict[str, Any]] = None
    ):
        self.spatial_metrics = spatial_metrics
        self.contradictions = contradictions
        self.proof_overlay = proof_overlay
        self.reconstruction_preview = reconstruction_preview
        self.reconstruction_metadata = reconstruction_metadata


class VerifierService:
    """Service for verifying VLM responses with geometric reasoning."""

    def __init__(self, depth_service):
        """Initialize verifier with depth service."""
        self.depth_service = depth_service

        # Thresholds for contradiction detection
        self.relative_size_threshold = 0.3  # 30% difference
        self.relative_distance_threshold = 0.2  # 20% difference

    async def verify(
        self,
        image_base64: str,
        vlm_response: Dict[str, Any],
        question: str,
        reconstruction: Optional[ReconstructionResult] = None
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

            if reconstruction and reconstruction.mesh:
                self._inject_mesh_metrics(spatial_metrics, reconstruction.mesh)

            # Detect contradictions
            contradictions = self._detect_contradictions(
                vlm_response["answer"],
                spatial_metrics,
                question
            )

            # Create proof overlay
            reconstruction_preview = reconstruction.preview_base64 if reconstruction else None
            proof_overlay = self._create_proof_overlay(
                image_base64,
                depth_map,
                spatial_metrics,
                contradictions,
                reconstruction_preview=reconstruction_preview
            )

            logger.info(f"Verification complete: {len(contradictions)} contradictions found")

            return VerificationResult(
                spatial_metrics=spatial_metrics,
                contradictions=contradictions,
                proof_overlay=proof_overlay,
                reconstruction_preview=reconstruction_preview,
                reconstruction_metadata=reconstruction.metadata if reconstruction else None
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

    @staticmethod
    def _inject_mesh_metrics(
        metrics: List[SpatialMetrics],
        mesh
    ) -> None:
        """Augment metrics with coarse 3D stats from the reconstructed mesh."""
        if not metrics:
            return

        for metric in metrics:
            stats = compute_mesh_object_stats(mesh, metric.bounding_box)
            if not stats:
                continue
            metric.mesh_centroid = stats.centroid
            metric.mesh_extent = stats.extent
            metric.mesh_point_count = stats.point_count

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
        """Check if claimed relative distances match depth measurements."""
        contradictions = []

        # Simple heuristic: check if answer claims one object is "closer" or "further"
        answer_lower = answer.lower()

        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                obj1 = metrics[i]
                obj2 = metrics[j]

                depth1 = self._get_depth_anchor(obj1)
                depth2 = self._get_depth_anchor(obj2)

                # Check depth difference
                depth_diff = abs(depth1 - depth2)
                avg_depth = (depth1 + depth2) / 2

                if avg_depth > 0:
                    relative_diff = depth_diff / avg_depth

                    # Look for distance claims in answer
                    if relative_diff > self.relative_distance_threshold:
                        # Significant depth difference detected
                        closer_obj = obj1 if depth1 < depth2 else obj2
                        further_obj = obj2 if depth1 < depth2 else obj1
                        closer_depth = depth1 if closer_obj is obj1 else depth2
                        further_depth = depth2 if closer_obj is obj1 else depth1

                        # This is a simplified check - in practice, need NLP analysis
                        if "same distance" in answer_lower or "equal distance" in answer_lower:
                            contradictions.append(Contradiction(
                                type="distance",
                                claim=f"Objects at same distance",
                                evidence=(
                                    f"{closer_obj.object_id} "
                                    f"({self._format_depth_hint(closer_obj, closer_depth)}) is "
                                    f"significantly closer than {further_obj.object_id} "
                                    f"({self._format_depth_hint(further_obj, further_depth)})"
                                ),
                                severity=0.7
                            ))

        return contradictions

    def _check_relative_sizes(
        self,
        answer: str,
        metrics: List[SpatialMetrics]
    ) -> List[Contradiction]:
        """Check if claimed relative sizes match measurements."""
        contradictions = []

        # Compare object sizes
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                obj1 = metrics[i]
                obj2 = metrics[j]

                size1 = self._get_size_estimate(obj1)
                size2 = self._get_size_estimate(obj2)

                if size1 > 0 and size2 > 0:
                    size_ratio = max(size1, size2) / min(size1, size2)

                    if size_ratio > (1 + self.relative_size_threshold):
                        # Significant size difference
                        # Check if answer claims they're similar size
                        answer_lower = answer.lower()

                        if "same size" in answer_lower or "similar size" in answer_lower:
                            larger = obj1 if size1 > size2 else obj2
                            smaller = obj2 if size1 > size2 else obj1

                            contradictions.append(Contradiction(
                                type="size",
                                claim="Objects are similar size",
                                evidence=f"{larger.object_id} is {size_ratio:.1f}x larger than {smaller.object_id}",
                                severity=0.6
                            ))

        return contradictions

    def _check_object_counts(
        self,
        answer: str,
        metrics: List[SpatialMetrics],
        question: str
    ) -> List[Contradiction]:
        """Check if counted objects match detected objects."""
        contradictions = []

        # Extract numbers from answer
        import re
        numbers = re.findall(r'\b\d+\b', answer)

        if numbers and metrics:
            claimed_count = int(numbers[0])  # First number in answer
            detected_count = len(metrics)

            if claimed_count != detected_count and abs(claimed_count - detected_count) > 1:
                contradictions.append(Contradiction(
                    type="count",
                    claim=f"Answer mentions {claimed_count} objects",
                    evidence=f"Detected {detected_count} objects in image",
                    severity=0.8
                ))

        return contradictions

    def _create_proof_overlay(
        self,
        image_base64: str,
        depth_map: np.ndarray,
        spatial_metrics: List[SpatialMetrics],
        contradictions: List[Contradiction],
        reconstruction_preview: Optional[str] = None
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

        preview_img = None
        if reconstruction_preview:
            try:
                preview_img = self._decode_base64_image(reconstruction_preview).resize(img.size)
            except Exception as exc:
                logger.warning(f"Unable to decode reconstruction preview: {exc}")
                preview_img = None

        # Create side-by-side comparison
        panels = 3 if preview_img else 2
        combined_width = img.width * panels
        combined = Image.new("RGB", (combined_width, img.height))
        combined.paste(img, (0, 0))
        combined.paste(depth_img, (img.width, 0))
        if preview_img:
            combined.paste(preview_img, (img.width * 2, 0))

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

        if preview_img:
            draw.text((img.width * 2 + 10, 10), "TripoSG mesh", fill="white")

        # Encode to base64
        buffered = BytesIO()
        combined.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    @staticmethod
    def _decode_base64_image(image_base64: str) -> Image.Image:
        """Decode base64 string or data URL into PIL image."""
        if image_base64.startswith("data:image"):
            _, encoded = image_base64.split(",", maxsplit=1)
        else:
            encoded = image_base64
        return Image.open(BytesIO(base64.b64decode(encoded)))

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

    @staticmethod
    def _get_depth_anchor(metric: SpatialMetrics) -> float:
        """Prefer mesh-derived depth if available."""
        if metric.mesh_centroid and len(metric.mesh_centroid) >= 3:
            return float(metric.mesh_centroid[2])
        return float(metric.depth_mean)

    @staticmethod
    def _format_depth_hint(metric: SpatialMetrics, value: float) -> str:
        """Format textual hint depending on available cues."""
        if metric.mesh_centroid:
            return f"z={value:.2f}"
        return f"depth={value:.2f}"

    @staticmethod
    def _get_size_estimate(metric: SpatialMetrics) -> float:
        """Combine mesh extents and 2D box into comparable scalar."""
        if metric.mesh_extent and len(metric.mesh_extent) == 3:
            extent = np.maximum(np.array(metric.mesh_extent), 1e-6)
            return float(extent[0] * extent[1] * extent[2])

        size = metric.estimated_size
        return float(size["width"] * size["height"])
