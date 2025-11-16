# src/services/verifier_service.py
"""
Geometric verification service that detects contradictions in VLM responses.
Uses depth estimation, geometric analysis, and optional 3D reconstruction via fVDB.
"""
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

from src.models.schemas import BoundingBox, SpatialMetrics, Contradiction
from src.services.fvdb_3d_service import Fvdb3DReconstructionService


class VerificationResult:
    """Result of geometric verification."""

    def __init__(
        self,
        spatial_metrics: List[SpatialMetrics],
        contradictions: List[Contradiction],
        proof_overlay: str,
        fvdb_debug: Optional[Dict[str, Any]] = None,
    ):
        self.spatial_metrics = spatial_metrics
        self.contradictions = contradictions
        self.proof_overlay = proof_overlay
        self.fvdb_debug = fvdb_debug or {}


class VerifierService:
    """Service for verifying VLM responses with geometric reasoning."""

    def __init__(
        self,
        depth_service,
        fvdb_3d_service: Optional[Fvdb3DReconstructionService] = None,
    ):
        """Initialize verifier with depth and optional fVDB 3D service."""
        self.depth_service = depth_service
        self.fvdb_3d_service = fvdb_3d_service

        # Thresholds for contradiction detection
        self.relative_size_threshold = 0.3  # 30% difference
        self.relative_distance_threshold = 0.2  # 20% difference

    async def verify(
        self,
        image_base64: str,
        vlm_response: Dict[str, Any],
        question: str,
    ) -> VerificationResult:
        """
        Verify VLM response using depth geometry (2.5D) and optional 3D metrics.

        Args:
            image_base64: Original image (base64)
            vlm_response: VLM response with answer and bounding boxes
            question: Original question

        Returns:
            VerificationResult with metrics, contradictions, proof overlay, and 3D debug info.
        """
        try:
            # Estimate depth map
            depth_map = await self.depth_service.estimate_depth(image_base64)

            # Extract spatial metrics for each detected object (2.5D)
            spatial_metrics = self._compute_spatial_metrics(
                depth_map,
                vlm_response.get("bounding_boxes", []),
            )

            # Augment with 3D centroids & distances if fVDB is enabled
            fvdb_debug: Optional[Dict[str, Any]] = None
            if self.fvdb_3d_service is not None and self.fvdb_3d_service.enabled:
                fvdb_debug = self.verify_with_3d(
                    depth_map=depth_map,
                    bounding_boxes=vlm_response.get("bounding_boxes", []),
                    spatial_metrics=spatial_metrics,
                    answer=vlm_response.get("answer", ""),
                    reasoning=vlm_response.get("reasoning", ""),
                )

            # Detect contradictions (uses 3D z if available)
            contradictions = self._detect_contradictions(
                vlm_response["answer"],
                spatial_metrics,
                question,
            )

            # Create proof overlay
            proof_overlay = self._create_proof_overlay(
                image_base64,
                depth_map,
                spatial_metrics,
                contradictions,
            )

            logger.info(
                f"Verification complete: {len(contradictions)} contradictions found "
                f"(3D enabled={fvdb_debug is not None and fvdb_debug.get('enabled', False)})"
            )

            return VerificationResult(
                spatial_metrics=spatial_metrics,
                contradictions=contradictions,
                proof_overlay=proof_overlay,
                fvdb_debug=fvdb_debug,
            )

        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            raise

    def _compute_spatial_metrics(
        self,
        depth_map: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> List[SpatialMetrics]:
        """
        Compute spatial metrics for detected objects using the depth map.

        Args:
            depth_map: Estimated depth map
            bounding_boxes: List of bounding boxes

        Returns:
            List of spatial metrics for each object
        """
        metrics: List[SpatialMetrics] = []

        for i, bbox in enumerate(bounding_boxes):
            # Extract depth for this object
            mean_depth, std_depth = self.depth_service.extract_object_depth(
                depth_map,
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
                normalized=True,
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
                bounding_box=bbox,
            )

            metrics.append(metric)

        return metrics

    def verify_with_3d(
        self,
        depth_map: np.ndarray,
        bounding_boxes: List[BoundingBox],
        spatial_metrics: List[SpatialMetrics],
        answer: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        """
        Run 3D reconstruction + analysis using fVDB and attach centroids to metrics.

        Returns:
            Dict with:
              - enabled: bool
              - voxel_count: int
              - centroids_3d: list[Optional[dict(x,y,z)]]
              - pairwise_distances: dict "i-j" -> float
        """
        if self.fvdb_3d_service is None or not self.fvdb_3d_service.enabled:
            return {"enabled": False}

        geom = self.fvdb_3d_service.compute_object_geometry(depth_map, bounding_boxes)
        scene = geom.get("scene")
        centroids = geom.get("centroids_3d", [])
        pairwise = geom.get("pairwise_distances", {})

        # Attach centroids to SpatialMetrics in-place
        centroids_dicts: List[Optional[Dict[str, float]]] = []
        for metric, c in zip(spatial_metrics, centroids):
            if c is None:
                metric.centroid_3d = None
                centroids_dicts.append(None)
            else:
                centroid_dict = {
                    "x": float(c[0]),
                    "y": float(c[1]),
                    "z": float(c[2]),
                }
                metric.centroid_3d = centroid_dict
                centroids_dicts.append(centroid_dict)

        # Serialize pairwise distances with friendly keys
        pairwise_str_keys = {
            f"{i}-{j}": float(d) for (i, j), d in pairwise.items()
        }

        debug = {
            "enabled": True,
            "voxel_count": int(scene.voxel_count) if scene is not None else 0,
            "centroids_3d": centroids_dicts,
            "pairwise_distances": pairwise_str_keys,
        }

        logger.info(
            f"fVDB 3D analysis: enabled=True, "
            f"voxel_count={debug['voxel_count']}, "
            f"objects_with_centroids="
            f"{sum(1 for c in centroids_dicts if c is not None)}"
        )

        return debug

    def _detect_contradictions(
        self,
        answer: str,
        spatial_metrics: List[SpatialMetrics],
        question: str,
    ) -> List[Contradiction]:
        """
        Detect contradictions between VLM answer and geometric measurements.

        Uses 3D centroid z when available; falls back to depth_mean.
        """
        contradictions: List[Contradiction] = []

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
        count_contradictions = self._check_object_counts(
            answer, spatial_metrics, question
        )
        contradictions.extend(count_contradictions)

        return contradictions

    def _check_relative_distances(
        self,
        answer: str,
        metrics: List[SpatialMetrics],
    ) -> List[Contradiction]:
        """Check if claimed relative distances match geometric measurements."""
        contradictions: List[Contradiction] = []

        answer_lower = answer.lower()

        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                obj1 = metrics[i]
                obj2 = metrics[j]

                # Prefer 3D centroid z if available; otherwise use depth_mean
                z1 = (
                    obj1.centroid_3d["z"]
                    if obj1.centroid_3d is not None
                    else obj1.depth_mean
                )
                z2 = (
                    obj2.centroid_3d["z"]
                    if obj2.centroid_3d is not None
                    else obj2.depth_mean
                )

                depth_diff = abs(z1 - z2)
                avg_depth = (z1 + z2) / 2 if (z1 + z2) != 0 else 0.0

                if avg_depth > 0:
                    relative_diff = depth_diff / avg_depth

                    if relative_diff > self.relative_distance_threshold:
                        # Significant depth difference detected
                        closer_obj = obj1 if z1 < z2 else obj2
                        further_obj = obj2 if z1 < z2 else obj1

                        # Simple heuristic: if answer claims "same distance"
                        if "same distance" in answer_lower or "equal distance" in answer_lower:
                            contradictions.append(
                                Contradiction(
                                    type="distance",
                                    claim="Objects at same distance",
                                    evidence=(
                                        f"{closer_obj.object_id} (z={z1:.2f if z1 < z2 else z2:.2f}) "
                                        f"is significantly closer than "
                                        f"{further_obj.object_id} (z={z2:.2f if z1 < z2 else z1:.2f})"
                                    ),
                                    severity=0.7,
                                )
                            )

        return contradictions

    def _check_relative_sizes(
        self,
        answer: str,
        metrics: List[SpatialMetrics],
    ) -> List[Contradiction]:
        """Check if claimed relative sizes match measurements."""
        contradictions: List[Contradiction] = []

        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                obj1 = metrics[i]
                obj2 = metrics[j]

                size1 = obj1.estimated_size["width"] * obj1.estimated_size["height"]
                size2 = obj2.estimated_size["width"] * obj2.estimated_size["height"]

                if size1 > 0 and size2 > 0:
                    size_ratio = max(size1, size2) / min(size1, size2)

                    if size_ratio > (1 + self.relative_size_threshold):
                        answer_lower = answer.lower()

                        if "same size" in answer_lower or "similar size" in answer_lower:
                            larger = obj1 if size1 > size2 else obj2
                            smaller = obj2 if size1 > size2 else obj1

                            contradictions.append(
                                Contradiction(
                                    type="size",
                                    claim="Objects are similar size",
                                    evidence=(
                                        f"{larger.object_id} is {size_ratio:.1f}x "
                                        f"larger than {smaller.object_id}"
                                    ),
                                    severity=0.6,
                                )
                            )

        return contradictions

    def _check_object_counts(
        self,
        answer: str,
        metrics: List[SpatialMetrics],
        question: str,
    ) -> List[Contradiction]:
        """Check if counted objects match detected objects."""
        contradictions: List[Contradiction] = []

        import re

        numbers = re.findall(r"\b\d+\b", answer)

        if numbers and metrics:
            claimed_count = int(numbers[0])
            detected_count = len(metrics)

            if claimed_count != detected_count and abs(claimed_count - detected_count) > 1:
                contradictions.append(
                    Contradiction(
                        type="count",
                        claim=f"Answer mentions {claimed_count} objects",
                        evidence=f"Detected {detected_count} objects in image",
                        severity=0.8,
                    )
                )

        return contradictions

    def _create_proof_overlay(
        self,
        image_base64: str,
        depth_map: np.ndarray,
        spatial_metrics: List[SpatialMetrics],
        contradictions: List[Contradiction],
    ) -> str:
        """
        Create proof overlay image with depth visualization and annotations.
        """
        # Decode original image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))

        # Create depth visualization
        depth_normalized = cv2.normalize(
            depth_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
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

        centroid_color = "#00FF7F"  # bright green
        text_color = "#FF3030"  # vivid red for numbering

        for idx, metric in enumerate(spatial_metrics):
            bbox = metric.bounding_box
            x1, y1, x2, y2 = self._bbox_to_pixels(bbox, img.width, img.height)

            if x2 <= x1 or y2 <= y1:
                continue

            number_label = str(idx)
            text_y = max(0, y1 - 18)

            # Draw bounding box + label on original image (left panel)
            draw.rectangle([x1, y1, x2, y2], outline="white", width=2)
            draw.text((x1, text_y), number_label, fill=text_color)

            # Mark centroid (approximate center of box)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=centroid_color, outline=centroid_color)

            # Mirror annotations on depth visualization (right panel)
            dx1 = x1 + img.width
            dx2 = x2 + img.width
            draw.rectangle([dx1, y1, dx2, y2], outline="white", width=2)
            draw.text((dx1, text_y), number_label, fill=text_color)
            dx = cx + img.width
            draw.ellipse([dx - 4, cy - 4, dx + 4, cy + 4], fill=centroid_color, outline=centroid_color)

        # Encode to base64
        buffered = BytesIO()
        combined.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    @staticmethod
    def _bbox_to_pixels(bbox: BoundingBox, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert potentially normalized bbox coordinates into pixel units."""
        coords = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=float)
        normalized_guess = np.all((coords >= -0.05) & (coords <= 1.05))

        if normalized_guess:
            x1 = int(coords[0] * width)
            y1 = int(coords[1] * height)
            x2 = int(coords[2] * width)
            y2 = int(coords[3] * height)
        else:
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        return x1, y1, x2, y2

    @staticmethod
    def _depth_to_distance(depth_value: float) -> float:
        """
        Convert depth map value to estimated distance (rough approximation).
        """
        return depth_value / 10.0
