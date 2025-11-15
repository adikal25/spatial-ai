"""
Depth estimation service using MiDaS or ZoeDepth.
Provides depth maps for geometric verification.
"""
import os
import base64
import numpy as np
from typing import Optional, Tuple
from io import BytesIO

import torch
import cv2
from PIL import Image
from loguru import logger


class DepthService:
    """Service for depth estimation from images."""

    def __init__(self):
        """Initialize depth estimation service."""
        self.model_name = os.getenv("DEPTH_MODEL", "midas_v3_small")
        self.device = torch.device("cuda" if torch.cuda.is_available() and os.getenv("ENABLE_GPU", "true").lower() == "true" else "cpu")
        self.model = None
        self.transform = None

        logger.info(f"Depth service initialized with model: {self.model_name}, device: {self.device}")

    async def load_model(self):
        """Load depth estimation model."""
        try:
            if "midas" in self.model_name.lower():
                await self._load_midas()
            elif "zoe" in self.model_name.lower():
                await self._load_zoedepth()
            else:
                raise ValueError(f"Unknown depth model: {self.model_name}")

            logger.info(f"Depth model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Error loading depth model: {str(e)}")
            raise

    async def _load_midas(self):
        """Load MiDaS depth estimation model."""
        # Load MiDaS model from torch hub
        # Available models: DPT_Large, DPT_Hybrid, MiDaS_small
        model_type = "MiDaS_small"  # Default to small for speed

        if "large" in self.model_name.lower():
            model_type = "DPT_Large"
        elif "hybrid" in self.model_name.lower():
            model_type = "DPT_Hybrid"
        else:
            model_type = "MiDaS_small"

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform

    async def _load_zoedepth(self):
        """Load ZoeDepth model (alternative)."""
        # This requires ZoeDepth to be installed
        # pip install git+https://github.com/isl-org/ZoeDepth.git
        raise NotImplementedError("ZoeDepth support not yet implemented. Use MiDaS instead.")

    async def estimate_depth(self, image_base64: str) -> np.ndarray:
        """
        Estimate depth map from image.

        Args:
            image_base64: Base64-encoded image

        Returns:
            Depth map as numpy array (higher values = further away)
        """
        if self.model is None:
            raise RuntimeError("Depth model not loaded. Call load_model() first.")

        try:
            # Decode image
            img = self._decode_image(image_base64)

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy array
            img_np = np.array(img)

            # Apply transforms
            input_batch = self.transform(img_np).to(self.device)

            # Predict depth
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_np.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            logger.info(f"Depth map estimated: shape={depth_map.shape}, min={depth_map.min():.2f}, max={depth_map.max():.2f}")

            return depth_map

        except Exception as e:
            logger.error(f"Error estimating depth: {str(e)}")
            raise

    def extract_object_depth(
        self,
        depth_map: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        normalized: bool = True
    ) -> Tuple[float, float]:
        """
        Extract depth statistics for a bounding box region.

        Args:
            depth_map: Depth map array
            x1, y1, x2, y2: Bounding box coordinates
            normalized: Whether coordinates are normalized (0-1)

        Returns:
            Tuple of (mean_depth, std_depth)
        """
        h, w = depth_map.shape

        # Convert normalized coordinates to pixel coordinates
        if normalized:
            x1_px = int(x1 * w)
            y1_px = int(y1 * h)
            x2_px = int(x2 * w)
            y2_px = int(y2 * h)
        else:
            x1_px, y1_px, x2_px, y2_px = int(x1), int(y1), int(x2), int(y2)

        # Ensure coordinates are within bounds
        x1_px = max(0, min(x1_px, w - 1))
        y1_px = max(0, min(y1_px, h - 1))
        x2_px = max(0, min(x2_px, w))
        y2_px = max(0, min(y2_px, h))

        # Extract region
        region = depth_map[y1_px:y2_px, x1_px:x2_px]

        if region.size == 0:
            logger.warning("Empty region for depth extraction")
            return 0.0, 0.0

        mean_depth = float(np.mean(region))
        std_depth = float(np.std(region))

        return mean_depth, std_depth

    def create_depth_visualization(self, depth_map: np.ndarray) -> str:
        """
        Create a colored visualization of the depth map.

        Args:
            depth_map: Depth map array

        Returns:
            Base64-encoded visualization image
        """
        # Normalize depth map to 0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

        # Convert to PIL Image
        depth_img = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))

        # Encode to base64
        buffered = BytesIO()
        depth_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    @staticmethod
    def _decode_image(image_base64: str) -> Image.Image:
        """Decode base64 image string to PIL Image."""
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_base64)

        # Open as PIL Image
        img = Image.open(BytesIO(image_bytes))

        return img
