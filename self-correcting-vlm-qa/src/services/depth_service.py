"""
Depth estimation service using Depth Anything V2, MiDaS, or ZoeDepth.
Provides depth maps for geometric verification.
"""
import asyncio
import os
import base64
import numpy as np
from typing import Optional, Tuple
from io import BytesIO

import torch
import cv2
from PIL import Image
from loguru import logger
from transformers import pipeline


class DepthService:
    """Service for depth estimation from images."""

    def __init__(self):
        """Initialize depth estimation service."""
        # Default to Depth Anything V2 (best performance)
        self.model_name = os.getenv("DEPTH_MODEL", "depth_anything_v2")
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and os.getenv("ENABLE_GPU", "true").lower() == "true"
            else "cpu"
        )
        self.model = None
        self.transform = None
        self.pipe = None  # For Depth Anything V2 pipeline
        self.invert_depth_map = self._should_invert_depth()

        logger.info(f"Depth service initialized with model: {self.model_name}, device: {self.device}")

    async def load_model(self):
        """Load depth estimation model."""
        try:
            if "depth_anything" in self.model_name.lower():
                await self._load_depth_anything_v2()
            elif "midas" in self.model_name.lower():
                await self._load_midas()
            elif "zoe" in self.model_name.lower():
                await self._load_zoedepth()
            else:
                # Default to Depth Anything V2 if unknown
                logger.warning(f"Unknown depth model: {self.model_name}, defaulting to Depth Anything V2")
                await self._load_depth_anything_v2()

            logger.info(f"Depth model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Error loading depth model: {str(e)}")
            raise

    async def _load_depth_anything_v2(self):
        """Load Depth Anything V2 model (recommended - state-of-the-art)."""
        # Determine model size based on config
        # Options: small, base, large
        model_size = "small"  # Default to small for speed

        if "large" in self.model_name.lower():
            model_size = "large"
        elif "base" in self.model_name.lower():
            model_size = "base"

        # Depth Anything V2 model from Hugging Face
        model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"

        logger.info(f"Loading Depth Anything V2 ({model_size}) from Hugging Face...")

        # Create pipeline
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if self.device.type == "cuda" else -1
        )

        logger.info(f"Depth Anything V2 ({model_size}) loaded successfully")

    async def _load_midas(self):
        """Load MiDaS depth estimation model (legacy fallback)."""
        # Load MiDaS model from torch hub
        # Available models: DPT_Large, DPT_Hybrid, MiDaS_small
        model_type = "MiDaS_small"  # Default to small for speed

        if "large" in self.model_name.lower():
            model_type = "DPT_Large"
        elif "hybrid" in self.model_name.lower():
            model_type = "DPT_Hybrid"
        else:
            model_type = "MiDaS_small"

        logger.info(f"Loading MiDaS model: {model_type}")
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
        raise NotImplementedError("ZoeDepth support not yet implemented. Use Depth Anything V2 or MiDaS instead.")

    async def estimate_depth(self, image_base64: str) -> np.ndarray:
        """
        Estimate depth map from image.

        Args:
            image_base64: Base64-encoded image

        Returns:
            Depth map as numpy array (higher values = further away)
        """
        if self.model is None and self.pipe is None:
            raise RuntimeError("Depth model not loaded. Call load_model() first.")

        try:
            # Decode image
            img = self._decode_image(image_base64)

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Use Depth Anything V2 if available
            if self.pipe is not None:
                # Use Hugging Face pipeline off the event loop
                result = await asyncio.to_thread(self.pipe, img)
                depth_map = np.array(result["depth"]).astype(np.float32)

                logger.info(
                    f"Depth map estimated (Depth Anything V2): shape={depth_map.shape}, "
                    f"min={depth_map.min():.2f}, max={depth_map.max():.2f}"
                )

            else:
                # Use MiDaS (legacy)
                async def _predict_midas() -> np.ndarray:
                    img_np = np.array(img)
                    input_batch = self.transform(img_np).to(self.device)

                    def _forward():
                        with torch.no_grad():
                            prediction = self.model(input_batch)
                            prediction_resized = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=img_np.shape[:2],
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze()
                            return prediction_resized.cpu().numpy()

                    return await asyncio.to_thread(_forward)

                depth_map = np.array(await _predict_midas()).astype(np.float32)

                logger.info(
                    f"Depth map estimated (MiDaS): shape={depth_map.shape}, "
                    f"min={depth_map.min():.2f}, max={depth_map.max():.2f}"
                )

            depth_map = self._standardize_depth_map(depth_map)

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

        if x1_px > x2_px:
            x1_px, x2_px = x2_px, x1_px
        if y1_px > y2_px:
            y1_px, y2_px = y2_px, y1_px

        if (x2_px - x1_px) < 1 or (y2_px - y1_px) < 1:
            logger.warning("Degenerate bounding box after clamping; skipping depth extraction")
            return 0.0, 0.0

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
        # Depth map is normalized so lower values should correspond to closer regions.
        # Invert for visualization so warmer colors highlight nearer objects.
        visualization_map = np.clip(1.0 - depth_map, 0.0, 1.0)
        depth_normalized = (visualization_map * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

        # Convert to PIL Image
        depth_img = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))

        # Encode to base64
        buffered = BytesIO()
        depth_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    def _standardize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to [0, 1] and orient according to model configuration.
        """
        if not isinstance(depth_map, np.ndarray):
            depth_map = np.array(depth_map, dtype=np.float32)

        depth_map = depth_map.astype(np.float32)
        depth_map = np.nan_to_num(depth_map, nan=0.0)

        min_val = float(depth_map.min())
        max_val = float(depth_map.max())

        if max_val - min_val > 1e-6:
            normalized = (depth_map - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(depth_map, dtype=np.float32)

        if self.invert_depth_map:
            oriented = 1.0 - normalized
        else:
            oriented = normalized

        logger.debug(
            "Depth map normalized%s: min=%.2f, max=%.2f -> (0=close,1=far)",
            " and inverted" if self.invert_depth_map else "",
            oriented.min(),
            oriented.max()
        )

        return oriented

    def _should_invert_depth(self) -> bool:
        """
        Determine whether to invert the normalized depth map.

        Many legacy models (MiDaS/ZoeDepth) emit inverse depth (larger = closer),
        while Depth Anything emits true depth (larger = further). Allow override
        via DEPTH_ORIENTATION env var with values:
          - "auto" (default): invert for MiDaS/Zoe, keep raw for Depth Anything
          - "invert": always invert
          - "raw": never invert
        """
        orientation = os.getenv("DEPTH_ORIENTATION", "auto").strip().lower()
        if orientation == "invert":
            return True
        if orientation == "raw":
            return False

        # Auto-detect based on model name
        model_lower = self.model_name.lower()
        if "depth_anything" in model_lower:
            return False
        # Assume MiDaS/Zoe output inverse depth by default
        return True

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
