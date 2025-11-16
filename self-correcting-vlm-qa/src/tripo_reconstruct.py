"""
Thin wrapper around the TripoSG reconstruction model.
"""
from __future__ import annotations

import base64
import os
from typing import Optional, Tuple

import torch
import trimesh
from loguru import logger
from triposg import TripoSG


def _encode_png(data: bytes) -> str:
    """Encode raw PNG bytes as a data URL."""
    return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"


class TripoReconstructor:
    """Utility for loading TripoSG and generating mesh previews."""

    def __init__(self, device: str = "cuda"):
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        model_id = os.getenv("TRIPOSG_MODEL_ID", "VAST-AI/TripoSG")

        logger.info(f"Loading TripoSG ({model_id}) on {device}")
        model_kwargs = {}
        if hf_token:
            model_kwargs["use_auth_token"] = hf_token

        self.model = TripoSG.from_pretrained(
            model_id,
            **model_kwargs
        ).to(device)
        self.device = device
        self.model_id = model_id

    def reconstruct_mesh(self, image) -> trimesh.Trimesh:
        """
        Input:
            image: PIL Image or numpy array (H, W, 3)
        Output:
            mesh: trimesh.Trimesh object
        """

        with torch.no_grad():
            result = self.model(
                image=image,
                output_format="mesh",
                return_dict=True
            )

        mesh_path = result.get("mesh_path")
        if not mesh_path:
            raise RuntimeError("TripoSG did not return a mesh_path")

        mesh = trimesh.load(mesh_path)
        mesh.metadata = mesh.metadata or {}
        mesh.metadata.setdefault("tripo", {})
        mesh.metadata["tripo"].update({
            "mesh_path": mesh_path,
            "score": result.get("score"),
            "device": self.device,
            "model_id": self.model_id
        })

        return mesh

    def render_preview(
        self,
        mesh: trimesh.Trimesh,
        resolution: Tuple[int, int] = (512, 512)
    ) -> Optional[str]:
        """
        Render a quick preview of the reconstructed mesh.

        Returns:
            data URL string or None if preview generation fails.
        """
        try:
            scene = mesh.scene()
            png = scene.save_image(resolution=resolution, visible=False)
            if png:
                return _encode_png(png)
        except Exception as exc:
            logger.warning(f"Unable to render TripoSG preview: {exc}")
        return None
