# render_voxel_demo.py (place in self-correcting-vlm-qa/)
import asyncio
import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.services.depth_service import DepthService
from src.services.fvdb_3d_service import Fvdb3DReconstructionService
from src.services.vlm_service import VLMService


IMAGE_PATH = "/Users/danielbashirov/Downloads/IMG_1051.jpg"  # <- change this
QUESTION = "Which object is closer to the camera?"


async def main() -> None:
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 1) Get bounding boxes from VLM
    vlm = VLMService()
    vlm_result = await vlm.ask_with_boxes(
        image_base64=img_b64,
        question=QUESTION,
        use_fallback=False,
    )
    bboxes = vlm_result.get("bounding_boxes", [])
    if not bboxes:
        print("No bounding boxes returned by VLM.")
        return
    print(f"VLM detected {len(bboxes)} objects.")

    # 2) Compute depth map
    depth_service = DepthService()
    await depth_service.load_model()
    depth_map = await depth_service.estimate_depth(img_b64)

    # 3) Build fVDB scene + voxel mesh
    fvdb_service = Fvdb3DReconstructionService()
    if not fvdb_service.enabled:
        print("fVDB is disabled or not installed in this environment.")
        return

    geom = fvdb_service.compute_object_geometry(depth_map, bboxes)
    scene = geom.get("scene")
    if scene is None or scene.voxel_count == 0:
        print("No voxels in scene.")
        return

    mesh_info = fvdb_service.render_voxel_mesh(scene)
    if not mesh_info:
        print("Could not render voxel mesh.")
        return

    print("Voxel mesh saved to:", mesh_info["file_path"])
    print("Downsample factor:", mesh_info["downsample_factor"])
    print("Grid shape:", mesh_info["grid_shape"])
    print("Rendered voxels:", mesh_info["rendered_voxels"])


if __name__ == "__main__":
    asyncio.run(main())
