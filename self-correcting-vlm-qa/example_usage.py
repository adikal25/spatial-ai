"""
Example usage of the Self-Correcting VLM QA API.

This script demonstrates how to use the API programmatically.
"""
import base64
import requests
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ask_question(api_url: str, image_path: str, question: str, use_fallback: bool = False):
    """
    Ask a spatial question about an image.

    Args:
        api_url: Base URL of the API (e.g., "http://localhost:8000")
        image_path: Path to image file
        question: Spatial question to ask
        use_fallback: Whether to use fallback VLM (LLaVA)

    Returns:
        API response as dictionary
    """
    # Encode image
    image_base64 = encode_image(image_path)

    # Prepare request
    payload = {
        "image": image_base64,
        "question": question,
        "use_fallback": use_fallback
    }

    # Make request
    response = requests.post(
        f"{api_url}/ask",
        json=payload,
        timeout=30
    )

    # Check response
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
# example_usage.py (add near the bottom, before main())

def demo_fvdb_3d(depth_map: np.ndarray, bboxes: list):
    """Example: run fVDB 3D reconstruction on an existing depth map."""
    from src.services.fvdb_3d_service import Fvdb3DReconstructionService

    service = Fvdb3DReconstructionService()
    if not service.enabled:
        print("fVDB 3D reconstruction is disabled or fvdb is not installed.")
        return

    geom = service.compute_object_geometry(depth_map, bboxes)
    scene = geom["scene"]

    print(f"fVDB 3D: voxel_count={scene.voxel_count if scene else 0}")
    for i, c in enumerate(geom["centroids_3d"]):
        print(f"Object {i} centroid_3d:", None if c is None else c.tolist())
    print("Pairwise distances:", {str(k): v for k, v in geom["pairwise_distances"].items()})


def main():
    """Example usage."""
    # Configuration
    API_URL = "http://localhost:8000"
    IMAGE_PATH = "/Users/danielbashirov/Downloads/IMG_1051.jpg"  # Change this to your image path
    QUESTION = "Which object is closer to the camera?"

    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"Error: Image not found at {IMAGE_PATH}")
        print("Please update IMAGE_PATH in the script to point to a valid image.")
        return

    print(f"Asking question: '{QUESTION}'")
    print(f"About image: {IMAGE_PATH}")
    print("\nProcessing... (this may take up to 8 seconds)\n")

    try:
        # Ask question
        result = ask_question(API_URL, IMAGE_PATH, QUESTION)

        # Print results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        print(f"\nInitial Answer:")
        print(f"  {result['answer']}")

        if result.get("revised_answer"):
            print(f"\nRevised Answer (After Self-Correction):")
            print(f"  {result['revised_answer']}")
            print(f"\nConfidence: {result['confidence']:.1%}")
        else:
            print("\nNo contradictions detected - initial answer verified!")

        print(f"\nDetected Objects: {len(result.get('detected_objects', []))}")
        print(f"Contradictions Found: {len(result.get('contradictions', []))}")

        if result.get("contradictions"):
            print("\nContradictions:")
            for i, contradiction in enumerate(result["contradictions"], 1):
                print(f"\n  {i}. {contradiction['type'].upper()}")
                print(f"     Claim: {contradiction['claim']}")
                print(f"     Evidence: {contradiction['evidence']}")
                print(f"     Severity: {contradiction['severity']:.1%}")

        # Performance metrics
        latency = result.get("latency_ms", {})
        print(f"\nPerformance:")
        print(f"  Ask Stage:    {latency.get('ask_ms', 0):.0f}ms")
        print(f"  Verify Stage: {latency.get('verify_ms', 0):.0f}ms")
        print(f"  Correct Stage: {latency.get('correct_ms', 0):.0f}ms")
        print(f"  Total Time:   {latency.get('total_ms', 0):.0f}ms")

        print("\n" + "=" * 60)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API.")
        print(f"Make sure the API is running at {API_URL}")
        print("Start it with: python -m uvicorn src.api.main:app")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
