# 🔍 Self-Correcting Vision-Language QA with Claude

This project implements automated verification and self-correction for spatial questions by combining Claude Sonnet 4 with geometric reasoning. It supports two geometry backends:

**1.** Depth-based 2D geometry (Depth Anything V2)

**2.** Experimental 3D voxel geometry (NVIDIA FVDB)

The system reduces spatial hallucinations by detecting contradictions using depth or voxel geometry, then prompting Claude to self-correct using explicit evidence.

## Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Why This Works](#why-this-works)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API](#api)
- [Use Cases](#use-cases)
- [Limitations](#limitations)
- [Extension Ideas](#extension-ideas)
- [Contributing](#contributing)
- [License](#license)

## Overview

Vision-Language Models can hallucinate about **relative distance, size, occlusion, and object counts**. This project implements a three-stage verification loop:

**1. Ask (≈1‑3 s):** Claude Sonnet 4 answers the question with reasoning and bounding boxes
**2. Verify (≈1‑4 s):**
   Depending on configuration, the system uses either:

   - **Depth Mode (default):** Depth Anything V2 depth estimation

   - **Voxel Mode (experimental):** NVIDIA FVDB sparse voxel reconstruction
This system supports a hybrid geometry pipeline: fast 2D depth verification for common spatial questions, with an optional 3D voxel refinement stage for complex containment,       occlusion, or world-coordinate reasoning. Combining depth maps with sparse voxel grids yields more robust spatial contradiction detection while keeping latency low.

Contradictions are detected using geometry signals (depth, occlusion, voxel occupancy, 3D distances, containment, etc.).

**3. Self-Correct (≈1‑4 s):** Claude self-reflects on contradictions and revises its answer

The backend is implemented with FastAPI, and an optional Streamlit UI displays depth maps, voxel renders, contradictions, and latency metrics.

## Getting Started

### Requirements
- Python **3.11+**
- **Anthropic API key** for Claude Sonnet 4
- (Optional) NVIDIA GPU for faster depth inference; CPU works but is slower
- Internet access on first run for downloading Hugging Face weights (~100 MB)

### Quick Setup

```bash
git clone https://github.com/yourusername/self-correcting-vlm-qa.git
cd self-correcting-vlm-qa
./setup.sh          # Creates venv, installs deps, copies .env template
# Edit .env with your ANTHROPIC_API_KEY
./run_demo.sh       # Boots FastAPI (port 8000) + Streamlit (port 8501)
```



### Manual Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp config/.env.example .env
# Edit .env to add ANTHROPIC_API_KEY

# Terminal 1: API
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit UI
streamlit run demo/app.py
```

**Additional helpers:**
- `make install`, `make run`, `make test` (see [Makefile](Makefile))
- `example_usage.py` - Python script example
- See [QUICKSTART.md](QUICKSTART.md) for detailed walkthrough
- See [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment

## Why This Works

- **Multi-signal verification:** Combines depth, occlusion, vertical position, and bounding box geometry. Contradictions require multiple signals to agree.
- **State-of-the-art depth:** Uses Depth Anything V2 (small/base/large variants) with auto depth orientation detection. Falls back to MiDaS for CPU-only deployments.
- **Transparent reasoning:** Claude provides explicit reasoning traces, and visual proof overlays show RGB + depth colormaps with bounding boxes.
- **Production-ready:** Handles image resizing, provides detailed latency metrics, and includes health checks.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User Input                               │
│              (Image + Spatial Question)                      │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1 · ASK (VLM)                                         │
│  • Claude Sonnet 4 produces answer + reasoning + boxes      │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2 · VERIFY (Depth Anything V2)                        │
│  • Generate depth maps                                       │
│  • Validate spatial claims with multi-signal checks          │
│  • Create proof overlay                                      │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3 · SELF-CORRECT                                      │
│  • Feed Claude the contradictions + proof overlay            │
│  • Enforce Review → Analyze → Reflect → Correct flow         │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
              Final payload (answers, metrics, proof, latency)
```

### Key Components

| Component | Implementation | Notes |
| --- | --- | --- |
| Vision-Language | Claude Sonnet 4 | Configurable via `CLAUDE_MODEL` |
| Depth estimation | Depth Anything V2 | GPU if `ENABLE_GPU=true`, otherwise CPU |
| Verification | Multi-signal (depth + occlusion + vertical + geometry) | Contradictions need multiple cues to agree |
| UI | Streamlit + FastAPI | Visualizes all stages with latency tracking |

### Repository Structure

```
self-correcting-vlm-qa/
├── README.md                     # You are here
├── QUICKSTART.md                 # Step-by-step setup guide
├── DEPLOYMENT.md                 # Cloud deployment instructions
├── requirements.txt              # Python dependencies
├── Makefile                      # Common commands
├── setup.sh                      # Automated setup script
├── run_demo.sh                   # Start API + Streamlit together
├── example_usage.py              # Python client example
├── config/
│   └── .env.example              # Environment variable template
├── demo/
│   └── app.py                    # Streamlit front-end
├── src/
│   ├── api/main.py               # FastAPI application
│   ├── models/schemas.py         # Pydantic models
│   ├── services/
│   │   ├── vlm_service.py        # Claude interaction
│   │   ├── depth_service.py      # Depth Anything V2 / MiDaS
│   │   └── verifier_service.py   # Claim validation + overlays
│   └── utils/image_utils.py      # Image processing utilities
└── tests/
    └── test_api.py               # API tests
```

## Configuration

Environment variables (create `.env` in repo root from `config/.env.example`):

```env
# Required
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL=claude-sonnet-4-20250514

# Depth models
DEPTH_MODEL=depth_anything_v2         # append _base or _large if desired
DEPTH_ORIENTATION=auto                # auto | invert | raw
ENABLE_GPU=true                       # set false for CPU-only

# Verification thresholds
RELATIVE_SIZE_THRESHOLD=0.3
RELATIVE_DISTANCE_THRESHOLD=0.3
DEPTH_CONFIDENCE_THRESHOLD=0.4

# API settings
CORS_ORIGINS=http://localhost:8501,http://localhost:3000
MAX_IMAGE_SIZE=1024
```

## API

### Endpoints
- `GET /health` – Returns status and loaded models
- `POST /ask` – Runs the Ask → Verify → Correct pipeline

### Example Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAAA...",  # base64 encoded
    "question": "Which object is closer to the camera?"
  }'
```

### Example Response

```json
{
  "answer": "The bus on the left is closest to the camera.",
  "revised_answer": "After reviewing the depth overlay, the cyclist in the foreground is actually closest.",
  "self_reflection": "I originally focused on size cues and ignored occlusion. The depth map shows the cyclist is nearer.",
  "confidence": 0.82,
  "proof_overlay": "data:image/png;base64,...",
  "detected_objects": [
    {"x1": 0.05, "y1": 0.32, "x2": 0.32, "y2": 0.88, "object_id": "obj_0_bicycle", "label": "cyclist"}
  ],
  "spatial_metrics": [
    {
      "object_id": "obj_0_bicycle",
      "depth_mean": 18.4,
      "depth_std": 2.1,
      "estimated_distance": 18.4,
      "estimated_size": {"width": 0.27, "height": 0.56}
    }
  ],
  "contradictions": [
    {
      "type": "distance",
      "claim": "The bus is closest to the camera.",
      "evidence": "Cyclist depth 18.4 vs bus depth 41.2; occlusion shows cyclist in front.",
      "severity": 0.72
    }
  ],
  "latency_ms": {"ask_ms": 2350, "verify_ms": 3010, "correct_ms": 1875, "total_ms": 7235},
  "metadata": {
    "model_used": "claude-sonnet-4-20250514",
    "contradictions_found": 1,
    "original_reasoning": "I judged by object size..."
  }
}
```

**Key fields:**
- `detected_objects` - Normalized coordinates (0‑1)
- `spatial_metrics.estimated_distance` - Relative depth (0‑100, no absolute scale)
- `proof_overlay` - PNG with original image + depth colormap
- `self_reflection` - Claude's reasoning about corrections

See [example_usage.py](example_usage.py) for programmatic usage.

## Use Cases

- **Autonomous perception QA:** Validate spatial statements in AV/robotics datasets
- **Robotics manipulation:** Verify "which object is reachable/closer" before executing commands
- **Accessibility tools:** Provide reliable scene descriptions with uncertainty indicators
- **Educational demos:** Show how reasoning, geometry, and self-correction interact
- **Research instrumentation:** Log contradiction rates to study VLM spatial hallucinations

## Limitations

1. **Depth coverage** - Depth Anything V2 struggles with reflective/textureless regions
2. **Bounding boxes** - Boxes come from Claude's output; missed objects can't be verified
3. **Heuristic contradictions** - Simple threshold-based checks may miss nuanced spatial logic
4. **Latency** - First request downloads ~100 MB of weights; CPU mode can take >10s
5. **Limited tests** - Test suite only covers health + validation; no depth/contradiction regression fixtures


## Extension Ideas

1. **Multi-VLM ensemble** - Wire `use_fallback` to open-source models (LLaVA, GPT-4o) or use majority voting
2. **Enhanced depth** - Implement ZoeDepth or multi-view depth reconstruction
3. **Richer contradictions** - Add orientation, containment, metric reasoning ("object is 2m behind line")
4. **Automated evaluation** - Create benchmark datasets to measure contradiction detection accuracy
5. **Async processing** - Add queue/background workers for concurrent `/ask` requests
6. **Metrics dashboard** - Persist stats to database with Prometheus/Grafana monitoring

## Future Directions

### 3D Voxel-Based Spatial Reasoning

Move beyond 2D depth maps to true 3D scene understanding using **NVIDIA FVDB** (Fast Voxel Database):

**Architecture:**
1. **3D Scene Reconstruction** - Integrate NVIDIA FVDB to convert monocular images into sparse voxel representations, creating a volumetric 3D scene graph
2. **Voxel-Level Verification** - Replace 2D bounding box verification with 3D occupancy grids, enabling accurate spatial relationship reasoning (containment, occlusion, relative positions)
3. **Fine-tuned Vision Model** - Train a specialized vision model on carefully labeled 3D spatial datasets with ground-truth voxel annotations
4. **Enhanced Self-Correction Loop** - Feed voxel-level contradictions back to the VLM with 3D proof visualizations (volumetric renderings, cross-sections)

**Benefits:**
- True 3D spatial understanding vs. pseudo-3D depth estimation
- Handle complex occlusions and multi-object spatial relationships
- Support volumetric queries ("Is object A inside object B?", "What's between A and B?")
- Enable robotic path planning and manipulation tasks with precise 3D coordinates

**Implementation Path:**
- Replace `DepthService` with `VoxelService` using NVIDIA FVDB + occupancy networks
- Create curated dataset of images with ground-truth 3D voxel labels
- Fine-tune vision model (e.g., ViT, CLIP) on voxel prediction task
- Extend `VerifierService` to validate claims against 3D voxel grids
- Add 3D visualization overlay (volumetric rendering or interactive 3D viewer)

This would transform the system from depth-augmented 2D reasoning to full 3D geometric understanding.

## Contributing

Pull requests are welcome! Please:

1. Fork the repo and create a feature branch
2. Run `pytest` and add tests for new features
3. Keep docs in sync with code changes
4. Include screenshots/logs for UI or pipeline changes

## License

MIT License - see [LICENSE](LICENSE)

---

**Built with:**
- [Anthropic Claude Sonnet 4](https://docs.anthropic.com/) - Vision + reasoning
- [Depth Anything V2](https://depth-anything-v2.github.io/) - Monocular depth estimation
- [MiDaS](https://github.com/isl-org/MiDaS) - Depth estimation fallback
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Streamlit](https://streamlit.io/) - UI framework
- PyTorch, Transformers, Pydantic
