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

### Depth Mode (2D)
Uses monocular depth estimation to compute relative distances, occlusions, vertical alignment, and size cues. Multi-signal heuristics detect incorrect spatial claims.

### Voxel Mode (3D, experimental)
Uses NVIDIA FVDB to reconstruct a sparse voxel representation from the input image.  
This enables:

- 3D distances instead of pixel-depth  
- Accurate behind/in-front reasoning  
- Above/below in world coordinates  
- Containment and adjacency checks  
- Collision and free-space evaluation  

### Self-Reflection Loop
Claude receives:

- Original reasoning  
- Verified contradictions  
- Geometry evidence (depth map or voxel render)  

It then produces a corrected answer with a confidence score.

This dramatically reduces spatial hallucination rates.


## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User Input                            │
│                 (Image + Spatial Question)                   │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1 · ASK (Claude)                                      │
│  • Initial answer                                            │
│  • Reasoning trace                                           │
│  • Bounding boxes                                            │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2 · VERIFY (Depth or Voxel Geometry)                  │
│                                                              │
│    Depth Mode (2D):                                          │
│      • Depth Anything V2                                     │
│      • Occlusion checks                                      │
│      • Distance estimates                                    │
│      • Vertical position + size cues                         │
│                                                              │
│    Voxel Mode (3D, experimental):                            │
│      • NVIDIA FVDB sparse voxel reconstruction               │
│      • 3D occupancy grid                                     │
│      • 3D distances, adjacency, containment                  │
│                                                              │
│    • Contradiction detection                                 │
│    • Proof overlay generation (depth map or voxel render)    │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3 · SELF-CORRECT (Claude)                             │
│  • Review contradictions                                     │
│  • Generate corrected answer                                 │
│  • Provide reflection + confidence score                     │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
                Final Output (answer, metrics, overlays, latency)

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
├── Makefile                      # Common developer commands
├── setup.sh                      # Automated environment setup
├── run_demo.sh                   # Launch API + demo UI together
├── example_usage.py              # Python usage example (client)
├── config/
│   ├── .env.example              # Environment variable template
│   ├── depth_models/             # Depth Anything / MiDaS weights
│   └── voxel_models/             # FVDB voxel model configs
├── demo/
│   └── app.py                    # Streamlit front-end demo
├── src/
│   ├── api/main.py               # FastAPI application entrypoint
│   ├── models/schemas.py         # Pydantic request/response models
│   ├── services/
│   │   ├── vlm_service.py        # Claude ask + reflect pipeline
│   │   ├── depth_service.py      # 2D depth estimation routines
│   │   ├── voxel_service.py      # 3D voxel reconstruction (FVDB)
│   │   └── verifier_service.py   # Geometry checks + contradictions
│   └── utils/
│       ├── image_utils.py        # Image preprocessing helpers
│       └── voxel_utils.py        # Voxel grid utilities
└── tests/
    └── test_api.py               # API tests (unit + integration)

```

## Configuration

Environment variables (create `.env` in repo root from `config/.env.example`):

```env
# Required
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL=claude-sonnet-4-20250514

# Geometry mode: depth | voxel
GEOMETRY_MODE=depth

# Depth mode settings
DEPTH_MODEL=depth_anything_v2
DEPTH_ORIENTATION=auto
ENABLE_GPU=true

# Voxel mode settings (experimental)
VOXEL_BACKEND=fvdb
VOXEL_RESOLUTION=128
VOXEL_DOWNSCALE=2
ENABLE_VOXEL_PROOF=true

# Verification thresholds
RELATIVE_SIZE_THRESHOLD=0.3
RELATIVE_DISTANCE_THRESHOLD=0.3
DEPTH_CONFIDENCE_THRESHOLD=0.4

# API
CORS_ORIGINS=http://localhost:8501
MAX_IMAGE_SIZE=1024
```

To enable 3D mode:

```env
GEOMETRY_MODE=voxel
```

### POST /ask
Runs the full **Ask → Verify → Self-Correct** pipeline.

- Supports **2D Depth Mode**
- Supports **3D Voxel Mode (experimental)**

## API

### Endpoints
- `GET /health` – Returns status and loaded models
- `POST /ask` – Runs the Ask → Verify → Correct pipeline

### Example Response (Voxel Mode)

```json
{
  "geometry_mode": "voxel",
  "answer": "The box is in front of the chair.",
  "revised_answer": "The voxel grid shows the chair is actually in front of the box.",
  "self_reflection": "I misinterpreted the size cue; 3D occupancy reveals true ordering.",
  "confidence": 0.89,
  "proof_overlay_voxel": "data:image/png;base64,...",
  "voxel_metrics": {
    "distance_3d": 1.92,
    "occupancy_score": 0.81
  },
  "latency_ms": {
    "ask_ms": 2140,
    "verify_ms": 4020,
    "correct_ms": 1980,
    "total_ms": 8140
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

### Depth Mode (2D)
- Accessible scene descriptions  
- Construction site safety QA  
- Validation of AV/robotics datasets  
- Spatial reasoning benchmarks  
- Classroom demos on depth + reasoning  

### Voxel Mode (3D)
- True 3D spatial QA for robotics  
- Volumetric containment + collision reasoning  
- Planning and manipulation QA  
- Construction site 3D safety checks  
- Research on VLM spatial hallucinations in 3D  
- Indoor navigation and mapping QA  

---

## Limitations
- Depth-only mode cannot reason about true 3D thickness or containment  
- FVDB voxel reconstruction is approximate from a single image  
- Voxel mode requires a GPU for real-time inference  
- Bounding boxes still come from Claude  
- Test suite currently covers only API + schema validation  

---

## Extension Ideas

1. **Multi-view voxel fusion**  
   Combine multiple images to reconstruct a fused 3D scene.

2. **NeRF or 3D Gaussian Splatting backend**  
   Provide more accurate geometry than coarse voxels.

3. **Voxel-aware VLM tuning**  
   Train Claude or an open VLM on voxel-grounded spatial reasoning tasks.

4. **Richer 3D contradiction types**  
   - inside vs. intersect  
   - reachable vs. blocked  
   - above vs. below (world coordinates)  
   - behind/front adjacency reasoning  

5. **Robotics integration**  
   Use voxel-based checks for motion planning or grasping verification.

6. **Full async pipeline**  
   Background workers and queues for high-throughput, parallel processing.

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
