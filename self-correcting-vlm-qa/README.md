# 🔍 Self-Correcting Vision-Language QA with Claude

An automated verification and self-correction pipeline using **Claude Sonnet 4** that addresses spatial reasoning hallucinations through depth geometry and explicit self-reasoning loops.

## 🎯 Overview

Vision-Language Models (VLMs) often hallucinate about object sizes, distances, and counts, contradicting basic spatial geometry. This project implements a three-stage pipeline with **Claude's self-reasoning capabilities**:

1. **Ask** (1-3s): Claude generates initial response with bounding boxes and reasoning
2. **Verify** (1-4s): Depth estimation + geometric contradiction detection
3. **Correct** (1-4s): Claude engages in explicit self-reflection and correction

## ✨ Key Features

- **Claude-Powered**: Uses Claude Sonnet 4 with vision capabilities and tool use
- **Self-Reasoning Loop**: Claude explicitly reflects on its mistakes and corrects them
- **Multi-Signal Verification**: Uses Depth Anything V2 + occlusion + position cues for reliable validation
- **State-of-the-Art Depth**: Depth Anything V2 (2024) - significantly better than MiDaS
- **Transparent Reasoning**: See Claude's internal reasoning and self-reflection
- **Real-time Processing**: Target latency <8s end-to-end
- **Visual Proof**: Generates proof overlays with depth maps and annotations
- **REST API**: FastAPI backend for easy integration
- **Interactive Demo**: Streamlit web interface

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input                              │
│              (Image + Spatial Question)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: ASK                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Claude Sonnet 4 with Vision                          │   │
│  │ - Tool use for structured bounding boxes             │   │
│  │ - Initial spatial reasoning                          │   │
│  │ - Explicit reasoning trace                           │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ Answer + Reasoning + Bounding Boxes
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: VERIFY                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Depth Estimation (MiDaS)                             │   │
│  │ - Generate depth map                                 │   │
│  │ - Extract object depths                              │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Geometric Verifier                                   │   │
│  │ - Detect size contradictions                         │   │
│  │ - Detect distance contradictions                     │   │
│  │ - Detect count contradictions                        │   │
│  │ - Generate proof overlay                             │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ Contradictions + Proof Image
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: SELF-CORRECTION LOOP (if contradictions found)    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Claude Self-Reasoning Process                        │   │
│  │ 1. Review: Re-examine original image                 │   │
│  │ 2. Analyze: Study depth visualization                │   │
│  │ 3. Evaluate: Compare reasoning vs evidence           │   │
│  │ 4. Reflect: Identify errors made                     │   │
│  │ 5. Correct: Provide revised answer                   │   │
│  │ - Explicit self-reflection                           │   │
│  │ - Honest error acknowledgment                        │   │
│  │ - Confidence score                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Final Output                                   │
│  - Original Answer + Reasoning                              │
│  - Revised Answer (if corrected)                            │
│  - Self-Reflection                                          │
│  - Confidence Score                                         │
│  - Proof Overlay                                            │
│  - Spatial Metrics                                          │
│  - Performance Stats                                        │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- Python 3.11+
- **Anthropic API key** (for Claude Sonnet 4)
- (Optional) GPU for faster depth estimation

## 🚀 Quick Start

> **Want to deploy to the cloud?** See [DEPLOYMENT.md](DEPLOYMENT.md) for instructions.

### Simple Demo (Recommended)

1. **Clone/navigate to the project**
```bash
cd self-correcting-vlm-qa
```

2. **Run setup script**
```bash
./setup.sh
```

3. **Add your Anthropic API key**
```bash
# Edit config/.env and add your key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

4. **Run the demo!**
```bash
./run_demo.sh
```

The demo will open in your browser at http://localhost:8501

**That's it!** Upload an image and ask spatial questions.

### Manual Setup (Alternative)

If you prefer manual setup:

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set up environment**
```bash
cp config/.env.example config/.env
# Edit config/.env and add your Anthropic API key
```

3. **Run API (Terminal 1)**
```bash
python -m uvicorn src.api.main:app --reload
```

4. **Run Demo (Terminal 2)**
```bash
streamlit run demo/app.py
```

## 🔧 Configuration

Edit `config/.env` to customize:

```env
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Claude Configuration
CLAUDE_MODEL=claude-sonnet-4-20250514

# Depth Model Configuration
DEPTH_MODEL=depth_anything_v2
# Options:
#   depth_anything_v2 (recommended - state-of-the-art, default)
#   depth_anything_v2_base (higher accuracy, slower)
#   depth_anything_v2_large (best accuracy, slowest)
#   midas_v3_small (legacy, faster but less accurate)
#   midas_v3_dpt_large (legacy)

# Performance Settings
MAX_IMAGE_SIZE=1024
ENABLE_GPU=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Ask Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_data",
    "question": "Which object is closer to the camera?",
    "use_fallback": false
  }'
```

### Response Format

```json
{
  "answer": "The car is closer to the camera.",
  "revised_answer": "Actually, based on depth analysis, the tree is closer.",
  "confidence": 0.85,
  "proof_overlay": "data:image/png;base64,...",
  "detected_objects": [
    {
      "x1": 0.1,
      "y1": 0.2,
      "x2": 0.5,
      "y2": 0.8,
      "label": "car",
      "confidence": 0.9
    }
  ],
  "spatial_metrics": [
    {
      "object_id": "obj_0_car",
      "depth_mean": 45.2,
      "depth_std": 3.1,
      "estimated_distance": 4.52
    }
  ],
  "contradictions": [
    {
      "type": "distance",
      "claim": "Car is closer",
      "evidence": "Tree has lower depth value (32.1 vs 45.2)",
      "severity": 0.7
    }
  ],
  "latency_ms": {
    "ask_ms": 2100,
    "verify_ms": 1800,
    "correct_ms": 1500,
    "total_ms": 5400
  }
}
```

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/
```

## 📊 Performance Targets

| Metric | Gold | Silver | Bronze |
|--------|------|--------|--------|
| Total Latency | <4s | <8s | <12s |
| Accuracy Improvement | +35pp | +25pp | +15pp |
| Code Complexity | <800 LOC | <1200 LOC | <2000 LOC |

## 🏗️ Project Structure

```
self-correcting-vlm-qa/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI application
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vlm_service.py          # VLM interaction
│   │   ├── depth_service.py        # Depth estimation
│   │   ├── verifier_service.py     # Contradiction detection
│   │   └── correction_service.py   # Self-correction logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic models
│   └── utils/
│       └── __init__.py
├── demo/
│   └── app.py                      # Streamlit demo
├── tests/
│   └── __init__.py
├── config/
│   └── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## 🔍 How It Works

### 1. Initial Claude Query (Ask Stage)

The system queries **Claude Sonnet 4** with the user's spatial question and image. Claude responds with:
- Natural language answer
- Internal reasoning about spatial relationships
- Bounding boxes for detected objects (via tool use)

### 2. Geometric Verification (Verify Stage)

The verifier:
1. Uses MiDaS to generate a depth map
2. Extracts depth values for each bounding box
3. Computes spatial metrics (mean depth, size, estimated distance)
4. Compares metrics against VLM claims
5. Detects contradictions in:
   - **Relative distances**: "Object A is closer than B" vs depth values
   - **Relative sizes**: "Same size" vs bounding box areas
   - **Object counts**: "3 cars" vs detected objects
6. Generates proof overlay with side-by-side comparison

### 3. Self-Correction with Reasoning Loop (Correct Stage)

If contradictions are found, **Claude engages in explicit self-reasoning**:
1. Claude receives:
   - Original image
   - Depth visualization proof overlay
   - Its original answer and reasoning
   - Detailed contradictions with geometric evidence

2. Claude follows a structured self-reflection process:
   - **Review**: Re-examines the original image
   - **Analyze**: Studies the depth map visualization
   - **Evaluate**: Compares its reasoning against geometric measurements
   - **Reflect**: Explicitly identifies where it went wrong
   - **Correct**: Provides revised answer with honest error acknowledgment

3. Claude outputs:
   - Self-reflection explaining its thought process
   - Revised answer (or reaffirmation if evidence is inconclusive)
   - Confidence score (0-1)

## 🎨 Example Use Cases

We extend our AI Spatial Planner for Robots and Workers with a geometry-aware vision-language module that can verify and correct spatial reasoning from construction-site camera footage. Below are real operational scenarios where this unified spatial-intelligence system creates measurable business impact:


- **1. Real-Time Hazard Detection from Helmet Cameras**
Workers wearing IronSite-style smart safety helmets capture live video.  
The system identifies hazards—open trenches, heavy machinery swing-zones, blind-spot intersections—and verifies spatial claims using depth geometry.

**Business Value:**  
- Prevent injuries through earlier, more accurate hazard detection  
- Reduce blind spots for safety supervisors  
- Provide automatic documentation for OSHA compliance  


- **2. Safe Path Planning for Robots and Workers**
Users upload or sketch a site map.  
The system computes safe navigation paths, checking distances, obstacle proximity, and line-of-sight constraints.

**Business Value:**  
- Reduce on-site collisions between humans and robots  
- Optimize material transport routes for efficiency  
- Support automation initiatives by ensuring robot paths are reliably safe  


- **3. Spatial Verification for Incident Prevention**
If the VLM misjudges distance, height, or object placement, the geometric verifier measures the true scene (from depth or multi-view geometry), detects contradictions, and corrects the model.

**Business Value:**  
- Dramatically reduce false negatives (missed hazards)  
- Provide trustworthy AI reasoning for safety-critical decisions  
- Build an audit trail that improves insurance and risk-assessment outcomes  


- **4. Construction Workflow Optimization**
By analyzing spatial layouts, object positions, and worker movement, the system identifies congestion points, unsafe traffic patterns, and inefficient tool/equipment placement.

**Business Value:**  
- Reduce workflow delays  
- Improve equipment utilization  
- Provide data-driven layout planning  


- **5. Automated Compliance & Safety Reporting**
The system continuously verifies PPE detection, restricted-area entry, and distance thresholds around dangerous equipment.

**Business Value:**  
- Remove hours of manual site auditing  
- Increase compliance accuracy  
- Give supervisors immediate feedback with trustworthy evidence  


- **6. Training & Simulation for Spatial Intelligence**
Teams can upload site diagrams or simulated scenes to test and improve both **human** and **robot** spatial reasoning.

**Business Value:**  
- Train workers on hazard recognition  
- Test robots’ spatial navigation before deployment  
- Lower onboarding time for new hires  



## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Anthropic**: Claude Sonnet 4 with vision capabilities and self-reasoning
- **Depth Anything V2**: State-of-the-art monocular depth estimation (2024)
- **MiDaS**: Intel ISL for depth estimation (legacy support)
- **FastAPI**: Web framework
- **Streamlit**: Demo UI framework

## 📚 References

- [Depth Anything V2: Foundation Models for Monocular Depth Estimation](https://depth-anything-v2.github.io/)
- [MiDaS: Monocular Depth Estimation](https://github.com/isl-org/MiDaS)
- [Claude 4 by Anthropic](https://www.anthropic.com/claude)
- [Anthropic API Documentation](https://docs.anthropic.com/)

## 🐛 Known Limitations

- Depth estimation accuracy depends on monocular depth model limitations
- Contradiction detection uses heuristics; may miss complex cases
- Requires good lighting and clear object boundaries
- Performance depends on VLM API latency

## 🗺️ Roadmap

- [x] Support for Depth Anything V2 (state-of-the-art)
- [x] Multi-signal validation (depth + occlusion + position)
- [ ] Support for ZoeDepth
- [ ] Advanced semantic NLP for better contradiction detection
- [ ] Multi-turn conversation support
- [ ] Fine-tuned VLM for spatial reasoning
- [ ] Batch processing support
- [ ] Metrics dashboard
- [ ] A/B testing framework

---

**Built with ❤️ for accurate spatial reasoning using Claude's self-correction capabilities**
