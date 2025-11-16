# 🔍 Spatial AI for Construction Intelligence
## Self-Correcting Vision-Language QA | IronSite x Vanderbilt Hackathon

---

# 🎯 The Problem

**IronSite's Vision:**

Construction workers wear smart safety helmets streaming continuous video from the jobsite. IronSite wants to automatically understand:

- **What tasks** are workers performing?
- **Where** are those tasks happening?
- **How long** does each activity take?

By mapping worker actions to locations and timelines, construction managers get a **real-time, measurable view of productivity**—enabling better planning, coordination, bottleneck detection, and resource allocation.

**But there's a critical problem:**

Current AI systems **hallucinate about spatial relationships**. They make incorrect claims about distances, sizes, and object counts.

**Real Example:**
- AI says: *"The worker is 5 feet from the equipment"*
- Reality: They're actually 12 feet away
- **Impact:** Wrong safety zones, incorrect productivity metrics

**For safety-critical construction, these errors are unacceptable.**

---

# 💡 Our Solution

We built the **first system that combines AI reasoning with geometric verification and explicit self-correction**.

**What makes us different:**
- ✅ **Depth-based verification** - Validates every spatial claim with geometric proof
- ✅ **AI self-correction** - Claude reviews its own mistakes and corrects them
- ✅ **Transparent reasoning** - Full trace of decision-making
- ✅ **Production-ready** - <8s latency, REST API, ready to integrate

**Result:** Spatial intelligence you can trust for real-time helmet video processing.

---

# 🚀 How It Works: The Three-Stage Pipeline

## Stage 1: ASK (1-3 seconds)
**Claude Sonnet 4 analyzes the image**

- Understands the scene: workers, equipment, tools
- Detects objects with precise bounding boxes
- Answers your spatial question with reasoning

**Example:**
- Question: *"Is there enough clearance for this ladder?"*
- Claude detects: ladder, worker, nearby wall
- Answers: *"Yes, approximately 3 feet of clearance"*

## Stage 2: VERIFY (1-4 seconds)
**MiDaS depth estimation validates the answer**

- Generates depth map showing 3D spatial relationships
- Extracts depth values for each detected object
- Compares Claude's claims against geometric measurements
- Detects contradictions automatically

**What gets verified:**
- Distance claims → Checked against depth values
- Size claims → Checked against bounding box areas
- Count claims → Checked against detected objects

**Output:** Spatial metrics + contradictions + visual proof

## Stage 3: CORRECT (1-4 seconds, only if needed)
**Claude self-reflects and corrects errors**

When contradictions are found:
1. Reviews the original image
2. Studies the depth map
3. Compares reasoning vs evidence
4. Identifies the error
5. Provides corrected answer with confidence

**Total Time: <8 seconds per frame**

---

# 🎬 Live Demo Flow

**Step 1:** Upload a frame from helmet video

**Step 2:** Ask a spatial question:
- *"Is there enough clearance for this ladder?"*
- *"Is this person in a fall hazard zone?"*
- *"Which object is closest to the worker?"*

**Step 3:** Get verified results in <8 seconds:
- ✅ Initial answer with reasoning
- ✅ Detected contradictions (if any)
- ✅ Corrected answer with self-reflection
- ✅ Confidence score
- ✅ Visual proof (image + depth map)
- ✅ Spatial metrics (exact distances, sizes)

---

# 🏗️ How This Powers IronSite's Vision

## What's Working Now ✅

**Frame-by-Frame Processing:**
- Processes individual frames from helmet video streams
- Object detection (workers, equipment, tools)
- Spatial analysis (distances, sizes, positions)
- Verified accuracy (depth-based validation)

**Integration:**
```
Helmet Camera → Video Stream → Extract Frame → Our API
                                          ↓
                                    Spatial Analysis
                                          ↓
                              Task Context + Location + Confidence
```

**Enables:**
- **Location Awareness:** Depth maps + bounding boxes show exactly where workers are
- **Task Understanding:** Spatial relationships indicate what workers are doing
- **Safety Verification:** Clearance and proximity checks with geometric proof
- **Quantitative Metrics:** Exact distances and sizes for productivity analysis

## Real-World Example

**Scenario:** Worker performing ladder work

1. Frame captured from helmet camera
2. Question: *"Is there enough clearance for this ladder?"*
3. Our system:
   - Detects: ladder, worker, nearby wall
   - Claude initially: *"Yes, about 2 feet clearance"*
   - Depth analysis: *Actually only 1.2 feet*
   - **Contradiction detected!**
   - Claude self-corrects: *"I was wrong. The depth map shows only 1.2 feet clearance, which may be insufficient."*
   - Confidence: 0.92

4. IronSite receives:
   - Task: Ladder work
   - Location: Coordinates from spatial metrics
   - Safety: Insufficient clearance detected
   - Confidence: High (0.92)

5. Manager sees: Real-time alert + location on jobsite map

---

# 🔬 Technical Innovation

## Innovation #1: Explicit Self-Reasoning Loop

**First system to make AI review its own spatial reasoning errors.**

Claude doesn't just get corrected—it **explicitly reflects** on mistakes:
- *"I initially thought 2 feet clearance, but the depth map shows only 1.2 feet. I was wrong."*

This transparency builds trust in safety-critical applications.

## Innovation #2: Depth-Geometry Verification

**Combines AI reasoning with geometric validation.**

We verify every claim:
- Depth maps provide ground truth
- Automated contradiction detection
- Visual proof overlays

## Innovation #3: Production-Ready Architecture

**Built for real-world deployment:**
- Async FastAPI backend
- Modular service design
- REST API for integration
- <8s latency

---

# 📊 System Architecture

```
Helmet Video Stream
    ↓
Frame Extraction
    ↓
FastAPI Backend (POST /ask)
    ↓
┌─────────────────────────────────┐
│  Stage 1: ASK                   │
│  Claude Sonnet 4                │
│  → Answer + Bounding Boxes      │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  Stage 2: VERIFY                 │
│  MiDaS Depth Estimation         │
│  → Depth Map + Metrics          │
│                                 │
│  Geometric Verifier             │
│  → Contradiction Detection      │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  Stage 3: CORRECT (if needed)    │
│  Claude Self-Correction         │
│  → Revised Answer + Reflection  │
└────────────┬────────────────────┘
             ↓
JSON Response with Verified Results
```

**Key Components:**
- **VLMService** - Claude integration
- **DepthService** - MiDaS depth estimation
- **VerifierService** - Contradiction detection
- **CorrectionService** - Self-correction

---

# ✨ Features & Capabilities

## What's Working Now ✅

**Spatial Question Answering:**
- Ask questions about distances, sizes, positions, counts
- Verified with depth-based geometric analysis
- Self-corrected when contradictions detected

**Object Detection & Localization:**
- Precise bounding boxes for all objects
- Normalized coordinates (works with any frame size)
- Object labels and confidence scores

**Depth-Based Spatial Metrics:**
- Mean depth per object (relative distance)
- Depth standard deviation
- Estimated distance
- Object size (width × height)

**Contradiction Detection:**
- Distance contradictions (20% threshold)
- Size contradictions (30% threshold)
- Count contradictions

**Self-Correction:**
- Explicit self-reflection on errors
- Revised answers with confidence scores
- Honest error acknowledgment

**Visual Proof:**
- Side-by-side: original image + depth map
- Annotated bounding boxes with depth values
- Color-coded depth (warmer = closer)

## How This Supports IronSite

**Task Identification:**
- Object detection identifies workers, equipment, tools
- Spatial relationships indicate task context
- **Current:** Single frame analysis
- **Future:** Multi-frame correlation

**Location Awareness:**
- Depth maps provide 3D spatial understanding
- Bounding boxes give 2D coordinates
- Spatial metrics quantify exact positions
- **Current:** Per-frame location
- **Future:** Temporal tracking

**Duration Tracking:**
- Each frame processed in <8s
- Can process ~7-12 frames per minute
- **Current:** Per-frame processing
- **Future:** Frame-to-frame correlation

**Safety Applications:**
- Clearance detection
- Hazard zone detection
- Proximity warnings
- **Current:** Real-time per-frame analysis

---

# 🎯 Impact: Why This Matters

## For Construction Safety

**Before:** AI makes spatial errors leading to incorrect safety zones and unreliable hazard detection.

**After:** Verified spatial reasoning ensures accurate safety zone detection, reliable measurements, and trustworthy hazard identification with confidence scores.

## For Productivity Measurement

**Before:** No reliable way to automatically understand what tasks workers perform and where.

**After:** Spatial intelligence provides location-aware task understanding, quantitative metrics, and foundation for duration tracking.

## For IronSite's Vision

**Enables:**
- Real-time task identification from helmet video
- Location mapping for productivity analysis
- Spatial metrics for operational efficiency
- Trustworthy automation for safety-critical applications

---

# 🔮 Roadmap: From MVP to Full Deployment

## What's Working Now ✅

- Single frame processing from video streams
- Spatial question answering with verification
- Depth-based geometric validation
- Self-correction for accuracy
- REST API for integration
- Real-time processing (<8s per frame)

## Next Steps

### Phase 1: Video Stream Processing
- Frame extraction from continuous streams
- Batch processing for multiple frames
- **Enables:** Continuous monitoring

### Phase 2: Temporal Tracking
- Object tracking across frames
- Worker movement trajectory analysis
- **Enables:** Duration tracking

### Phase 3: Task Classification
- Activity recognition from spatial patterns
- Multi-frame analysis
- **Enables:** Automatic task identification

### Phase 4: Full Integration
- Direct helmet camera stream ingestion
- Real-time dashboard
- Historical data analysis
- **Enables:** Complete productivity system

---

# 🚀 Quick Start

## Installation

```bash
cd self-correcting-vlm-qa
./setup.sh
```

## Configuration

Edit `config/.env`:
```env
ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
# Terminal 1 - Backend
python -m uvicorn src.api.main:app --reload

# Terminal 2 - UI
streamlit run demo/app.py
```

Open: http://localhost:8501

## Try It

1. Upload a construction site image
2. Ask: *"Is there enough clearance for this ladder?"*
3. See verified results in <8 seconds

---

# 📈 Performance

- **Total Latency:** <8 seconds per frame
- **Ask Stage:** 1-3 seconds
- **Verify Stage:** 1-4 seconds
- **Correct Stage:** 1-4 seconds (if needed)

**For 30 FPS video:** Process every 15th frame for real-time analysis

---

# 🏛️ Technical Stack

- **Claude Sonnet 4** - Vision understanding & reasoning
- **MiDaS Depth Estimation** - Geometric verification
- **FastAPI** - Async web framework
- **Streamlit** - Interactive UI
- **Python 3.11+** - Core language

---

# 💻 Codebase

**Total:** ~1,941 lines of production Python

**Key Files:**
- `src/services/vlm_service.py` (432 lines) - Claude integration
- `src/services/depth_service.py` (212 lines) - Depth estimation
- `src/services/verifier_service.py` (362 lines) - Contradiction detection
- `src/api/main.py` (200 lines) - FastAPI application
- `demo/app.py` (257 lines) - Streamlit UI

---

# 🎓 Why This Wins

**Innovation:**
- First system combining depth geometry with AI self-reasoning
- Novel approach to eliminating spatial hallucinations
- Transparent error correction

**Impact:**
- Solves critical problem in safety-critical applications
- Enables IronSite's vision for spatial intelligence
- Production-ready with <8s latency

**Execution:**
- Clean, modular architecture
- Comprehensive error handling
- REST API for easy integration

**Applicability:**
- Directly addresses IronSite's use case
- Processes video frames from helmet cameras
- Provides spatial metrics for productivity measurement

---

# 🔗 Integration Example

```
Helmet Camera → Video Stream
    ↓
Extract Frame
    ↓
POST /ask {
  image: frame_base64,
  question: "What task is the worker performing?"
}
    ↓
Our System:
  - Detects objects
  - Estimates spatial relationships
  - Verifies with depth geometry
  - Returns: task + location + confidence
    ↓
IronSite System:
  - Maps task to location
  - Tracks duration
  - Updates productivity dashboard
```

---

# 📝 License

MIT License

---

# 🙏 Acknowledgments

- **IronSite** - For the inspiring vision
- **Anthropic** - Claude Sonnet 4
- **MiDaS** - Depth estimation
- **FastAPI & Streamlit** - Web frameworks

---

**Built for the IronSite x Vanderbilt Hackathon**  
**Enabling trustworthy spatial intelligence for construction** 🚀
