"""
Premium Streamlit UI for Self-Correcting Vision-Language QA.
A highly engaging, visually appealing interface with rich interactivity.
"""
import base64
import json
import time
from io import BytesIO
from typing import Dict, Any, Optional

import requests
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Spatial AI - Self-Correcting VLM QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Self-Correcting Vision-Language QA powered by Claude Sonnet 4"
    }
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #ec4899;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    /* Stage indicators */
    .stage-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Animation for processing */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .processing {
        animation: pulse 2s infinite;
    }
    
    /* Custom button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "result" not in st.session_state:
    st.session_state.result = None
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "processing_stage" not in st.session_state:
    st.session_state.processing_stage = None
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def check_api_health(api_url: str) -> tuple:
    """Check if API is healthy and return status."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def format_latency(ms: float) -> str:
    """Format latency in human-readable format."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.2f}s"

def get_performance_tier(total_ms: float) -> tuple:
    """Get performance tier based on total latency."""
    if total_ms < 4000:
        return "🥇 Gold", "success"
    elif total_ms < 8000:
        return "🥈 Silver", "normal"
    else:
        return "🥉 Bronze", "off"

def create_latency_chart(latency_data: Dict[str, float]) -> go.Figure:
    """Create a bar chart for latency breakdown."""
    stages = ["Ask", "Verify", "Correct"]
    values = [
        latency_data.get("ask_ms", 0),
        latency_data.get("verify_ms", 0),
        latency_data.get("correct_ms", 0)
    ]
    
    colors = ["#6366f1", "#8b5cf6", "#ec4899"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=stages,
            y=values,
            marker_color=colors,
            text=[f"{v:.0f}ms" for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.0f}ms<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Pipeline Stage Latency",
        xaxis_title="Stage",
        yaxis_title="Time (ms)",
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    return fig

def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a gauge chart for confidence score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#6366f1"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14)
    )
    
    return fig

def create_contradiction_severity_chart(contradictions: list) -> go.Figure:
    """Create a bar chart for contradiction severities."""
    if not contradictions:
        return None
    
    types = [c.get("type", "unknown").upper() for c in contradictions]
    severities = [c.get("severity", 0) * 100 for c in contradictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=types,
            y=severities,
            marker_color=["#ef4444" if s > 70 else "#f59e0b" if s > 50 else "#fbbf24" for s in severities],
            text=[f"{s:.1f}%" for s in severities],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Severity: %{y:.1f}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Contradiction Severity Analysis",
        xaxis_title="Contradiction Type",
        yaxis_title="Severity (%)",
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    return fig

# ============================================================================
# SIDEBAR - NAVIGATION & CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #6366f1; margin: 0;">🔍 Spatial AI</h1>
        <p style="color: #666; margin: 0.5rem 0;">Self-Correcting VLM QA</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Configuration
    st.subheader("⚙️ Configuration")
    
    api_url = st.text_input(
        "API Endpoint",
        value=st.session_state.api_url,
        help="URL of the FastAPI backend",
        key="api_url_input"
    )
    st.session_state.api_url = api_url
    
    # Health Check
    health_status, health_data = check_api_health(api_url)
    if health_status:
        st.success("✅ API Connected")
        if health_data:
            with st.expander("📊 API Status"):
                st.json(health_data)
    else:
        st.error("❌ API Not Available")
        st.info("Make sure the backend is running:\n```bash\npython -m uvicorn src.api.main:app --reload\n```")
    
    st.markdown("---")
    
    # Navigation
    st.subheader("🧭 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Main Analysis", "📚 History", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Examples
    st.subheader("💡 Example Questions")
    example_questions = [
        "Which object is closer to the camera?",
        "How many cars are in the image?",
        "Is the person taller than the building?",
        "What is the relative size of the two objects?",
        "Which object appears larger in the image?",
        "Are the objects at the same distance?",
    ]
    
    for i, example in enumerate(example_questions):
        if st.button(f"📌 {example}", key=f"example_{i}", use_container_width=True):
            st.session_state.example_question = example
            st.rerun()
    
    st.markdown("---")
    
    # About Section
    st.subheader("ℹ️ About")
    st.markdown("""
    **Powered by:**
    - Claude Sonnet 4
    - MiDaS Depth Estimation
    - FastAPI Backend
    
    **Features:**
    - 🧠 AI-powered spatial reasoning
    - 🔍 Automatic verification
    - ✨ Self-correction with reasoning
    - 📊 Detailed analytics
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with ❤️ for accurate spatial reasoning
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
if page == "🏠 Main Analysis":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Self-Correcting Vision-Language QA</h1>
        <p>Powered by Claude Sonnet 4 • Automated Verification & Self-Correction Pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Layout
    col_input, col_output = st.columns([1, 1], gap="large")
    
    # ========================================================================
    # INPUT COLUMN
    # ========================================================================
    with col_input:
        st.markdown("### 📤 Input")
        
        # Image Upload Section
        uploaded_file = st.file_uploader(
            "Upload an Image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload an image for spatial question answering. Supported formats: PNG, JPG, JPEG, WEBP",
            label_visibility="visible"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Image preview with info
            col_img, col_info = st.columns([2, 1])
            with col_img:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col_info:
                st.metric("Width", f"{image.width}px")
                st.metric("Height", f"{image.height}px")
                st.metric("Format", image.format or "Unknown")
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Question Input
            st.markdown("#### 💬 Ask a Spatial Question")
            
            # Pre-fill example question if selected
            default_question = st.session_state.get("example_question", "Which object is closer to the camera?")
            if "example_question" in st.session_state:
                del st.session_state.example_question
            
            question = st.text_area(
                "Question",
                value=default_question,
                height=120,
                help="Ask questions about object positions, sizes, distances, or counts in the image",
                placeholder="Example: Which object is closer to the camera?"
            )
            
            # Advanced Options
            with st.expander("⚙️ Advanced Options"):
                use_fallback = st.checkbox(
                    "Use Fallback Model",
                    value=False,
                    help="Use fallback VLM if primary model fails (not currently used)"
                )
                
                st.info("💡 Tip: The system automatically handles image optimization and model selection.")
            
            # Submit Button
            submit_button = st.button(
                "🚀 Analyze Image",
                type="primary",
                use_container_width=True,
                disabled=not health_status
            )
            
            if submit_button and question.strip():
                with st.spinner("🔄 Processing your request..."):
                    # Show processing stages
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Stage 1: Ask
                        status_text.info("📡 Stage 1/3: Asking Claude...")
                        progress_bar.progress(33)
                        
                        response = requests.post(
                            f"{api_url}/ask",
                            json={
                                "image": img_base64,
                                "question": question,
                                "use_fallback": use_fallback
                            },
                            timeout=90
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.result = result
                            st.session_state.show_result = True
                            
                            # Add to history
                            st.session_state.history.append({
                                "timestamp": time.time(),
                                "question": question,
                                "result": result
                            })
                            
                            progress_bar.progress(100)
                            status_text.success("✅ Analysis Complete!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.json(response.json() if response.content else {"error": "No response body"})
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Could not connect to API. Make sure the backend is running.")
                        st.info("Start the API with: `python -m uvicorn src.api.main:app --reload`")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. The image might be too large or the API is processing.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    # ========================================================================
    # OUTPUT COLUMN
    # ========================================================================
    with col_output:
        st.markdown("### 📥 Results")
        
        if st.session_state.get("show_result", False) and st.session_state.result:
            result = st.session_state.result
            
            # Quick Stats Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Objects Detected",
                    len(result.get("detected_objects", [])),
                    help="Number of objects detected by Claude"
                )
            with col2:
                contradictions_count = len(result.get("contradictions", []))
                st.metric(
                    "Contradictions",
                    contradictions_count,
                    delta="Found" if contradictions_count > 0 else "None",
                    delta_color="inverse" if contradictions_count > 0 else "off",
                    help="Number of geometric contradictions detected"
                )
            with col3:
                confidence = result.get("confidence", 0.0)
                st.metric(
                    "Confidence",
                    f"{confidence:.1%}",
                    help="Confidence score of the final answer"
                )
            
            # Main Results Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📝 Answers",
                "🔍 Analysis",
                "📊 Metrics",
                "⏱️ Performance",
                "🖼️ Visualizations"
            ])
            
            # ================================================================
            # TAB 1: ANSWERS
            # ================================================================
            with tab1:
                st.markdown("#### Initial Answer")
                st.info(result["answer"])
                
                # Original Reasoning (if available)
                if result.get("metadata", {}).get("original_reasoning"):
                    with st.expander("🧠 Claude's Original Reasoning"):
                        st.markdown(result["metadata"]["original_reasoning"])
                
                # Revised Answer Section
                if result.get("revised_answer"):
                    st.markdown("---")
                    st.markdown("#### ✨ Revised Answer (After Self-Correction)")
                    st.success(result["revised_answer"])
                    
                    # Self-Reflection
                    if result.get("self_reflection"):
                        st.markdown("---")
                        st.markdown("#### 🤔 Claude's Self-Reflection")
                        with st.container():
                            st.markdown(f"""
                            <div class="info-box">
                                {result['self_reflection']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Confidence Gauge
                    st.markdown("---")
                    st.markdown("#### 📊 Confidence Score")
                    confidence_gauge = create_confidence_gauge(confidence)
                    st.plotly_chart(confidence_gauge, use_container_width=True)
                else:
                    st.markdown("---")
                    st.success("✅ No contradictions detected - initial answer verified!")
                    st.balloons()
            
            # ================================================================
            # TAB 2: ANALYSIS
            # ================================================================
            with tab2:
                # Contradictions Section
                contradictions = result.get("contradictions", [])
                
                if contradictions:
                    st.markdown("#### ⚠️ Detected Contradictions")
                    st.warning(f"Found {len(contradictions)} contradiction(s) that triggered self-correction")
                    
                    # Contradiction Cards
                    for i, contradiction in enumerate(contradictions, 1):
                        with st.expander(f"🔴 Contradiction {i}: {contradiction['type'].upper()}", expanded=True):
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                st.markdown(f"**Claim:** {contradiction['claim']}")
                                st.markdown(f"**Evidence:** {contradiction['evidence']}")
                            
                            with col_b:
                                severity = contradiction.get("severity", 0)
                                st.progress(
                                    severity,
                                    text=f"Severity: {severity:.1%}"
                                )
                                
                                if severity > 0.7:
                                    st.error("High Severity")
                                elif severity > 0.5:
                                    st.warning("Medium Severity")
                                else:
                                    st.info("Low Severity")
                    
                    # Severity Chart
                    severity_chart = create_contradiction_severity_chart(contradictions)
                    if severity_chart:
                        st.plotly_chart(severity_chart, use_container_width=True)
                else:
                    st.success("✅ No contradictions found! Claude's initial answer was geometrically consistent.")
                
                # Proof Overlay
                st.markdown("---")
                st.markdown("#### 🖼️ Proof Overlay (Image + Depth Map)")
                
                if result.get("proof_overlay"):
                    # Decode base64 image
                    proof_overlay = result["proof_overlay"]
                    if "," in proof_overlay:
                        proof_base64 = proof_overlay.split(",")[1]
                    else:
                        proof_base64 = proof_overlay
                    
                    proof_bytes = base64.b64decode(proof_base64)
                    proof_img = Image.open(BytesIO(proof_bytes))
                    
                    st.image(proof_img, caption="Left: Original Image | Right: Depth Map (warmer colors = closer)", use_container_width=True)
                    
                    st.info("💡 The depth map shows relative distances. Warmer colors (red/yellow) indicate objects closer to the camera, while cooler colors (blue/purple) indicate objects further away.")
                else:
                    st.info("No proof overlay available")
            
            # ================================================================
            # TAB 3: METRICS
            # ================================================================
            with tab3:
                # Spatial Metrics
                spatial_metrics = result.get("spatial_metrics", [])
                
                if spatial_metrics:
                    st.markdown("#### 📐 Spatial Metrics")
                    st.info(f"Detailed geometric measurements for {len(spatial_metrics)} detected object(s)")
                    
                    for i, metric in enumerate(spatial_metrics):
                        with st.expander(f"📦 Object {i+1}: {metric.get('object_id', 'Unknown')}", expanded=False):
                            col_m1, col_m2, col_m3 = st.columns(3)
                            
                            with col_m1:
                                st.metric("Mean Depth", f"{metric.get('depth_mean', 0):.2f}")
                                st.metric("Depth Std Dev", f"{metric.get('depth_std', 0):.2f}")
                            
                            with col_m2:
                                if metric.get("estimated_distance"):
                                    st.metric("Est. Distance", f"{metric['estimated_distance']:.2f}")
                                if metric.get("estimated_size"):
                                    size = metric["estimated_size"]
                                    st.metric("Size (W×H)", f"{size.get('width', 0):.3f} × {size.get('height', 0):.3f}")
                            
                            with col_m3:
                                bbox = metric.get("bounding_box", {})
                                if bbox:
                                    st.markdown("**Bounding Box:**")
                                    st.code(f"x1: {bbox.get('x1', 0):.3f}\ny1: {bbox.get('y1', 0):.3f}\nx2: {bbox.get('x2', 0):.3f}\ny2: {bbox.get('y2', 0):.3f}")
                                    if bbox.get("label"):
                                        st.markdown(f"**Label:** {bbox['label']}")
                                    if bbox.get("confidence"):
                                        st.metric("Confidence", f"{bbox['confidence']:.1%}")
                else:
                    st.info("No spatial metrics available")
                
                # Detected Objects
                st.markdown("---")
                st.markdown("#### 🎯 Detected Objects")
                
                detected_objects = result.get("detected_objects", [])
                if detected_objects:
                    for i, obj in enumerate(detected_objects, 1):
                        st.markdown(f"**Object {i}:** {obj.get('label', 'Unknown')}")
                        st.json(obj)
                else:
                    st.info("No object details available")
            
            # ================================================================
            # TAB 4: PERFORMANCE
            # ================================================================
            with tab4:
                latency = result.get("latency_ms", {})
                
                # Performance Overview
                st.markdown("#### ⚡ Performance Overview")
                
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                with col_p1:
                    st.metric(
                        "Ask Stage",
                        format_latency(latency.get("ask_ms", 0)),
                        help="VLM initial response time"
                    )
                
                with col_p2:
                    st.metric(
                        "Verify Stage",
                        format_latency(latency.get("verify_ms", 0)),
                        help="Depth estimation and verification time"
                    )
                
                with col_p3:
                    correct_ms = latency.get("correct_ms", 0)
                    st.metric(
                        "Correct Stage",
                        format_latency(correct_ms),
                        delta="Skipped" if correct_ms == 0 else "Executed",
                        delta_color="off" if correct_ms == 0 else "normal",
                        help="Self-correction time (0 if no contradictions)"
                    )
                
                with col_p4:
                    total_ms = latency.get("total_ms", 0)
                    tier, tier_color = get_performance_tier(total_ms)
                    st.metric(
                        "Total Time",
                        format_latency(total_ms),
                        delta=tier,
                        delta_color=tier_color
                    )
                
                # Latency Chart
                st.markdown("---")
                latency_chart = create_latency_chart(latency)
                st.plotly_chart(latency_chart, use_container_width=True)
                
                # Metadata
                st.markdown("---")
                st.markdown("#### 📋 Metadata")
                
                if result.get("metadata"):
                    metadata = result["metadata"]
                    
                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        st.markdown(f"**Model Used:** {metadata.get('model_used', 'Unknown')}")
                        st.markdown(f"**Contradictions Found:** {metadata.get('contradictions_found', 0)}")
                    
                    with col_meta2:
                        if metadata.get("original_reasoning"):
                            with st.expander("View Original Reasoning"):
                                st.markdown(metadata["original_reasoning"])
                    
                    with st.expander("View Full Metadata"):
                        st.json(metadata)
            
            # ================================================================
            # TAB 5: VISUALIZATIONS
            # ================================================================
            with tab5:
                st.markdown("#### 📊 Data Visualizations")
                
                # Depth Distribution
                spatial_metrics = result.get("spatial_metrics", [])
                if spatial_metrics:
                    st.markdown("##### Depth Distribution")
                    depths = [m.get("depth_mean", 0) for m in spatial_metrics]
                    labels = [m.get("object_id", f"Object {i+1}") for i, m in enumerate(spatial_metrics)]
                    
                    if depths:
                        depth_fig = px.bar(
                            x=labels,
                            y=depths,
                            title="Mean Depth by Object",
                            labels={"x": "Object", "y": "Mean Depth"},
                            color=depths,
                            color_continuous_scale="viridis"
                        )
                        depth_fig.update_layout(
                            height=400,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(depth_fig, use_container_width=True)
                
                # Proof Overlay (duplicate from Analysis tab for convenience)
                if result.get("proof_overlay"):
                    st.markdown("---")
                    st.markdown("##### Proof Overlay")
                    proof_overlay = result["proof_overlay"]
                    if "," in proof_overlay:
                        proof_base64 = proof_overlay.split(",")[1]
                    else:
                        proof_base64 = proof_overlay
                    
                    proof_bytes = base64.b64decode(proof_base64)
                    proof_img = Image.open(BytesIO(proof_bytes))
                    st.image(proof_img, use_container_width=True)
        else:
            # Empty State
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #666;">
                <h2>👈 Upload an Image</h2>
                <p>Upload an image and ask a spatial question to get started!</p>
                <p style="font-size: 0.9rem; margin-top: 1rem;">
                    Try questions like:<br>
                    "Which object is closer to the camera?"<br>
                    "How many cars are in the image?"<br>
                    "Is the person taller than the building?"
                </p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📚 History":
    st.markdown("""
    <div class="main-header">
        <h1>📚 Analysis History</h1>
        <p>View your previous spatial reasoning analyses</p>
    </div>
    """, unsafe_allow_html=True)
    
    history = st.session_state.get("history", [])
    
    if history:
        st.info(f"📊 You have {len(history)} analysis(es) in your history")
        
        # Sort by timestamp (newest first)
        history_sorted = sorted(history, key=lambda x: x.get("timestamp", 0), reverse=True)
        
        for i, entry in enumerate(history_sorted):
            with st.expander(f"🔍 Analysis {i+1}: {entry.get('question', 'Unknown question')[:50]}...", expanded=False):
                col_h1, col_h2 = st.columns([2, 1])
                
                with col_h1:
                    st.markdown(f"**Question:** {entry.get('question', 'N/A')}")
                    st.markdown(f"**Answer:** {entry.get('result', {}).get('answer', 'N/A')}")
                    
                    if entry.get('result', {}).get('revised_answer'):
                        st.success(f"**Revised:** {entry['result']['revised_answer']}")
                
                with col_h2:
                    result = entry.get('result', {})
                    st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                    st.metric("Contradictions", len(result.get('contradictions', [])))
                    st.metric("Total Time", format_latency(result.get('latency_ms', {}).get('total_ms', 0)))
                
                if st.button(f"View Details", key=f"view_{i}"):
                    st.session_state.result = result
                    st.session_state.show_result = True
                    st.rerun()
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("📭 No analysis history yet. Start analyzing images to build your history!")

elif page == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About Spatial AI</h1>
        <p>Self-Correcting Vision-Language QA System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Overview
    
    This system addresses a critical limitation of Vision-Language Models (VLMs): 
    **spatial reasoning hallucinations**. VLMs often make incorrect claims about object 
    sizes, distances, and counts that contradict basic spatial geometry.
    
    ## 🔄 Three-Stage Pipeline
    
    ### 1️⃣ Ask Stage (1-3s)
    Claude Sonnet 4 analyzes the image and generates an initial answer with:
    - Natural language response
    - Bounding boxes for detected objects
    - Explicit reasoning about spatial relationships
    
    ### 2️⃣ Verify Stage (1-4s)
    Automated geometric verification:
    - **MiDaS depth estimation** generates a depth map
    - Spatial metrics are extracted (depth, size, distance)
    - Contradictions are detected by comparing VLM claims against geometric evidence
    - Proof overlay is generated showing side-by-side comparison
    
    ### 3️⃣ Correct Stage (1-4s)
    If contradictions are found, Claude engages in **explicit self-reasoning**:
    - Reviews the original image and depth visualization
    - Compares its reasoning against geometric evidence
    - Identifies errors and provides honest self-reflection
    - Generates a revised answer with confidence score
    
    ## ✨ Key Features
    
    - 🧠 **Claude-Powered**: Uses Claude Sonnet 4 with vision capabilities
    - 🔍 **Automated Verification**: Depth-based geometric contradiction detection
    - ✨ **Self-Correction**: Claude explicitly reflects on mistakes and corrects them
    - 📊 **Transparent**: See Claude's reasoning and self-reflection process
    - ⚡ **Fast**: Target latency <8s end-to-end
    - 🖼️ **Visual Proof**: Depth maps and annotated overlays
    
    ## 🏗️ Architecture
    
    - **Backend**: FastAPI with async processing
    - **VLM Service**: Claude Sonnet 4 integration
    - **Depth Service**: MiDaS depth estimation
    - **Verifier Service**: Geometric contradiction detection
    - **Correction Service**: Self-correction orchestration
    
    ## 📚 Use Cases
    
    - Autonomous vehicles: Verify object distance estimates
    - Robotics: Validate spatial reasoning for manipulation
    - Accessibility: Describe scene layouts accurately
    - Education: Teach spatial reasoning with feedback
    - Research: Study VLM spatial understanding limitations
    
    ## 🔧 Technical Details
    
    - **Python 3.11+**
    - **Claude Sonnet 4** (Anthropic API)
    - **MiDaS** depth estimation model
    - **FastAPI** web framework
    - **Streamlit** UI framework
    - **PyTorch** for depth model inference
    
    ## 📝 License
    
    MIT License - see LICENSE file for details
    
    ## 🙏 Acknowledgments
    
    - **Anthropic**: Claude Sonnet 4 with vision capabilities
    - **MiDaS**: Intel ISL for depth estimation
    - **FastAPI**: Web framework
    - **Streamlit**: UI framework
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Built with ❤️ for accurate spatial reasoning</strong></p>
        <p style="font-size: 0.9rem;">Using Claude's self-correction capabilities to improve VLM accuracy</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔍 Spatial AI - Self-Correcting Vision-Language QA</p>
    <p style="font-size: 0.85rem;">Powered by Claude Sonnet 4 • MiDaS Depth Estimation • FastAPI • Streamlit</p>
</div>
""", unsafe_allow_html=True)
