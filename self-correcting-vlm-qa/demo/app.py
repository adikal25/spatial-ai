"""
Streamlit demo application for Self-Correcting Vision-Language QA.
"""
import base64
import requests
from io import BytesIO

import streamlit as st
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="Self-Correcting VLM QA",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Self-Correcting Vision-Language QA")
st.markdown("""
This demo showcases an automated verification and self-correction pipeline for vision-language models.
The system operates in three stages:
1. **Ask**: VLM generates initial response with bounding boxes (1-3s)
2. **Verify**: Depth estimation + geometric contradiction detection (1-4s)
3. **Correct**: VLM self-corrects using evidence (1-4s)
""")

# API endpoint configuration
API_URL = st.text_input(
    "API Endpoint",
    value="http://localhost:8000",
    help="URL of the FastAPI backend"
)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("**Model:** Claude Sonnet (vision-enabled)")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system addresses VLM hallucinations in spatial reasoning by:
    - Using Claude Sonnet with a self-reasoning loop
    - Automatically verifying claims with depth geometry
    - Detecting contradictions in size, distance, and count
    - Claude Sonnet self-corrects with geometric evidence
    - Shows Claude Sonnet's self-reflection process
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Input")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image for spatial question answering"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Question input
        question = st.text_area(
            "Ask a spatial question about the image",
            value="Which object is closer to the camera?",
            height=100,
            help="Ask questions about object positions, sizes, distances, or counts"
        )

        # Submit button
        if st.button("🚀 Ask Question", type="primary"):
            with st.spinner("Processing... This may take up to 60 seconds (3-stage pipeline with depth analysis)"):
                try:
                    # Make API request with longer timeout for depth processing
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "image": img_base64,
                            "question": question,
                            "use_fallback": False
                        },
                        timeout=90  # Increased to 90 seconds for complex images
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["result"] = result
                        st.session_state["show_result"] = True
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to API. Make sure the backend is running.")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The API might be processing a large image.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.header("📥 Output")

    if st.session_state.get("show_result", False):
        result = st.session_state["result"]

        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Answers", "🔍 Analysis", "📊 Metrics", "⏱️ Performance"])

        with tab1:
            st.subheader("Initial Answer")
            st.info(result["answer"])

            if result.get("revised_answer"):
                st.subheader("Revised Answer (After Self-Correction)")
                st.success(result["revised_answer"])

                if result.get("self_reflection"):
                    with st.expander("🤔 Claude Sonnet's Self-Reflection"):
                        st.markdown(result["self_reflection"])

                st.metric(
                    "Confidence Score",
                    f"{result['confidence']:.1%}",
                    delta=None
                )
            else:
                st.success("No contradictions detected - initial answer verified!")

        with tab2:
            st.subheader("Geometric Analysis")

            # Show contradictions
            if result.get("contradictions"):
                st.warning(f"Found {len(result['contradictions'])} contradiction(s)")

                for i, contradiction in enumerate(result["contradictions"], 1):
                    with st.expander(f"Contradiction {i}: {contradiction['type'].upper()}"):
                        st.write(f"**Claim**: {contradiction['claim']}")
                        st.write(f"**Evidence**: {contradiction['evidence']}")
                        st.progress(contradiction["severity"], text=f"Severity: {contradiction['severity']:.1%}")
            else:
                st.success("No contradictions found!")

            # Show proof overlay
            if result.get("proof_overlay"):
                st.subheader("Proof Overlay (Image + Depth Map)")
                # Decode base64 image
                if "," in result["proof_overlay"]:
                    proof_base64 = result["proof_overlay"].split(",")[1]
                else:
                    proof_base64 = result["proof_overlay"]

                proof_bytes = base64.b64decode(proof_base64)
                proof_img = Image.open(BytesIO(proof_bytes))
                st.image(proof_img, use_column_width=True)

        with tab3:
            st.subheader("Spatial Metrics")

            if result.get("spatial_metrics"):
                for metric in result["spatial_metrics"]:
                    with st.expander(f"Object: {metric['object_id']}"):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.metric("Mean Depth", f"{metric['depth_mean']:.2f}")
                            st.metric("Depth Std Dev", f"{metric['depth_std']:.2f}")

                        with col_b:
                            if metric.get("estimated_distance"):
                                st.metric("Est. Distance", f"{metric['estimated_distance']:.2f}")

                            if metric.get("estimated_size"):
                                st.write(f"**Size**: {metric['estimated_size']['width']:.3f} × {metric['estimated_size']['height']:.3f}")
            else:
                st.info("No spatial metrics available")

        with tab4:
            st.subheader("Performance Metrics")

            latency = result.get("latency_ms", {})

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric(
                    "Ask Stage",
                    f"{latency.get('ask_ms', 0):.0f}ms",
                    help="VLM initial response time"
                )

            with col_b:
                st.metric(
                    "Verify Stage",
                    f"{latency.get('verify_ms', 0):.0f}ms",
                    help="Depth estimation and verification time"
                )

            with col_c:
                st.metric(
                    "Correct Stage",
                    f"{latency.get('correct_ms', 0):.0f}ms",
                    help="Self-correction time (0 if no contradictions)"
                )

            with col_d:
                total_ms = latency.get('total_ms', 0)
                # Determine color based on performance tier
                if total_ms < 4000:
                    delta_color = "off"
                    tier = "🥇 Gold"
                elif total_ms < 8000:
                    delta_color = "off"
                    tier = "🥈 Silver"
                else:
                    delta_color = "off"
                    tier = "🥉 Bronze"

                st.metric(
                    "Total Time",
                    f"{total_ms:.0f}ms",
                    delta=tier,
                    delta_color=delta_color
                )

            # Metadata
            if result.get("metadata"):
                st.subheader("Metadata")
                st.json(result["metadata"])

    else:
        st.info("👈 Upload an image and ask a question to get started!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        Built with Claude Sonnet, FastAPI, MiDaS Depth Estimation, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
