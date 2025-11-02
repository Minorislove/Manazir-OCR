"""
Professional Streamlit Frontend for Manazir OCR (Arabic optics-inspired Multi-Model OCR)

A modern, user-friendly interface for document OCR with model selection.
"""

import streamlit as st
from PIL import Image
import io
from pathlib import Path
import tempfile
import os
import sys

# Ensure project root is on sys.path so `import docustruct` works even when
# Streamlit changes the working directory.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from docustruct.model import (
    create_model,
    list_models,
    MODEL_REGISTRY,
    ModelTier,
)
from docustruct.input import load_pdf_images, load_image


# Page configuration
st.set_page_config(
    page_title="Manazir OCR - Multi-Model OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .model-name {
        font-weight: 600;
        color: #1f77b4;
        font-size: 1.1rem;
    }
    .model-desc {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        color: #0c5460;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📄 Manazir OCR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Model OCR Framework by Hesham Haroon</div>', unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.title("⚙️ Configuration")
nerd_theme = st.sidebar.checkbox("🧪 Nerd Theme", value=False, help="Monospace + neon dark UI")
if nerd_theme:
    st.markdown(
        """
        <style>
        :root { --neon:#39FF14; --bg:#0b0f14; }
        html, body, [class*=\"css\"]  { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
        .stApp { background: radial-gradient(circle at 20% 20%, #0e1520, var(--bg)); }
        h1, h2, h3 { color: var(--neon) !important; text-shadow: 0 0 8px rgba(57,255,20,.3); }
        .stButton>button { border:1px solid var(--neon); color: var(--neon); background: transparent; }
        .stButton>button:hover { box-shadow: 0 0 12px rgba(57,255,20,.4); }
        .metric-card { background: rgba(16,24,32,.6); border: 1px solid rgba(57,255,20,.25); }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Model tier filter
tier_options = {
    "All Models": None,
    "🏆 Primary Models": ModelTier.TIER_1_PRIMARY,
    "🎯 Specialized Models": ModelTier.TIER_2_SPECIALIZED,
    "⚡ Lightweight Models": ModelTier.TIER_3_LIGHTWEIGHT,
    "📊 Baseline Models": ModelTier.TIER_4_BASELINE,
    "💼 Commercial APIs": ModelTier.COMMERCIAL,
}

selected_tier_name = st.sidebar.selectbox(
    "Filter by Tier",
    options=list(tier_options.keys()),
    index=0
)
selected_tier = tier_options[selected_tier_name]

# Get filtered models
if selected_tier:
    available_models = list_models(tier=selected_tier)
else:
    available_models = list_models()

# Model selection
model_options = {
    f"{model.display_name} ({model.model_id})": model.model_id
    for model in available_models
}

if not model_options:
    st.sidebar.error("No models available for the selected tier.")
    st.stop()

selected_model_display = st.sidebar.selectbox(
    "Select OCR Model",
    options=list(model_options.keys()),
    index=0
)
selected_model_id = model_options[selected_model_display]

# Display model information
selected_model_config = MODEL_REGISTRY[selected_model_id]

with st.sidebar.expander("ℹ️ Model Information", expanded=True):
    st.markdown(f"**Name:** {selected_model_config.display_name}")
    st.markdown(f"**Languages:** {', '.join(selected_model_config.languages[:5])}")
    st.markdown(f"**License:** {selected_model_config.license}")
    st.markdown(f"**Commercial Use:** {'✅ Yes' if selected_model_config.commercial_use else '❌ No'}")
    st.markdown(f"**GPU Required:** {'✅ Yes' if selected_model_config.requires_gpu else '❌ No'}")
    st.markdown(f"**Description:** {selected_model_config.description}")

# Device selection
device = st.sidebar.radio(
    "Processing Device",
    options=["cuda", "cpu"],
    index=0 if selected_model_config.requires_gpu else 1,
    help="Select GPU (cuda) for faster processing or CPU for compatibility"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info(
    "Manazir OCR is a powerful multi-model OCR framework supporting 14+ models "
    "with a focus on Arabic and multilingual documents.\n\n"
    "Developed by **Hesham Haroon**"
)

# Main content area
tab1, tab2 = st.tabs(["📤 Upload & Process", "📋 Available Models"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose an image or PDF file",
            type=["png", "jpg", "jpeg", "pdf"],
            help="Upload a document for OCR processing"
        )
        
        if uploaded_file:
            # Display uploaded file
            if uploaded_file.type == "application/pdf":
                st.info(f"📄 PDF uploaded: {uploaded_file.name}")
                st.markdown("*First page will be processed*")
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner(f"Processing with {selected_model_config.display_name}..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = Path(tmp_file.name)
                        
                        # Load image(s)
                        if uploaded_file.type == "application/pdf":
                            images = load_pdf_images(tmp_path)
                            image_to_process = images[0] if images else None
                        else:
                            image_to_process = Image.open(uploaded_file)
                        
                        if image_to_process is None:
                            st.error("Failed to load image from file.")
                        else:
                            # Create model and process
                            model = create_model(selected_model_id, device=device)
                            result = model.process_image(image_to_process)
                            
                            # Store result in session state
                            st.session_state['result'] = result
                            st.session_state['processed'] = True
                            
                            st.success("✅ Processing complete!")
                    
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.markdown("### Results")
        
        if 'processed' in st.session_state and st.session_state['processed']:
            result = st.session_state['result']
            
            # Display metadata
            st.markdown(f"**Model Used:** {result.model_name}")
            st.markdown(f"**Confidence:** {result.confidence:.2%}")
            
            # Display extracted text
            st.markdown("### Extracted Text")
            st.text_area(
                "OCR Output",
                value=result.text,
                height=400,
                help="Extracted text from the document"
            )
            
            # Download button
            st.download_button(
                label="💾 Download as Text",
                data=result.text,
                file_name="ocr_result.txt",
                mime="text/plain"
            )
            
            # Display metadata
            if result.metadata:
                with st.expander("🔍 View Metadata"):
                    st.json(result.metadata)
        
        else:
            st.info("👈 Upload a document and click 'Process Document' to see results here.")

with tab2:
    st.markdown("### 📋 All Available Models")
    
    # Group models by tier
    tiers = {
        ModelTier.TIER_1_PRIMARY: "🏆 Tier 1: Primary Models",
        ModelTier.TIER_2_SPECIALIZED: "🎯 Tier 2: Specialized Models",
        ModelTier.TIER_3_LIGHTWEIGHT: "⚡ Tier 3: Lightweight Models",
        ModelTier.TIER_4_BASELINE: "📊 Tier 4: Baseline Models",
        ModelTier.COMMERCIAL: "💼 Commercial APIs",
    }
    
    for tier, tier_name in tiers.items():
        st.markdown(f"#### {tier_name}")
        
        tier_models = list_models(tier=tier)
        
        if not tier_models:
            st.info("No models in this tier.")
            continue
        
        for model in tier_models:
            with st.expander(f"{model.display_name} ({model.model_id})"):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"**Description:** {model.description}")
                    st.markdown(f"**Languages:** {', '.join(model.languages[:10])}")
                    st.markdown(f"**Strengths:** {', '.join(model.strengths)}")
                
                with col_b:
                    st.markdown(f"**License:** {model.license}")
                    st.markdown(f"**Commercial:** {'✅' if model.commercial_use else '❌'}")
                    st.markdown(f"**GPU:** {'✅' if model.requires_gpu else '❌'}")
                    st.markdown(f"**Size:** {model.model_size}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Manazir OCR v0.2.0 | Developed by <strong>Hesham Haroon</strong> | "
    "<a href='https://github.com/hesham-haroon/docustruct'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)


def main():
    """Entry point for the Streamlit app"""
    pass


if __name__ == "__main__":
    main()
