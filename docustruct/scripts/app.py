import pypdfium2 as pdfium
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import re

from docustruct.model import InferenceManager
from docustruct.model import create_model, MODEL_REGISTRY
from docustruct.util import draw_layout
from docustruct.input import load_pdf_images
from docustruct.model.schema import BatchInputItem
from docustruct.output import parse_layout


@st.cache_resource()
def load_model(method: str):
    return InferenceManager(method=method)


@st.cache_data()
def get_page_image(pdf_file, page_num):
    return load_pdf_images(pdf_file, [page_num])[0]


@st.cache_data()
def page_counter(pdf_file):
    doc = pdfium.PdfDocument(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def embed_images_in_markdown(markdown: str, images: dict) -> str:
    """Replace image filenames in markdown with base64 data URLs."""
    for img_name, pil_image in images.items():
        # Convert PIL image to base64 data URL
        data_url = pil_image_to_base64(pil_image, format="PNG")
        # Replace the image reference in markdown
        # Pattern matches: ![...](img_name) or ![...](img_name "title")
        pattern = rf'(!\[.*?\])\({re.escape(img_name)}(?:\s+"[^"]*")?\)'
        markdown = re.sub(pattern, rf"\1({data_url})", markdown)
    return markdown


def ocr_layout(
    img: Image.Image,
    model=None,
) -> (Image.Image, str):
    batch = BatchInputItem(
        image=img,
        prompt_type="ocr_layout",
    )
    result = model.generate([batch])[0]
    layout = parse_layout(result.raw, img)
    layout_image = draw_layout(img, layout)
    return result, layout_image


st.set_page_config(layout="wide", page_title="Manazir OCR Demo")

# Nerd theme CSS
st.markdown(
    """
    <style>
      :root { --neon:#39FF14; --bg:#0b0f14; --muted:#94a3b8; }
      html, body, [class*="css"]  { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .stApp { background: radial-gradient(circle at 20% 20%, #0e1520, var(--bg)); }
      h1, h2, h3, .stMarkdown h1 { color: var(--neon) !important; text-shadow: 0 0 8px rgba(57,255,20,.3); }
      .nerd-panel { border: 1px solid rgba(57,255,20,.3); border-radius: 8px; padding: 12px; background: rgba(16,24,32,.6); }
      .stButton>button { border:1px solid var(--neon); color: var(--neon); background: transparent; }
      .stButton>button:hover { box-shadow: 0 0 12px rgba(57,255,20,.4); }
      .metric-card { background: rgba(16,24,32,.6); border: 1px solid rgba(57,255,20,.25); border-radius: 8px; padding: 10px; }
      .metric-value { color: var(--neon); font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)
col1, col2 = st.columns([0.5, 0.5])

st.markdown("""
# Manazir OCR — Nerd Mode

This app lets you try Manazir OCR, inspired by Ibn al-Haytham's Kitab al-Manazir (Book of Optics).
""")

# Get mode selection
mode = st.sidebar.selectbox(
    "Run Mode",
    ["None", "Classic (hf/vllm)", "Select Model (Registry)"],
    index=0,
)

# Classic method sub-selection
model_mode = None
if mode == "Classic (hf/vllm)":
    model_mode = st.sidebar.radio("Backend", ["hf", "vllm"], index=0, help="hf = local; vllm = remote server")

# Registry model selection
selected_model_id = None
custom_device = None
if mode == "Select Model (Registry)":
    arabic_models = [m for m in MODEL_REGISTRY.values() if ("*" in m.languages or "ar" in m.languages)]
    display_to_id = {f"{m.display_name} ({m.model_id})": m.model_id for m in arabic_models}
    if not display_to_id:
        st.sidebar.error("No models available in registry.")
    else:
        selected_display = st.sidebar.selectbox("Model", list(display_to_id.keys()))
        selected_model_id = display_to_id[selected_display]
        custom_device = st.sidebar.radio("Device", ["cuda", "cpu"], index=0)

# Only load model if a mode is selected
model = None
if mode == "None":
    st.warning("Please select a model mode (Local Model or vLLM Server) to run OCR.")
elif mode == "Classic (hf/vllm)":
    model = load_model(model_mode)
else:
    model = None  # created on demand for selected model

in_file = st.sidebar.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)

if in_file is None:
    st.stop()

filetype = in_file.type
page_count = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
    page_number = st.sidebar.number_input(
        f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count
    )

    pil_image = get_page_image(in_file, page_number)
else:
    pil_image = Image.open(in_file).convert("RGB")
    page_number = None

run_ocr = st.sidebar.button("Run OCR")

if pil_image is None:
    st.stop()

if run_ocr:
    if mode == "None":
        st.error("Please select a model mode (hf or vllm) to run OCR.")
    elif mode == "Classic (hf/vllm)":
        result, layout_image = ocr_layout(
            pil_image,
            model,
        )

        # Embed images as base64 data URLs in the markdown
        markdown_with_images = embed_images_in_markdown(result.markdown, result.images)

        with col1:
            html_tab, text_tab, layout_tab = st.tabs(
                ["HTML", "HTML as text", "Layout Image"]
            )
            with html_tab:
                st.markdown(markdown_with_images, unsafe_allow_html=True)
                st.download_button(
                    label="Download Markdown",
                    data=result.markdown,
                    file_name=f"{in_file.name.rsplit('.', 1)[0]}_page{page_number if page_number is not None else 0}.md",
                    mime="text/markdown",
                )
            with text_tab:
                st.text(result.html)

            if layout_image:
                with layout_tab:
                    st.image(
                        layout_image,
                        caption="Detected Layout",
                        use_container_width=True,
                    )
                    st.text_area(result.raw)
    else:
        # Registry-selected custom model path
        if not selected_model_id:
            st.error("Select a model from the registry.")
        else:
            with st.spinner(f"Running {selected_model_id}..."):
                try:
                    ocr_model = create_model(selected_model_id, device=custom_device or "cuda")
                    ocr_result = ocr_model.process_image(pil_image)
                except Exception as e:
                    st.error(f"Error running model: {e}")
                    ocr_result = None
            if ocr_result:
                with col1:
                    st.markdown("### Output (Plain Text)")
                    st.markdown(f"<div class='nerd-panel'><pre>{ocr_result.text}</pre></div>", unsafe_allow_html=True)
                    st.download_button(
                        label="Download Text",
                        data=ocr_result.text,
                        file_name=f"{in_file.name.rsplit('.', 1)[0]}_ocr.txt",
                        mime="text/plain",
                    )

with col2:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)
