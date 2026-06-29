"""DocVision-AI Streamlit Web Application.

A production-ready web application for document intelligence, allowing users to upload
document images or PDF files to classify them into 'Resume' or 'Invoice' categories.
Reuses existing preprocessing and prediction modules for model inference.
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Tuple
import tensorflow as tf
from PIL import Image
import streamlit as st

# Setup project root path for modular imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import BEST_MODEL_PATH, CLASSES, OUTPUTS_DIR
from src.predict import predict_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DocVisionApp")


@st.cache_resource
def load_classification_model() -> tf.keras.Model:
    """Loads and caches the trained Keras model.

    Applies the Keras 3 / TF 2.16+ deserialization patch.
    """
    logger.info(f"Loading best model from: {BEST_MODEL_PATH}")
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {BEST_MODEL_PATH}. "
            "Please run src/trainer.py first to train and save the model."
        )

    # Pop quantization_config argument during Dense initialization to prevent
    # "TypeError: Unrecognized keyword arguments passed to Dense" when loading models.
    original_dense_init = tf.keras.layers.Dense.__init__

    def patched_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        original_dense_init(self, *args, **kwargs)

    tf.keras.layers.Dense.__init__ = patched_dense_init

    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    logger.info("Model loaded successfully into Streamlit cache.")
    return model


def pdf_to_first_page_image(pdf_bytes: bytes, output_path: Path) -> Path:
    """Converts the first page of a PDF bytes object into a PNG image saved at output_path.

    Uses PyMuPDF (fitz) with a fallback to pdf2image.
    """
    try:
        # Try PyMuPDF first (runs on Windows without system requirements like Poppler)
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("The uploaded PDF file is empty.")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=150)
        pix.save(str(output_path))
        return output_path
    except Exception as e_fitz:
        logger.warning(
            f"PyMuPDF PDF conversion failed or not installed: {e_fitz}. Trying pdf2image..."
        )
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
            if not images:
                raise ValueError("Could not convert PDF first page.")
            images[0].save(output_path, "PNG")
            return output_path
        except Exception as e_pdf2image:
            logger.error(
                f"Both PDF conversion libraries failed. PyMuPDF error: {e_fitz}, pdf2image error: {e_pdf2image}"
            )
            raise RuntimeError(
                "Failed to convert PDF. Ensure PyMuPDF is installed or Poppler is configured for pdf2image."
            ) from e_pdf2image


def process_and_predict(
    file_bytes: bytes,
    file_name: str,
    model: tf.keras.Model,
    temp_dir: Path = OUTPUTS_DIR
) -> Tuple[dict, Image.Image]:
    """Processes an uploaded file (Image or PDF), runs model prediction, and returns results.

    Designed to be highly modular and easily exportable for future FastAPI integrations.

    Args:
        file_bytes (bytes): The raw uploaded file bytes.
        file_name (str): The name of the file (to extract extension).
        model (tf.keras.Model): Loaded classification model.
        temp_dir (Path): The directory to store temporary processing files.

    Returns:
        Tuple[dict, Image.Image]: A tuple containing:
            - dict: Prediction results (class_name, confidence, probabilities)
            - Image.Image: PIL Image of the processed preview image.
    """
    suffix = Path(file_name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".pdf"]:
        raise ValueError(
            f"Unsupported file format: {suffix}. Please upload JPG, PNG, or PDF."
        )

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_img_path = temp_dir / f"temp_inference_{uuid.uuid4().hex}.png"

    try:
        if suffix == ".pdf":
            logger.info("Converting PDF first page to image...")
            pdf_to_first_page_image(file_bytes, temp_img_path)
        else:
            logger.info("Writing image upload to disk...")
            with open(temp_img_path, "wb") as f:
                f.write(file_bytes)

        # Run inference using predict_image from predict.py
        results = predict_image(temp_img_path, model)

        # Load image into memory for rendering preview and clean up file
        preview_img = Image.open(temp_img_path)
        preview_img.load()  # Force load image data

        return results, preview_img

    finally:
        # Clean up temp file to prevent disk fill-up
        if temp_img_path.exists():
            try:
                temp_img_path.unlink()
                logger.info(f"Temporary file cleaned up: {temp_img_path.name}")
            except Exception as cleanup_err:
                logger.warning(
                    f"Failed to clean up file {temp_img_path}: {cleanup_err}"
                )


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="DocVision AI - Intelligent Document Processing",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Inject Custom CSS for Premium UI Styling
    st.markdown("""
    <style>
    /* Import modern Outfit font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Modern Gradient Title */
    .title-gradient {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        line-height: 1.2;
    }
    .subtitle-text {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }

    /* Sidebar Glassmorphic Card */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
    }
    .sidebar-card-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #F8FAFC;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.4rem;
    }
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .sidebar-stat-label {
        color: #94A3B8;
    }
    .sidebar-stat-value {
        color: #818CF8;
        font-weight: 600;
    }

    /* Result Card Layout */
    .result-card {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    .result-header {
        font-size: 0.9rem;
        color: #FFFFFF;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .confidence-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.2rem 0.75rem;
        border-radius: 9999px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }

    /* Dynamic Probability Chart Styles */
    .prob-container {
        margin-top: 1.2rem;
    }
    .prob-row {
        margin-bottom: 1rem;
    }
    .prob-info {
        display: flex;
        justify-content: space-between;
        font-weight: 500;
        font-size: 0.95rem;
        color: #000000;
        margin-bottom: 0.3rem;
    }
    .prob-track {
        background-color: rgba(0, 0, 0, 0.10);
        height: 10px;
        border-radius: 9999px;
        overflow: hidden;
    }
    .prob-bar {
        height: 100%;
        border-radius: 9999px;
    }
    .prob-bar-resume {
        background: linear-gradient(90deg, #6366F1 0%, #4F46E5 100%);
        box-shadow: 0 0 8px rgba(99, 102, 241, 0.4);
    }
    .prob-bar-invoice {
        background: linear-gradient(90deg, #EC4899 0%, #DB2777 100%);
        box-shadow: 0 0 8px rgba(236, 72, 153, 0.4);
    }
    .prob-bar-other {
        background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%);
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    # 3. Sidebar UI
    st.sidebar.image(
        "https://img.icons8.com/nolan/128/artificial-intelligence.png",
        width=80
    )
    st.sidebar.markdown(
        "<h2 style='margin-top: 0;'>DocVision AI</h2>",
        unsafe_allow_html=True
    )
    st.sidebar.write("Document Intelligence Portal for classification and parsing.")

    # Sidebar model info card
    st.sidebar.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Model Specifications</div>
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Model</span>
            <span class="sidebar-stat-value">Custom CNN</span>
        </div>
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Classes</span>
            <span class="sidebar-stat-value">Resume, Invoice</span>
        </div>
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Test Accuracy</span>
            <span class="sidebar-stat-value">94.0%</span>
        </div>
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">Input Shape</span>
            <span class="sidebar-stat-value">224 x 224 x 3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.info(
        "ℹ️ **Usage Notes:** Upload a Resume or an Invoice document. "
        "If a PDF is uploaded, the app extracts and runs prediction on the first page."
    )

    # 4. Main Page Header
    st.markdown(
        "<div class='title-container'>"
        "<h1 class='title-gradient'>DocVision AI 👁️📄</h1>"
        "<div class='subtitle-text'>Production-grade Document Classifier using Deep Convolutional Neural Networks</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # 5. Load Cached Model
    try:
        model = load_classification_model()
    except Exception as e:
        st.error(
            f"❌ **Failed to load model.** Please verify that the model exists at "
            f"`{BEST_MODEL_PATH}`.\n\nError details: `{e}`"
        )
        logger.error(f"Model load exception: {e}")
        return

    # 6. Upload Section
    st.markdown("### 📥 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document image (PNG, JPG, JPEG) or PDF file",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload the document to automatically identify if it is a Resume or an Invoice."
    )

    # 7. Prediction and Visualization Logic
    if uploaded_file is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 1], gap="large")

        # Read uploaded bytes
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name

        # Run Prediction inside a beautiful spinner
        with st.spinner("🔍 Analyzing document structure and contents..."):
            try:
                results, preview_img = process_and_predict(
                    file_bytes,
                    file_name,
                    model
                )

                # Column 1: Preview
                with col1:
                    st.markdown("### 📄 Document Preview")
                    st.image(
                        preview_img,
                        use_container_width=True,
                        caption=f"Preview of {file_name} (First page for PDFs)"
                    )

                # Column 2: Results
                with col2:
                    st.markdown("### 📊 Classification Analysis")

                    class_name = results["class_name"]
                    confidence = results["confidence"]
                    probabilities = results["probabilities"]

                    # Display main card
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">Predicted Classification</div>
                        <div class="result-value">
                            <span>{class_name}</span>
                            <span class="confidence-badge">{confidence:.2f}% Match</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Dynamic Custom HTML Bars for Probabilities
                    st.markdown("#### Probability Distribution")
                    prob_html_lines = ['<div class="prob-container">']
                    for cls_name, prob_val in probabilities.items():
                        prob_pct = prob_val * 100
                        color_class = "prob-bar-other"
                        if cls_name.lower() == "resume":
                            color_class = "prob-bar-resume"
                        elif cls_name.lower() == "invoice":
                            color_class = "prob-bar-invoice"

                        prob_html_lines.append(
                            f'<div class="prob-row">'
                            f'<div class="prob-info">'
                            f'<span>{cls_name}</span>'
                            f'<span>{prob_pct:.2f}%</span>'
                            f'</div>'
                            f'<div class="prob-track">'
                            f'<div class="prob-bar {color_class}" style="width: {prob_pct}%;"></div>'
                            f'</div>'
                            f'</div>'
                        )
                    prob_html_lines.append('</div>')
                    prob_html = "".join(prob_html_lines)
                    st.markdown(prob_html, unsafe_allow_html=True)

                    # Additional info card
                    st.success(
                        f"✅ Classified as **{class_name}** with "
                        f"**{confidence:.1f}%** confidence."
                    )

            except Exception as e:
                st.error(
                    f"❌ **An error occurred during prediction.**\n\n"
                    f"Details: `{e}`"
                )
                logger.exception("Error processing uploaded file.")


if __name__ == "__main__":
    main()
