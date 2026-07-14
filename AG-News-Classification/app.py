"""Production-grade Streamlit web application for AG News text classification.

This app allows users to input raw news articles and classify them using four
different neural network models: RNN, LSTM, BiLSTM + Self-Attention, and BERT.
It supports single-model inference as well as a side-by-side comparison mode
to evaluate latency, confidence scores, and probability distributions.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import streamlit as st

# Configure project root to enable module resolution
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.pipeline.predict import NLPPredictor
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("streamlit_app")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
MODELS: Dict[str, Dict[str, str]] = {
    "rnn": {
        "name": "Simple RNN",
        "description": (
            "A basic Recurrent Neural Network. Fast to train and run, but "
            "can suffer from vanishing gradients on longer sequences. Trained with "
            "a sequence length of 120."
        ),
    },
    "lstm": {
        "name": "LSTM",
        "description": (
            "Long Short-Term Memory network. Capable of learning long-term "
            "dependencies using gating mechanisms. Trained with a sequence length of 80."
        ),
    },
    "attention": {
        "name": "BiLSTM + Self-Attention",
        "description": (
            "Bidirectional LSTM coupled with a custom self-attention mechanism. "
            "Allows the network to focus on the key informative words in the text."
        ),
    },
    "bert": {
        "name": "BERT",
        "description": (
            "Fine-tuned BERT (Bidirectional Encoder Representations from Transformers) "
            "model. Delivers state-of-the-art accuracy by leveraging contextual representation."
        ),
    },
}

CATEGORIES: List[str] = ["World", "Sports", "Business", "Sci/Tech"]

SAMPLE_ARTICLES: Dict[str, Dict[str, str]] = {
    "World News (Ceasefire)": {
        "category": "World",
        "text": (
            "The UN Security Council today demanded an immediate ceasefire in the conflict "
            "zone, calling on all nations to cease shipments of weapons and open corridors "
            "for humanitarian aid."
        ),
    },
    "Sports News (Football Match)": {
        "category": "Sports",
        "text": (
            "A dramatic 93rd-minute header by the star striker sealed a historic 2-1 victory "
            "in the league cup final today. The stadium erupted as the referee blew the final whistle."
        ),
    },
    "Business News (Market Surge)": {
        "category": "Business",
        "text": (
            "Stock indices surged to record highs on Tuesday after a major tech conglomerate "
            "reported quarterly earnings that significantly beat analysts' expectations, coupled "
            "with a new stock buyback plan."
        ),
    },
    "Sci/Tech News (Space Telescope)": {
        "category": "Sci/Tech",
        "text": (
            "Astronomers utilizing the James Webb Space Telescope have detected traces of carbon "
            "dioxide and methane in the atmosphere of a habitable-zone exoplanet located 120 light-years away."
        ),
    },
}

BERT_HELPER_CODE = """import argparse
import os
import sys
from pathlib import Path

# Redirect stdout to stderr immediately to prevent logger output in stdout
original_stdout = sys.stdout
sys.stdout = sys.stderr

# Configure legacy Keras before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Add project root to sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.pipeline.predict import NLPPredictor

# Patch the load_model method for BERT
def patched_load_model(self):
    from src.models.bert_model import BERTModel
    bert_builder = BERTModel()
    model = bert_builder.build_model()
    model.load_weights(str(self.model_path))
    return model

NLPPredictor._load_model = patched_load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    
    predictor = NLPPredictor(model_type="bert")
    result = predictor.predict(args.text)
    
    import json
    # Print only the JSON result to the original stdout
    print(json.dumps(result), file=original_stdout)
"""

# ==============================================================================
# SUBPROCESS BERT WRAPPER
# ==============================================================================
class SubprocessBertPredictor:
    """Runs BERT predictions in an isolated subprocess to bypass Keras/TF version conflicts."""

    def __init__(self) -> None:
        self.helper_path = Path(_project_root) / "bert_subprocess_helper.py"
        self._ensure_helper_exists()

    def _ensure_helper_exists(self) -> None:
        """Writes the helper script to disk if it is missing."""
        if not self.helper_path.exists():
            logger.info(f"Writing BERT subprocess helper to: {self.helper_path}")
            with open(self.helper_path, "w", encoding="utf-8") as f:
                f.write(BERT_HELPER_CODE)

    def predict(self, text: str) -> Dict[str, Any]:
        """Runs the helper script in a subprocess and parses the JSON result."""
        cmd = [sys.executable, str(self.helper_path), "--text", text]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=_project_root,
        )
        return json.loads(result.stdout)


# ==============================================================================
# CACHED MODEL LOADER
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_cached_predictor(model_key: str) -> Any:
    """Loads and caches the predictor instance for the specified model key.

    Args:
        model_key (str): Key of the model to load.

    Returns:
        Any: The cached predictor instance.
    """
    logger.info(f"App requested loading of model: {model_key}")
    if model_key == "bert":
        return SubprocessBertPredictor()
    elif model_key == "rnn":
        # RNN was trained with max_sequence_length=120
        return NLPPredictor(model_type=model_key, max_sequence_length=120)
    else:
        return NLPPredictor(model_type=model_key)


def get_predictor_safely(model_key: str) -> Optional[Any]:
    """Retrieves the cached predictor, handling exceptions gracefully.

    Args:
        model_key (str): The key representing the model.

    Returns:
        Optional[Any]: The predictor if successfully loaded, else None.
    """
    try:
        return load_cached_predictor(model_key)
    except Exception as e:
        logger.error(f"Error loading model '{model_key}': {str(e)}", exc_info=True)
        st.error(
            f"⚠️ **Failed to load the {MODELS[model_key]['name']} model.**\n\n"
            f"Please verify that the model checkpoint exists in your `saved_models` directory "
            f"and that all dependencies are correctly configured.\n\n"
            f"*Error details: {str(e)}*"
        )
        return None


# ==============================================================================
# INFERENCE UTILITIES
# ==============================================================================
def run_single_inference(
    predictor: Any, text: str
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Executes prediction using a predictor and measures latency.

    Args:
        predictor (Any): Loaded predictor object.
        text (str): Raw input text.

    Returns:
        Tuple[Optional[Dict[str, Any]], float]: (prediction_result, latency_ms)
    """
    start_time = time.perf_counter()
    try:
        result = predictor.predict(text)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return result, latency_ms
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        st.error(f"Prediction execution failed. *Details: {str(e)}*")
        return None, 0.0


# ==============================================================================
# APP LAYOUT & LOGIC
# ==============================================================================
def main() -> None:
    """Configures and runs the Streamlit app."""
    # Set page configuration
    st.set_page_config(
        page_title="AG News NLP Text Classifier",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject clean custom styling
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #1e293b;
        }
        .subtitle {
            font-size: 1.1rem;
            margin-bottom: 2rem;
            color: #64748b;
        }
        .prediction-card {
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e2e8f0;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main Header
    st.markdown(
        '<div class="main-title">📰 AG News NLP Text Classifier</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Production-grade classification of news articles into World, '
        "Sports, Business, or Science/Technology categories using Deep Learning and Transformers.</div>",
        unsafe_allow_html=True,
    )

    # ==============================================================================
    # SIDEBAR CONFIGURATION
    # ==============================================================================
    st.sidebar.title("⚙️ App Settings")
    st.sidebar.markdown("---")

    # Model selector list
    model_labels = {k: v["name"] for k, v in MODELS.items()}
    model_labels["compare"] = "📊 Compare All Models"

    selected_label_key = st.sidebar.selectbox(
        "Choose Classification Model",
        options=list(model_labels.keys()),
        format_func=lambda x: model_labels[x],
        index=len(model_labels) - 1,  # Default to "Compare All Models"
    )

    st.sidebar.markdown("---")

    # Display information about the selected model
    if selected_label_key == "compare":
        st.sidebar.subheader("📊 Compare All Models")
        st.sidebar.info(
            "Loads all four architectures and runs prediction. "
            "Compares predicted classes, confidence scores, and latency."
        )
    else:
        st.sidebar.subheader(f"🧠 {MODELS[selected_label_key]['name']}")
        st.sidebar.write(MODELS[selected_label_key]["description"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ Target Categories")
    for category in CATEGORIES:
        st.sidebar.markdown(f"- **{category}**")

    # Sidebar utility: Cache clearance
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Model Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared! Models will be reloaded on next prediction.")
        logger.info("Streamlit resource cache cleared by user.")

    # ==============================================================================
    # INPUT SECTION
    # ==============================================================================
    st.write("### 📝 Input News Article")

    # Sample Selector
    sample_options = ["None (Write custom text)"] + list(SAMPLE_ARTICLES.keys())
    selected_sample = st.selectbox(
        "Load a sample news article to quickly test:",
        options=sample_options,
        index=0,
    )

    # Determine input text
    default_text = ""
    if selected_sample != "None (Write custom text)":
        default_text = SAMPLE_ARTICLES[selected_sample]["text"]

    text_input = st.text_area(
        "Paste the news headline or summary text here:",
        value=default_text,
        placeholder="Type here...",
        height=150,
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_prediction = st.button("🚀 Classify Text", use_container_width=True)

    with col_info:
        if text_input:
            word_count = len(text_input.split())
            char_count = len(text_input)
            st.markdown(
                f"<div style='padding-top: 10px; color: #64748b;'>{word_count} words | {char_count} characters</div>",
                unsafe_allow_html=True,
            )

    # ==============================================================================
    # PREDICTION & RESULT DISPLAY
    # ==============================================================================
    if run_prediction:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text before classifying.")
            return

        st.write("### 🎯 Classification Results")

        if selected_label_key != "compare":
            # ------------------------------------------------------------------
            # SINGLE MODEL INFERENCE
            # ------------------------------------------------------------------
            model_name = MODELS[selected_label_key]["name"]
            
            with st.spinner(f"Loading {model_name} and running inference..."):
                predictor = get_predictor_safely(selected_label_key)

            if predictor:
                with st.spinner("Processing prediction..."):
                    result, latency = run_single_inference(predictor, text_input)

                if result:
                    # Layout predicted results in columns
                    col_pred, col_details = st.columns([1, 1])

                    with col_pred:
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <h3 style="margin-top:0; color:#475569;">Predicted Class</h3>
                                <h1 style="color:#0284c7; font-size: 2.8rem; margin: 0.5rem 0;">
                                    {result['predicted_class']}
                                </h1>
                                <p style="font-size: 1.2rem; color:#0d9488; font-weight:bold; margin-bottom:0;">
                                    Confidence: {result['confidence_score']:.2%}
                                </p>
                                <p style="font-size: 0.9rem; color:#64748b; margin-top: 0.5rem;">
                                    Inference Latency: <b>{latency:.2f} ms</b>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_details:
                        st.write("#### Probability Distribution")
                        # Present progress bars for probabilities
                        prob_dist = result["probability_distribution"]
                        sorted_probs = sorted(
                            prob_dist.items(), key=lambda x: x[1], reverse=True
                        )

                        for category, prob in sorted_probs:
                            st.write(f"**{category}** ({prob:.2%})")
                            st.progress(prob)

        else:
            # ------------------------------------------------------------------
            # COMPARE ALL MODELS
            # ------------------------------------------------------------------
            st.write("Running predictions across all 4 architectures side-by-side...")
            
            results: Dict[str, Optional[Dict[str, Any]]] = {}
            latencies: Dict[str, float] = {}
            
            # Progress bar for comparison loading
            comp_progress = st.progress(0.0)
            
            for idx, (m_key, m_info) in enumerate(MODELS.items()):
                m_name = m_info["name"]
                
                # Update progress
                progress_val = float(idx) / len(MODELS)
                comp_progress.progress(progress_val, text=f"Evaluating {m_name}...")
                
                predictor = get_predictor_safely(m_key)
                if predictor:
                    res, latency = run_single_inference(predictor, text_input)
                    results[m_key] = res
                    latencies[m_key] = latency
                else:
                    results[m_key] = None
                    latencies[m_key] = 0.0
            
            comp_progress.progress(1.0, text="All models evaluated!")
            time.sleep(0.5)
            comp_progress.empty()

            # Compile comparison data
            table_rows = []
            valid_comparison = False
            
            for m_key, m_info in MODELS.items():
                res = results[m_key]
                latency = latencies[m_key]
                if res:
                    valid_comparison = True
                    table_rows.append(
                        {
                            "Model": m_info["name"],
                            "Predicted Class": res["predicted_class"],
                            "Confidence": f"{res['confidence_score']:.2%}",
                            "Latency (ms)": f"{latency:.2f} ms",
                            "Raw Latency": latency,
                        }
                    )
                else:
                    table_rows.append(
                        {
                            "Model": m_info["name"],
                            "Predicted Class": "N/A (Failed)",
                            "Confidence": "N/A",
                            "Latency (ms)": "N/A",
                            "Raw Latency": 0.0,
                        }
                    )

            if valid_comparison:
                st.write("#### Comparison Overview")
                st.table(pd.DataFrame(table_rows).drop(columns=["Raw Latency"]))

                # Draw grouped bar chart comparing probabilities
                st.write("#### Probability Distribution Comparison")
                
                chart_data = pd.DataFrame(index=CATEGORIES)
                for m_key, m_info in MODELS.items():
                    res = results[m_key]
                    if res:
                        prob_dist = res["probability_distribution"]
                        chart_data[m_info["name"]] = [
                            prob_dist.get(cat, 0.0) for cat in CATEGORIES
                        ]
                
                st.bar_chart(chart_data, height=350)
            else:
                st.error("Could not generate comparison since all model inferences failed.")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    main()
