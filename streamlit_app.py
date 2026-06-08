import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lstm_inference import load_lstm_checkpoint, predict_duplicate  # noqa: E402


MODEL_PATH = ROOT / "models" / "siamese_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.set_page_config(
    page_title="Duplicate Question Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(37, 99, 235, 0.30), transparent 32rem),
          linear-gradient(135deg, #f8fbff 0%, #e8eef8 45%, #f5f7fb 100%);
      }
      .block-container {
        max-width: 920px;
        padding-top: 4rem;
      }
      [data-testid="stHeader"] {
        background: transparent;
      }
      .app-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #d9e0ea;
        border-radius: 8px;
        box-shadow: 0 18px 50px rgba(23, 32, 51, 0.12);
        padding: 28px;
      }
      .app-title {
        color: #172033;
        font-size: 36px;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 20px;
      }
      .result-box {
        border-top: 1px solid #d9e0ea;
        margin-top: 24px;
        padding-top: 20px;
      }
      .verdict {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 4px;
      }
      .duplicate {
        color: #0f766e;
      }
      .not-duplicate {
        color: #b42318;
      }
      .score-text {
        color: #607089;
        font-size: 15px;
      }
      textarea {
        border-radius: 8px !important;
      }
      div.stButton > button {
        border-radius: 8px;
        font-weight: 700;
        height: 44px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained LSTM model...")
def get_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Add models/siamese_lstm.pt to this project.")
        st.stop()
    return load_lstm_checkpoint(MODEL_PATH, DEVICE)


model, vocab, max_len = get_model()

st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">Duplicate Question Detector</h1>', unsafe_allow_html=True)

question1 = st.text_area("Question 1", placeholder="Enter the first question", height=130)
question2 = st.text_area("Question 2", placeholder="Enter the second question", height=130)

submitted = st.button("Predict", type="primary", use_container_width=True)

if submitted:
    q1 = question1.strip()
    q2 = question2.strip()
    if not q1 or not q2:
        st.warning("Please enter both questions.")
    else:
        result = predict_duplicate(model, vocab, max_len, q1, q2, DEVICE)
        css_class = "duplicate" if result["is_duplicate"] else "not-duplicate"
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="verdict {css_class}">{result["label"]}</div>'
            f'<div class="score-text">{result["percentage"]}% duplicate probability</div>',
            unsafe_allow_html=True,
        )
        st.progress(result["probability"])
        with st.expander("Raw output"):
            st.json(result)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
