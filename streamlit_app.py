import sys
from pathlib import Path

import torch
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_siamese_lstm import SiameseLSTM, encode  # noqa: E402

MODEL_PATH = ROOT / "models" / "siamese_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    args = checkpoint.get("args", {})
    vocab = checkpoint["vocab"]
    max_len = int(args.get("max_len", 40))

    model = SiameseLSTM(
        vocab_size=len(vocab),
        embedding_dim=int(args.get("embedding_dim", 128)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        dropout=float(args.get("dropout", 0.2)),
        num_layers=int(args.get("num_layers", 1)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, vocab, max_len


def predict_duplicate(model, vocab, max_len, question1: str, question2: str):
    q1_ids = encode(question1, vocab, max_len)
    q2_ids = encode(question2, vocab, max_len)
    q1 = torch.tensor([q1_ids], dtype=torch.long, device=DEVICE)
    q2 = torch.tensor([q2_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        probability = torch.sigmoid(model(q1, q2)).item()

    return {
        "probability": probability,
        "percentage": round(probability * 100, 2),
        "is_duplicate": probability >= 0.5,
        "label": "Duplicate" if probability >= 0.5 else "Not Duplicate",
    }


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
model, vocab, max_len = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Duplicate Question Detector")
st.caption(f"Model: `{MODEL_PATH}` · Device: `{DEVICE}`")
st.divider()

col1, col2 = st.columns(2)
with col1:
    question1 = st.text_area("Question 1", placeholder="Enter the first question…", height=120)
with col2:
    question2 = st.text_area("Question 2", placeholder="Enter the second question…", height=120)

if st.button("Check for Duplicates", type="primary", use_container_width=True):
    q1 = question1.strip()
    q2 = question2.strip()

    if not q1 or not q2:
        st.warning("Please enter both questions before submitting.")
    else:
        with st.spinner("Analysing…"):
            result = predict_duplicate(model, vocab, max_len, q1, q2)

        st.divider()

        # Verdict banner
        if result["is_duplicate"]:
            st.success(f"✅ **{result['label']}** — {result['percentage']}% confidence")
        else:
            st.error(f"❌ **{result['label']}** — {result['percentage']}% confidence")

        # Progress bar
        st.progress(result["probability"], text=f"Similarity score: {result['percentage']}%")

        # Details expander
        with st.expander("Raw output"):
            st.json(result)
