import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_siamese_lstm import SiameseLSTM, encode  # noqa: E402


MODEL_PATH = ROOT / "models" / "siamese_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
model = None
vocab = None
max_len = None


def load_model():
    global model, vocab, max_len
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

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


def predict_duplicate(question1: str, question2: str):
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
        "label": "Duplicate" if probability >= 0.5 else "Not duplicate",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    question1 = ""
    question2 = ""
    error = None

    if request.method == "POST":
        question1 = request.form.get("question1", "").strip()
        question2 = request.form.get("question2", "").strip()
        if not question1 or not question2:
            error = "Please enter both questions."
        else:
            result = predict_duplicate(question1, question2)

    return render_template(
        "index.html",
        result=result,
        question1=question1,
        question2=question2,
        error=error,
        model_path=str(MODEL_PATH),
        device=str(DEVICE),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    question1 = str(payload.get("question1", "")).strip()
    question2 = str(payload.get("question2", "")).strip()
    if not question1 or not question2:
        return jsonify({"error": "question1 and question2 are required"}), 400
    return jsonify(predict_duplicate(question1, question2))


load_model()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
