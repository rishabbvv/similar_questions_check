import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lstm_inference import load_lstm_checkpoint, predict_duplicate as infer_duplicate  # noqa: E402


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

    model, vocab, max_len = load_lstm_checkpoint(MODEL_PATH, DEVICE)


def predict_duplicate(question1: str, question2: str):
    return infer_duplicate(model, vocab, max_len, question1, question2, DEVICE)


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
