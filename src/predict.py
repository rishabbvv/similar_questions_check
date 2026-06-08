import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Predict if two questions are duplicates.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--q1", required=True)
    parser.add_argument("--q2", required=True)
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    row = pd.DataFrame([{"question1": args.q1, "question2": args.q2}])
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model[-1], "predict_proba"):
        probability = float(model.predict_proba(row)[0, 1])

    print({"is_duplicate": prediction, "duplicate_probability": probability})


if __name__ == "__main__":
    main()
