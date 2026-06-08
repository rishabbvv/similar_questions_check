import argparse
import json
from pathlib import Path

import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data import load_question_pairs, split_pairs
from features import TfidfPairVectorizer, Word2VecPairVectorizer


def build_pipeline(model_name: str, feature_name: str) -> Pipeline:
    if feature_name == "tfidf":
        vectorizer = TfidfPairVectorizer()
    elif feature_name == "word2vec":
        vectorizer = Word2VecPairVectorizer()
    else:
        raise ValueError(f"Unsupported feature type: {feature_name}")

    if model_name == "logistic":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        steps = [("features", vectorizer), ("classifier", classifier)]
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )
        steps = [("features", vectorizer)]
        if feature_name == "tfidf":
            steps.append(("svd", TruncatedSVD(n_components=300, random_state=42)))
        else:
            steps.append(("identity", FunctionTransformer(validate=False)))
        steps.append(("classifier", classifier))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps)


def evaluate(model: Pipeline, df):
    y_true = df["is_duplicate"].to_numpy()
    y_pred = model.predict(df)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    if hasattr(model[-1], "predict_proba"):
        y_score = model.predict_proba(df)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train duplicate-question baseline models.")
    parser.add_argument("--data", required=True, help="Path to question-pair CSV.")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--features", choices=["tfidf", "word2vec"], default="tfidf")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    df = load_question_pairs(args.data, sample_size=args.sample_size)
    train_df, test_df = split_pairs(df, test_size=args.test_size)

    pipeline = build_pipeline(args.model, args.features)
    pipeline.fit(train_df, train_df["is_duplicate"])
    metrics = evaluate(pipeline, test_df)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{args.model}_{args.features}.joblib"
    report_path = report_dir / f"{args.model}_{args.features}_metrics.json"
    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {report_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
