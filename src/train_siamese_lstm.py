import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import load_question_pairs, split_pairs
from text_utils import tokenize


PAD = "<pad>"
UNK = "<unk>"


def build_vocab(df, max_vocab: int, min_freq: int):
    counter = Counter()
    for text in df["question1"].tolist() + df["question2"].tolist():
        counter.update(tokenize(text))
    tokens = [token for token, count in counter.most_common(max_vocab - 2) if count >= min_freq]
    vocab = {PAD: 0, UNK: 1}
    vocab.update({token: index + 2 for index, token in enumerate(tokens)})
    return vocab


def encode(text: str, vocab, max_len: int):
    ids = [vocab.get(token, vocab[UNK]) for token in tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab[PAD]] * (max_len - len(ids)))
    return ids


class QuestionPairDataset(Dataset):
    def __init__(self, df, vocab, max_len: int):
        self.q1 = np.array([encode(text, vocab, max_len) for text in df["question1"]], dtype=np.int64)
        self.q2 = np.array([encode(text, vocab, max_len) for text in df["question2"]], dtype=np.int64)
        self.y = df["is_duplicate"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.tensor(self.q1[index], dtype=torch.long),
            torch.tensor(self.q2[index], dtype=torch.long),
            torch.tensor(self.y[index], dtype=torch.float32),
        )


class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            num_layers=num_layers,
        )
        encoded_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode_once(self, tokens):
        embedded = self.embedding(tokens)
        output, _ = self.encoder(embedded)
        mask = (tokens != 0).unsqueeze(-1)
        output = output.masked_fill(~mask, -1e9)
        return output.max(dim=1).values

    def forward(self, q1, q2):
        v1 = self.encode_once(q1)
        v2 = self.encode_once(q2)
        pair = torch.cat([v1, v2, torch.abs(v1 - v2), v1 * v2], dim=1)
        return self.classifier(pair).squeeze(1)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for q1, q2, labels in loader:
            logits = model(q1.to(device), q2.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a Siamese BiLSTM duplicate-question model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--max-vocab", type=int, default=60000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    df = load_question_pairs(args.data, sample_size=args.sample_size)
    train_df, test_df = split_pairs(df)
    vocab = build_vocab(train_df, args.max_vocab, args.min_freq)

    train_ds = QuestionPairDataset(train_df, vocab, args.max_len)
    test_ds = QuestionPairDataset(test_df, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseLSTM(len(vocab), args.embedding_dim, args.hidden_dim, args.dropout, args.num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for q1, q2, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            q1, q2, labels = q1.to(device), q2.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(q1, q2)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        print(f"epoch={epoch} train_loss={total_loss / len(train_ds):.4f}")

    metrics = evaluate(model, test_loader, device)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "siamese_lstm.pt"
    report_path = report_dir / "siamese_lstm_metrics.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "args": vars(args),
        },
        model_path,
    )
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {report_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
