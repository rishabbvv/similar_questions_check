from pathlib import Path

import torch
from torch import nn

from text_utils import tokenize


PAD = "<pad>"
UNK = "<unk>"


def encode(text: str, vocab, max_len: int):
    ids = [vocab.get(token, vocab[UNK]) for token in tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab[PAD]] * (max_len - len(ids)))
    return ids


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


def load_lstm_checkpoint(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint.get("args", {})
    vocab = checkpoint["vocab"]
    max_len = int(args.get("max_len", 40))
    model = SiameseLSTM(
        vocab_size=len(vocab),
        embedding_dim=int(args.get("embedding_dim", 128)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        dropout=float(args.get("dropout", 0.2)),
        num_layers=int(args.get("num_layers", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab, max_len


def predict_duplicate(model, vocab, max_len, question1: str, question2: str, device):
    q1_ids = encode(question1, vocab, max_len)
    q2_ids = encode(question2, vocab, max_len)
    q1 = torch.tensor([q1_ids], dtype=torch.long, device=device)
    q2 = torch.tensor([q2_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        probability = torch.sigmoid(model(q1, q2)).item()

    return {
        "probability": probability,
        "percentage": round(probability * 100, 2),
        "is_duplicate": probability >= 0.5,
        "label": "Duplicate" if probability >= 0.5 else "Not duplicate",
    }
