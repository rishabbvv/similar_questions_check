from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from text_utils import tokenize


def _pair_text(df):
    return df["question1"].astype(str), df["question2"].astype(str)


@dataclass
class TfidfPairVectorizer:
    max_features: int = 50000
    ngram_range: tuple = (1, 2)
    min_df: int = 2

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            strip_accents="unicode",
        )

    def fit(self, df, y=None):
        q1, q2 = _pair_text(df)
        self.vectorizer.fit(q1.tolist() + q2.tolist())
        return self

    def transform(self, df):
        q1, q2 = _pair_text(df)
        x1 = self.vectorizer.transform(q1)
        x2 = self.vectorizer.transform(q2)
        return sparse.hstack([x1, x2, abs(x1 - x2), x1.multiply(x2)], format="csr")

    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df)


class Word2VecPairVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, workers: int = 2, epochs: int = 10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def fit(self, df, y=None):
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError("Install gensim to use --features word2vec.") from exc

        sentences: List[List[str]] = []
        for value in df["question1"].tolist() + df["question2"].tolist():
            sentences.append(tokenize(value))

        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=42,
        )
        return self

    def _average(self, text: str) -> np.ndarray:
        tokens = [token for token in tokenize(text) if token in self.model_.wv]
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(self.model_.wv[tokens], axis=0)

    def transform(self, df):
        q1 = np.vstack([self._average(text) for text in df["question1"].astype(str)])
        q2 = np.vstack([self._average(text) for text in df["question2"].astype(str)])
        return np.hstack([q1, q2, np.abs(q1 - q2), q1 * q2])

    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df)
