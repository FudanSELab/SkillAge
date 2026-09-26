from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class TokenLengthProvider(Protocol):
    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...


class TfidfEmbeddingProvider:
    def __init__(self, max_features: int = 8192) -> None:
        self.max_features = max_features

    @property
    def name(self) -> str:
        return f"tfidf-word-bigram-{self.max_features}"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.max_features,
            norm="l2",
        )
        return vectorizer.fit_transform(texts).astype(np.float32).toarray()


class WhitespaceTokenLengthProvider:
    @property
    def name(self) -> str:
        return "whitespace-v1"

    def count(self, text: str) -> int:
        return max(1, len(text.split()))

