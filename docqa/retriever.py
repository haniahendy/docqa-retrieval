"""Keyword/TF-IDF passage retrieval for English and Arabic text.

This is retrieval only. It does not generate answers and uses no neural
embeddings or LLM. It returns the most similar passages with scores.
"""
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")


def normalize(text: str) -> str:
    """Lowercase, strip Arabic diacritics/tatweel, unify alef/yaa/taa-marbuta."""
    text = _AR_DIACRITICS.sub("", text.lower())
    for src in "إأآ":
        text = text.replace(src, "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    # light English plural stripping (vaccines -> vaccine); not a real stemmer
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w.isascii() else w for w in text.split())


def chunk_text(text: str, max_words: int = 60, overlap: int = 10) -> list[str]:
    """Split text into overlapping word windows."""
    if max_words <= overlap:
        raise ValueError("max_words must be larger than overlap")
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks


@dataclass
class Hit:
    doc: str
    text: str
    score: float


class TfidfRetriever:
    def __init__(self, max_words: int = 60, overlap: int = 10):
        self.max_words, self.overlap = max_words, overlap
        self._vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)
        self._chunks: list[tuple[str, str]] = []
        self._matrix = None

    def add_documents(self, docs: dict[str, str]) -> None:
        for name, text in docs.items():
            for c in chunk_text(text, self.max_words, self.overlap):
                self._chunks.append((name, c))
        self._matrix = self._vec.fit_transform([c for _, c in self._chunks])

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("no documents indexed")
        scores = linear_kernel(self._vec.transform([query]), self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [Hit(self._chunks[i][0], self._chunks[i][1], float(scores[i])) for i in order if scores[i] > 0]
