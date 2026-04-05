
import re
from rank_bm25 import BM25Okapi


class SparseEncoder:
    """BM25 encoder for Russian text using rank_bm25."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.corpus_texts: list[str] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[а-яёa-z0-9]+", text.lower())

    def fit(self, documents: list[str]):
        self.corpus_texts = documents
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """Returns list of (doc_index, score) sorted by score desc."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:limit]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
