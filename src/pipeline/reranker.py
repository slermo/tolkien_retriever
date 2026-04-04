from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        if not results:
            return []
        pairs = [(query, r["text"]) for r in results]
        scores = self._model.predict(pairs)
        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)
        ranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
        return ranked[:top_k]
