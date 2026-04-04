from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._model.encode([f"query: {text}"], normalize_embeddings=True)
        return embeddings.tolist()[0]
