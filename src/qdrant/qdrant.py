from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantDB:
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "lotr_collection"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int):
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, ids: list[int], vectors: list[list[float]], payloads: list[dict]):
        points = [
            PointStruct(id=id_, vector=vector, payload=payload)
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)

    def search(self, vector: list[float], limit: int = 5) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
        )
        return [
            {"text": r.payload["text"], "source": r.payload["source"], "score": r.score}
            for r in results.points
        ]
