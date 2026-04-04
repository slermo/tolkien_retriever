import os
from tqdm import tqdm
from src.pipeline.chunker import Chunker
from src.pipeline.embedder import Embedder
from src.qdrant.qdrant import QdrantDB

DATA_DIR = "data/cleaned"
BATCH_SIZE = 64


def ingest():
    chunker = Chunker(chunk_size=500, chunk_overlap=100)
    embedder = Embedder()
    db = QdrantDB()

    # Создаём коллекцию с нужной размерностью
    db.create_collection(vector_size=embedder.dimension)

    point_id = 0

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        source = os.path.splitext(filename)[0]
        chunks = chunker.chunk(text, source=source)
        print(f"{source}: {len(chunks)} chunks")

        # Загружаем батчами
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Uploading {source}"):
            batch = chunks[i : i + BATCH_SIZE]

            texts = [c.text for c in batch]
            vectors = embedder.embed_documents(texts)
            ids = list(range(point_id, point_id + len(batch)))
            payloads = [{"text": c.text, "source": c.source, "index": c.index} for c in batch]

            db.upsert(ids=ids, vectors=vectors, payloads=payloads)
            point_id += len(batch)

    print(f"Done. Total points: {point_id}")


if __name__ == "__main__":
    ingest()
