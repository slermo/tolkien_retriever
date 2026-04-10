from src.pipeline.embedder import Embedder
from src.pipeline.reranker import Reranker
from src.qdrant.qdrant import QdrantDB
from src.llm.base import BaseLLM
from src.llm.local.llm import LocalLLM

SYSTEM_PROMPT = (
    "Ты — рассказчик в стиле Дж. Р. Р. Толкина. "
    "Отвечай на вопрос, опираясь на предоставленный контекст из книг. "
    "Сохраняй стилистику и дух оригинала."
)


class RAGPipeline:
    def __init__(self, llm: BaseLLM | None = None, adapter_path: str = "tolkien_lora_adapter"):
        self.embedder = Embedder()
        self.reranker = Reranker()
        self.db = QdrantDB()
        self.llm = llm or LocalLLM(adapter_path=adapter_path)

    def retrieve(self, query: str, top_k: int = 5, retrieve: int = 20) -> list[dict]:
        vector = self.embedder.embed_query(query)
        results = self.db.search(vector, limit=retrieve)
        return self.reranker.rerank(query, results, top_k=top_k)

    def build_prompt(self, query: str, contexts: list[dict]) -> str:
        context_text = "\n\n".join(
            f"[{i+1}] {r['text']}" for i, r in enumerate(contexts)
        )
        return f"Контекст:\n{context_text}\n\nВопрос: {query}"

    def generate(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
    ) -> str:
        contexts = self.retrieve(query, top_k=top_k)
        user_message = self.build_prompt(query, contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        text, _ = self.llm.chat(messages, temperature=temperature)
        return text


if __name__ == "__main__":
    rag = RAGPipeline()
    while True:
        query = input("\nВопрос: ")
        if query.lower() in ("exit", "quit", "q"):
            break
        answer = rag.generate(query)
        print(f"\nОтвет:\n{answer}")
