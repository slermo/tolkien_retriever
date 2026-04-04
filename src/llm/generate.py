from src.pipeline.embedder import Embedder
from src.pipeline.reranker import Reranker
from src.qdrant.qdrant import QdrantDB
from src.llm.model import load_for_inference

SYSTEM_PROMPT = (
    "Ты — рассказчик в стиле Дж. Р. Р. Толкина. "
    "Отвечай на вопрос, опираясь на предоставленный контекст из книг. "
    "Сохраняй стилистику и дух оригинала."
)


class RAGPipeline:
    def __init__(self, adapter_path: str = "tolkien_lora_adapter"):
        self.embedder = Embedder()
        self.reranker = Reranker()
        self.db = QdrantDB()
        self.model, self.tokenizer = load_for_inference(adapter_path)

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
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        contexts = self.retrieve(query, top_k=top_k)
        user_message = self.build_prompt(query, contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    rag = RAGPipeline()
    while True:
        query = input("\nВопрос: ")
        if query.lower() in ("exit", "quit", "q"):
            break
        answer = rag.generate(query)
        print(f"\nОтвет:\n{answer}")
