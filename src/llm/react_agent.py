import re
from src.llm.base import BaseLLM
from src.llm.tracing import get_langfuse
from src.pipeline.embedder import Embedder
from src.pipeline.reranker import Reranker
from src.pipeline.sparse_encoder import SparseEncoder
from src.pipeline.chunker import Chunker
from src.qdrant.qdrant import QdrantDB
import os

SYSTEM_PROMPT = """Ты — эксперт по произведениям Толкина. Отвечай на вопросы, используя инструмент поиска по книгам.

У тебя есть инструмент:
- search(query) — поиск по книгам Толкина. Возвращает релевантные отрывки.

Формат работы:
1. Thought: рассуждение о том, что нужно найти
2. Action: search("поисковый запрос")
3. Observation: результаты поиска (заполняется автоматически)
4. ... (можно повторять Thought/Action/Observation до 3 раз)
5. Answer: финальный ответ на основе найденной информации

Правила:
- Делай конкретные, точные поисковые запросы
- Если первый поиск не дал достаточно информации — уточни запрос
- Ответ формулируй развёрнуто, опираясь на текст книги
- Если информация не найдена — честно скажи об этом"""

ACTION_PATTERN = re.compile(r'Action:\s*search\("(.+?)"\)', re.DOTALL)
ANSWER_PATTERN = re.compile(r'Answer:\s*(.+)', re.DOTALL)
MAX_STEPS = 3


class ReActAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.embedder = Embedder()
        self.reranker = Reranker()
        self.sparse_encoder = SparseEncoder()
        self.db = QdrantDB()
        self._all_chunks = []
        self._init_bm25()

    def _init_bm25(self):
        chunker = Chunker(chunk_size=500, chunk_overlap=100)
        for filename in os.listdir("data/cleaned"):
            with open(f"data/cleaned/{filename}", "r", encoding="utf-8") as f:
                text = f.read()
            source = os.path.splitext(filename)[0]
            self._all_chunks.extend(chunker.chunk(text, source=source))
        self.sparse_encoder.fit([c.text for c in self._all_chunks])

    def _search(self, query: str, top_k: int = 5, retrieve: int = 20, trace_span=None) -> str:
        # Retrieval span
        retrieval_span = None
        if trace_span:
            retrieval_span = trace_span.span(name="retrieval", input={"query": query})

        dense_vec = self.embedder.embed_query(query)
        dense_results = self.db.search(dense_vec, limit=retrieve)
        bm25_results = self.sparse_encoder.search(query, limit=retrieve)

        # RRF merge
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        k = 60
        for rank, r in enumerate(dense_results):
            key = r["text"][:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_map[key] = r
        for rank, (idx, _) in enumerate(bm25_results):
            chunk = self._all_chunks[idx]
            key = chunk.text[:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = {"text": chunk.text, "source": chunk.source, "score": 0.0}

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:retrieve]
        results = [doc_map[key] for key in sorted_keys]

        reranked = self.reranker.rerank(query, results, top_k=top_k)

        parts = []
        for i, r in enumerate(reranked):
            parts.append(f"[{i+1}] ({r['source']}) {r['text']}")
        observation = "\n\n".join(parts)

        if retrieval_span:
            retrieval_span.end(output={
                "num_results": len(reranked),
                "top_rerank_score": reranked[0]["rerank_score"] if reranked else 0,
            })

        return observation

    def run(self, question: str, verbose: bool = False) -> str:
        langfuse = get_langfuse()
        trace = None
        if langfuse:
            trace = langfuse.trace(name="react_agent", input={"question": question})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        full_response = ""

        for step in range(MAX_STEPS):
            # LLM generation span
            generation = None
            if trace:
                generation = trace.generation(
                    name=f"llm_step_{step + 1}",
                    input=messages,
                    model=getattr(self.llm, "model", "unknown"),
                )

            response, usage = self.llm.chat(messages, temperature=0.3)
            full_response += response

            if generation:
                generation.end(
                    output=response,
                    usage={
                        "input": usage.get("input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                    },
                )

            if verbose:
                print(f"\n--- Step {step + 1} ---")
                print(response)

            # Check for final answer
            answer_match = ANSWER_PATTERN.search(response)
            action_match = ACTION_PATTERN.search(response)

            if answer_match and not action_match:
                answer = answer_match.group(1).strip()
                if trace:
                    trace.update(output={"answer": answer, "steps": step + 1})
                    langfuse.flush()
                return answer

            if action_match:
                query = action_match.group(1)
                if verbose:
                    print(f"\n🔍 Searching: {query}")

                observation = self._search(query, trace_span=trace)

                if verbose:
                    print(f"📄 Found {len(observation.split('['))-1} results")

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})
                full_response += f"\nObservation:\n{observation}\n"
            else:
                if trace:
                    trace.update(output={"answer": response.strip(), "steps": step + 1})
                    langfuse.flush()
                return response.strip()

        # Max steps reached
        messages.append({
            "role": "user",
            "content": "Пожалуйста, сформулируй финальный ответ на основе найденной информации. Начни с 'Answer:'"
        })

        generation = None
        if trace:
            generation = trace.generation(
                name="llm_final",
                input=messages,
                model=getattr(self.llm, "model", "unknown"),
            )

        response, usage = self.llm.chat(messages, temperature=0.3)

        if generation:
            generation.end(
                output=response,
                usage={
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                },
            )

        answer_match = ANSWER_PATTERN.search(response)
        answer = answer_match.group(1).strip() if answer_match else response.strip()

        if trace:
            trace.update(output={"answer": answer, "steps": MAX_STEPS})
            langfuse.flush()

        return answer


if __name__ == "__main__":
    from src.llm.openrouter_llm import OpenRouterLLM

    llm = OpenRouterLLM()
    agent = ReActAgent(llm)

    while True:
        question = input("\nВопрос: ")
        if question.lower() in ("exit", "quit", "q"):
            break
        answer = agent.run(question, verbose=True)
        print(f"\n{'='*50}")
        print(f"Ответ: {answer}")
