import json
import os
import time
import random

import requests
from dotenv import load_dotenv

from src.pipeline.chunker import Chunker

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.6-plus:free"
# MODEL = "google/gemma-3n-e2b-it:free"

DATA_DIR = "data/cleaned"
OUTPUT_PATH = "data/qa_pairs.jsonl"

SYSTEM_PROMPT = """Ты — эксперт по произведениям Толкина. По данному отрывку из книги сгенерируй 3 пары вопрос-ответ.

Требования:
- Вопросы должны быть разнообразными: фактические, аналитические, описательные
- Ответы должны быть написаны в стилистике Толкина, опираясь строго на текст отрывка
- Ответы должны быть развёрнутыми (2-4 предложения)

Верни результат строго в JSON формате (без маркдауна, без ```):
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."}
]"""


MAX_RETRIES = 5


def call_openrouter(chunk_text: str) -> list[dict] | None:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Отрывок:\n{chunk_text}"},
        ],
        "temperature": 0.7,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limit, жду {wait}с...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"  Ошибка: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 3)
            continue
    return None


def generate(max_chunks: int | None = None):
    chunker = Chunker(chunk_size=1500, chunk_overlap=200)

    all_chunks = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.splitext(filename)[0]
        all_chunks.extend(chunker.chunk(text, source=source))

    random.shuffle(all_chunks)
    if max_chunks:
        all_chunks = all_chunks[:max_chunks]
    print(f"Всего чанков: {len(all_chunks)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    generated = 0
    errors = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for i, chunk in enumerate(all_chunks):
            print(f"[{i+1}/{len(all_chunks)}] {chunk.source} chunk {chunk.index}...")
            pairs = call_openrouter(chunk.text)
            if not pairs:
                errors += 1
                continue

            for pair in pairs:
                record = {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "context": chunk.text,
                    "source": chunk.source,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated += 1

            print(f"  +{len(pairs)} пар (всего: {generated})")
            time.sleep(3)

    print(f"\nГотово. Сгенерировано {generated} пар, ошибок: {errors} → {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=None, help="Макс. кол-во чанков")
    args = parser.parse_args()
    generate(max_chunks=args.max_chunks)
