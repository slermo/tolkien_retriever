import os
import time
import requests
from dotenv import load_dotenv
from src.llm.base import BaseLLM

load_dotenv()

MAX_RETRIES = 5


class OpenRouterLLM(BaseLLM):
    def __init__(self, model: str = "qwen/qwen3.6-plus:free"):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  Rate limit, жду {wait}с...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"OpenRouter: {MAX_RETRIES} попыток исчерпано")
