import os
import requests
from dotenv import load_dotenv
from src.llm.base import BaseLLM

load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")


class VllmLLM(BaseLLM):
    def __init__(
        self,
        model: str = "unsloth/Qwen2.5-7B-Instruct",
        base_url: str | None = None,
    ):
        self.model = model
        self.base_url = (base_url or VLLM_BASE_URL).rstrip("/")

    @property
    def model_id(self) -> str:
        return self.model

    def chat(self, messages: list[dict], temperature: float = 0.7) -> tuple[str, dict]:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 512,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return content, {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model": self.model,
        }
