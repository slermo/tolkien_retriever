from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.7) -> tuple[str, dict]:
        """Send messages and return (response_text, usage_metadata)."""
        ...
