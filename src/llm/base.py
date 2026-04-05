from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send messages and return assistant response text."""
        ...
