from src.llm.base import BaseLLM
from src.llm.local.model import load_for_inference


class LocalLLM(BaseLLM):
    def __init__(
        self,
        adapter_path: str = "tolkien_lora_adapter",
        max_new_tokens: int = 512,
    ):
        self.model, self.tokenizer = load_for_inference(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.model_name = adapter_path

    @property
    def model_id(self) -> str:
        return self.model_name

    def chat(self, messages: list[dict], temperature: float = 0.7) -> tuple[str, dict]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = input_ids.shape[1]

        output = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )
        generated = output[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text, {
            "input_tokens": int(input_len),
            "output_tokens": int(generated.shape[0]),
            "model": self.model_name,
        }
