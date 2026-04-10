from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    return model, tokenizer


def attach_lora(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
    )
    return model


def load_for_inference(adapter_path: str = "tolkien_lora_adapter"):
    model, tokenizer = load_base_model()
    model.load_adapter(adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer
