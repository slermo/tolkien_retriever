import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

from src.llm.model import load_base_model, attach_lora, MAX_SEQ_LENGTH

DATA_DIR = "data/cleaned"
OUTPUT_DIR = "tolkien_lora"
ADAPTER_DIR = "tolkien_lora_adapter"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def build_dataset(tokenizer) -> Dataset:
    all_chunks = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens) - CHUNK_SIZE, CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = tokens[i : i + CHUNK_SIZE]
            all_chunks.append(tokenizer.decode(chunk))

    print(f"Всего чанков: {len(all_chunks)}")
    return Dataset.from_dict({"text": all_chunks})


def train():
    model, tokenizer = load_base_model()
    model = attach_lora(model)
    FastLanguageModel.for_training(model)

    dataset = build_dataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            output_dir=OUTPUT_DIR,
            save_strategy="epoch",
        ),
    )
    trainer.train()
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Адаптер сохранён в {ADAPTER_DIR}")


if __name__ == "__main__":
    train()
