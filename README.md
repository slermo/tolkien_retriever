# Проект

QWEN модель с дообученными LORA весами на книгах

```bash
poetry install
```

```bash
python -m ipykernel install --user --name=tolkien-rag --display-name "tolkien-rag"
```

Дообучение lora

```bash
python -m src.llm.train
```

Тест RAG

```bash
python -m src.llm.generate
```
