from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    source: str
    index: int


class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        pieces = self._splitter.split_text(text)
        return [
            Chunk(text=t, source=source, index=i)
            for i, t in enumerate(pieces)
        ]
