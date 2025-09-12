import os
import re
from typing import List, Dict

from bs4 import BeautifulSoup

PROCESSED_DATA_DIR = "data/processed"


class TextChunk:
    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata


class TextProcessor:
    def __init__(self):
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    def extract_main_content(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main")
        if main:
            return main.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[TextChunk]:
        paragraphs = text.split("\n")
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < max_chunk_size:
                current += para + "\n"
            else:
                if current:
                    chunks.append(TextChunk(current.strip(), {}))
                current = para + "\n"
        if current:
            chunks.append(TextChunk(current.strip(), {}))
        return chunks

    def save_chunks(self, chunks: List[TextChunk], base_filename: str):
        for i, chunk in enumerate(chunks):
            path = os.path.join(PROCESSED_DATA_DIR, f"{base_filename}_chunk_{i}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(chunk.text)
