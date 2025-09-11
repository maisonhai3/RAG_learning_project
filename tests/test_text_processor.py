import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.crawler.text_processor import TextProcessor, TextChunk


def test_text_processor_chunking():
    processor = TextProcessor()
    sample_text = "Section 1\nThis is a test.\nSection 2\nAnother test."
    chunks = processor.chunk_text(sample_text, max_chunk_size=20)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert len(chunk.text) <= 20
