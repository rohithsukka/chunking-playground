"""
Naive Chunker
Chunks text by splitting on line breaks.
"""

import time
# pyrefly: ignore [missing-import]
from base_chunker import BaseChunker, ChunkResult


class NaiveChunker(BaseChunker):
    """Chunks text using line-break separation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "Segments text by line breaks – the most basic chunking method"
        self.use_cases = [
            "Note documents with structured line content",
            "Project lists and FAQ documents",
            "Chat logs and transcripts",
            "Content where each line contains complete semantic units",
        ]

    def chunk(self, text: str) -> ChunkResult:
        """
        Chunk text by splitting on newline characters.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        self.validate_text(text)

        chunks = [line.strip() for line in text.split('\n') if line.strip()]

        return ChunkResult(chunks, {'method': 'line_break_separation'})