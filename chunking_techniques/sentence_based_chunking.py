"""
Sentence Based Chunker
Chunks text by splitting on sentence-ending punctuation.
"""

import re

# pyrefly: ignore [missing-import]
from base_chunker import BaseChunker, ChunkResult


class SentenceChunker(BaseChunker):
    """Chunks text by splitting on sentence-ending punctuation (. ! ?)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "Splits text at sentence boundaries using punctuation rules"
        self.use_cases = [
            "News articles and factual documents",
            "Q&A datasets and chatbot training data",
            "Legal and medical text requiring sentence-level precision",
            "Any text where sentences are the natural semantic unit",
        ]

    def chunk(self, text: str) -> ChunkResult:
        """
        Chunk text by splitting at sentence-ending punctuation.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        self.validate_text(text)

        chunks = [
            sentence.strip()
            for sentence in re.split(r'(?<=[.!?])\s+', text.strip())
            if sentence.strip()
        ]

        return ChunkResult(chunks, {'method': 'sentence_splitting'})
