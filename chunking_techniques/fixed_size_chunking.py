"""
Fixed Size Chunker
Chunks text into fixed-size windows measured in words.
"""

# pyrefly: ignore [missing-import]
from base_chunker import BaseChunker, ChunkResult


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size windows measured in words."""

    def __init__(self, chunk_size: int = 100, **kwargs):
        """
        Args:
            chunk_size: Number of words per chunk
        """
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.description = "Splits text into equal-sized word-count windows"
        self.use_cases = [
            "Long documents requiring uniform chunk sizes",
            "Token-budget constrained LLM pipelines",
            "Baseline chunking for retrieval benchmarks",
            "Any text where consistent chunk size matters more than semantics",
        ]

    def chunk(self, text: str) -> ChunkResult:
        """
        Chunk text into fixed-size word windows.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        self.validate_text(text)

        words = text.split()
        chunks = [
            " ".join(words[i:i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]

        return ChunkResult(chunks, {'method': 'fixed_size', 'chunk_size': self.chunk_size})
