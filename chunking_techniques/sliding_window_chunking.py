"""
Sliding Window Chunker
Chunks text using a sliding word window with configurable overlap.
"""

# pyrefly: ignore [missing-import]
from base_chunker import BaseChunker, ChunkResult


class SlidingWindowChunker(BaseChunker):
    """Chunks text using a sliding word window with configurable overlap."""

    def __init__(self, chunk_size: int = 100, overlap: int = 20, **kwargs):
        """
        Args:
            chunk_size: Number of words per chunk
            overlap:    Number of words shared between consecutive chunks
        """
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.description = "Splits text into overlapping word windows to preserve context at boundaries"
        self.use_cases = [
            "Retrieval-augmented generation (RAG) pipelines",
            "Long documents where cross-chunk context matters",
            "Semantic search over dense technical content",
            "Any task where boundary information must not be lost",
        ]

    def chunk(self, text: str) -> ChunkResult:
        """
        Chunk text using a sliding window with overlap.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        self.validate_text(text)

        words = text.split()
        step = self.chunk_size - self.overlap
        chunks = []

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))

        return ChunkResult(
            chunks,
            {
                'method': 'sliding_window',
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
            }
        )
