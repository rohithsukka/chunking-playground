"""
Paragraph Based Chunker
Chunks text by splitting on double newlines (blank lines between paragraphs).
"""

# pyrefly: ignore [missing-import]
from base_chunker import BaseChunker, ChunkResult


class ParagraphChunker(BaseChunker):
    """Chunks text by splitting on blank lines between paragraphs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "Splits text on double newlines to preserve paragraph boundaries"
        self.use_cases = [
            "Articles, blog posts, and essays",
            "Books and long-form prose",
            "Documentation with paragraph structure",
            "Any text where paragraphs are the natural semantic unit",
        ]

    def chunk(self, text: str) -> ChunkResult:
        """
        Chunk text by splitting on double newline characters.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        self.validate_text(text)

        chunks = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

        return ChunkResult(chunks, {'method': 'paragraph_separation'})
