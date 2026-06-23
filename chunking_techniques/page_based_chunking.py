"""
Page Based Chunker
Chunks text by splitting on a page-break separator.
"""

# pyrefly: ignore [missing-import]
from base_chunker import BaseChunker, ChunkResult


class PageChunker(BaseChunker):
    """Chunks text by splitting on a configurable page-break separator."""

    def __init__(self, page_separator: str = "<PAGE_BREAK>", **kwargs):
        super().__init__(**kwargs)
        self.page_separator = page_separator
        self.description = "Splits text on explicit page-break markers"
        self.use_cases = [
            "PDFs or documents exported with page-break tokens",
            "Reports and books with clearly delimited pages",
            "Any text with consistent structural page markers",
        ]

    def chunk(self, text: str) -> ChunkResult:
        """
        Chunk text by splitting on the page-break separator.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        self.validate_text(text)

        chunks = [page.strip() for page in text.split(self.page_separator) if page.strip()]

        return ChunkResult(chunks, {'method': 'page_break_separation', 'separator': self.page_separator})
