"""
Base Chunker Class
Abstract base class for all chunking techniques.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time


class ChunkResult:
    """Data class to hold chunk results with metadata."""

    def __init__(self, chunks: List[str], metadata: Optional[Dict[str, Any]] = None):
        self.chunks = chunks
        self.metadata = metadata or {}
        self.chunk_count = len(chunks)
        self.total_length = sum(len(chunk) for chunk in chunks)
        self.avg_chunk_size = self.total_length / self.chunk_count if self.chunk_count > 0 else 0


class BaseChunker(ABC):
    """Abstract base class for all chunking techniques."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.name = self.__class__.__name__
        self.description = ""
        self.use_cases = []

    @abstractmethod
    def chunk(self, text: str) -> ChunkResult:
        """
        Abstract method to chunk text.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        pass

    def validate_text(self, text: str) -> bool:
        """
        Validate input text.

        Args:
            text: Input text to validate

        Returns:
            bool: True if text is valid
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        return True

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the chunking technique.

        Returns:
            Dict containing technique information
        """
        return {
            'name': self.name,
            'description': self.description,
            'use_cases': self.use_cases,
            'parameters': self.params,
        }

    def chunk_with_timing(self, text: str) -> ChunkResult:
        """
        Chunk text and measure processing time.

        Args:
            text: Text to be chunked

        Returns:
            ChunkResult: Object containing chunks, metadata, and timing info
        """
        start_time = time.time()
        result = self.chunk(text)
        end_time = time.time()

        result.metadata['processing_time'] = end_time - start_time
        result.metadata['technique'] = self.name
        result.metadata['parameters'] = self.params

        return result
