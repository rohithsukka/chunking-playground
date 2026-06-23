"""
Semantic Chunker
Chunks text using sentence embeddings and K-Means clustering.
"""

import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# pyrefly: ignore [missing-import]
from base_chunker import BaseChunker, ChunkResult


class SemanticChunker(BaseChunker):
    """Chunks text by grouping semantically similar sentences via K-Means clustering."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", n_clusters: int = 3, **kwargs):
        """
        Args:
            embedding_model: Name of the sentence-transformers model to use
            n_clusters:      Number of semantic clusters (chunks) to produce
        """
        super().__init__(**kwargs)
        self.model = SentenceTransformer(embedding_model)
        self.n_clusters = n_clusters
        self.description = "Groups semantically similar sentences into chunks using embeddings and K-Means"
        self.use_cases = [
            "Topic-coherent chunking for RAG pipelines",
            "Long documents with mixed topics",
            "Research papers and technical reports",
            "Any text where semantic cohesion per chunk matters",
        ]

    def chunk(self, text: str) -> ChunkResult:
        """
        Chunk text by clustering sentences with their embeddings.

        Args:
            text: Input text to be chunked

        Returns:
            ChunkResult: Object containing chunks and metadata
        """
        self.validate_text(text)

        sentences = [
            sentence.strip()
            for sentence in re.split(r'(?<=[.!?])\s+', text.strip())
            if sentence.strip()
        ]

        if len(sentences) <= self.n_clusters:
            return ChunkResult(sentences, {'method': 'semantic_clustering', 'note': 'too_few_sentences'})

        embeddings = self.model.encode(sentences, show_progress_bar=False)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

        clusters: dict = {}
        for sentence, label in zip(sentences, labels):
            clusters.setdefault(label, []).append(sentence)

        chunks = [" ".join(cluster) for cluster in clusters.values()]

        return ChunkResult(
            chunks,
            {
                'method': 'semantic_clustering',
                'n_clusters': self.n_clusters,
                'embedding_model': self.model.get_embedding_dimension(),
            }
        )