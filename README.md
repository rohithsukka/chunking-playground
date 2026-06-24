# Chunking Playground

Welcome to the Chunking Playground, an interactive utility and library for testing, visualizing, and comparing different text chunking techniques. In natural language processing (NLP) and Retrieval-Augmented Generation (RAG) pipelines, dividing text into optimal segments (chunks) is a critical preprocessing step. This project provides a collection of rule-based and AI-powered chunking strategies, paired with a web application to compare their behaviors.

<img width="1524" height="665" alt="image" src="https://github.com/user-attachments/assets/5454ff76-6735-4cbe-8609-f58290290829" />

## Try it here

### (https://chunking-playground-a6wro26ovukk2ah9bg8ctg.streamlit.app/)

## Project Overview

The project consists of:
1. **Core Chunkers**: A modular set of text chunking algorithms implemented under the `chunking_techniques` directory. All chunkers inherit from a standardized abstract base class.
2. **Streamlit App**: An interactive web-based interface (`streamlit_app.py`) that allows you to paste text or use preloaded samples, tune chunker parameters in real time, visualize the boundaries of the resulting chunks, and view performance metrics (processing time, chunk count, average chunk size).

## Getting Started

### Prerequisites

* Python (version 3.13 or higher is recommended)

### Installation

The project uses the `uv` package manager for dependency resolution, but you can also install using standard `pip`.

#### Option A: Using uv
To install the environment and synchronize dependencies:
```bash
uv sync
```

#### Option B: Using pip
To install packages directly from the requirements file:
```bash
pip install -r requirements.txt
```

### Running the Interactive App

Launch the Streamlit dashboard to interactively test the chunking strategies:
```bash
streamlit run streamlit_app.py
```
Open the provided local URL (typically `http://localhost:8502`) in your browser to view the application.

---

## Chunking Methods Reference

This project implements seven chunking methods, split into Rule-Based and AI-Powered categories.

### Rule-Based Chunkers

#### 1. Naive Chunker
* **Implementation file**: [naive_chunker.py](file:///e:/Technical/chunking-playgorund/chunking_techniques/naive_chunker.py)
* **Class**: `NaiveChunker`
* **Splitting Strategy**: Segments text by single line breaks (`\n`).
* **Parameters**: None.
* **Pros & Cons**: Extremely fast and simple. However, it falls apart on prose or narrative articles where paragraphs are formatted as multi-line text blocks.
* **Ideal Use Cases**: Structured line-by-line documents, note files, lists, FAQ documents, chat logs, and transcripts.

#### 2. Fixed Size Chunker
* **Implementation file**: [fixed_size_chunking.py](file:///e:/Technical/chunking-playgorund/chunking_techniques/fixed_size_chunking.py)
* **Class**: `FixedSizeChunker`
* **Splitting Strategy**: Splits text into equal-sized windows measured in word count.
* **Parameters**:
  * `chunk_size` (int, default: 100): The maximum number of words per chunk.
* **Pros & Cons**: Predictable and fast. The downside is that it has no awareness of linguistic structure, meaning it can cut directly through the middle of a sentence.
* **Ideal Use Cases**: Token-budget constrained LLM pipelines, baseline chunking benchmarks, or any pipeline where consistent chunk size matters more than semantics.

#### 3. Sliding Window Chunker
* **Implementation file**: [sliding_window_chunking.py](file:///e:/Technical/chunking-playgorund/chunking_techniques/sliding_window_chunking.py)
* **Class**: `SlidingWindowChunker`
* **Splitting Strategy**: Splits text using a sliding window of words with a configurable number of overlapping words shared between consecutive chunks.
* **Parameters**:
  * `chunk_size` (int, default: 100): Word count per chunk.
  * `overlap` (int, default: 20): Number of words shared between adjacent chunks.
* **Pros & Cons**: Prevents contextual loss at boundaries by repeating words. It is slightly more resource-intensive than fixed-size chunking but significantly improves search retrieval quality.
* **Ideal Use Cases**: Retrieval-Augmented Generation (RAG) pipelines, semantic search over dense technical documentation, and tasks where boundary information must not be lost.

#### 4. Sentence-Based Chunker
* **Implementation file**: [sentence_based_chunking.py](file:///e:/Technical/chunking-playgorund/chunking_techniques/sentence_based_chunking.py)
* **Class**: `SentenceChunker`
* **Splitting Strategy**: Splits text on sentence-ending punctuation (periods, exclamation marks, or question marks) using regular expression rules.
* **Parameters**: None.
* **Pros & Cons**: Respects grammar boundaries, ensuring each chunk contains complete sentences. It results in a larger number of smaller, highly cohesive chunks.
* **Ideal Use Cases**: News articles, factual documents, Q&A datasets, and legal or medical texts requiring high grammatical precision.

#### 5. Paragraph-Based Chunker
* **Implementation file**: [paragraph_based_Chunking.py](file:///e:/Technical/chunking-playgorund/chunking_techniques/paragraph_based_Chunking.py)
* **Class**: `ParagraphChunker`
* **Splitting Strategy**: Splits text on double newlines (`\n\n`), preserving natural paragraph boundaries.
* **Parameters**: None.
* **Pros & Cons**: Highly coherent as it groups related thoughts together. It depends entirely on the formatting of the source document having blank lines between paragraphs.
* **Ideal Use Cases**: Articles, blog posts, essays, books, documentation, and prose where paragraphs represent natural semantic divisions.

#### 6. Page-Based Chunker
* **Implementation file**: [page_based_chunking.py](file:///e:/Technical/chunking-playgorund/chunking_techniques/page_based_chunking.py)
* **Class**: `PageChunker`
* **Splitting Strategy**: Splits text on an explicit, configurable page-break separator.
* **Parameters**:
  * `page_separator` (str, default: `"<PAGE_BREAK>"`): The token representing a page division.
* **Pros & Cons**: Highly precise for documents exported with embedded layout structure. If the separator is not present in the document, it will fallback to treating the entire text as a single chunk.
* **Ideal Use Cases**: PDFs, books, and reports that are programmatically pre-processed to include page-break markers.

---

### AI-Powered Chunkers

#### 7. Semantic Chunker
* **Implementation file**: [semantic_chunker.py](file:///e:/Technical/chunking-playgorund/chunking_techniques/semantic_chunker.py)
* **Class**: `SemanticChunker`
* **Splitting Strategy**: Splits text into sentences, generates embeddings for each sentence using a sentence-transformer model, and runs K-Means clustering on the embeddings to group semantically similar sentences into chunks.
* **Parameters**:
  * `embedding_model` (str, default: `"all-MiniLM-L6-v2"`): The Hugging Face sentence-transformers model.
  * `n_clusters` (int, default: 3): The number of semantic clusters (chunks) to generate.
* **Pros & Cons**: The only strategy that leverages text meaning rather than layout formatting. It can group sentences together even if they are physically far apart in the document. The drawback is that it requires machine learning inference (embeddings and clustering), which is slower and more resource-intensive.
* **Ideal Use Cases**: Topic-coherent chunking for RAG pipelines, long mixed-topic documents, research papers, and technical reports.

---

## Technical Architecture

### Base Chunker and Results Interface

All chunkers inherit from `BaseChunker` defined in `chunking_techniques/base_chunker.py`. This class provides core utilities, basic validation, and timing capture.

#### Base Chunker API:
* `chunk(text: str) -> ChunkResult`: Abstract method that concrete classes must implement.
* `chunk_with_timing(text: str) -> ChunkResult`: Automatically wraps the `chunk()` method and records processing execution time in the metadata.
* `validate_text(text: str) -> bool`: Validates that the input is a non-empty string.
* `get_info() -> dict`: Returns the chunker's metadata, description, parameters, and configured use cases.

#### ChunkResult Class:
An instance returned by chunkers containing the following attributes:
* `chunks` (List[str]): List of output text strings.
* `chunk_count` (int): Total count of chunks.
* `total_length` (int): Combined character length.
* `avg_chunk_size` (float): Average character length per chunk.
* `metadata` (dict): Dictionary with technique-specific settings and execution time.

---

## Programmatic Usage Example

You can import and use these chunkers directly in your Python projects. Below is an example using `SlidingWindowChunker` and `SemanticChunker`:

```python
from chunking_techniques.sliding_window_chunking import SlidingWindowChunker
from chunking_techniques.semantic_chunker import SemanticChunker

text = """
Retrieval-Augmented Generation is a powerful pattern. It grounds LLM responses using external documents.
We split the document into pieces, store them in a vector database, and retrieve them on demand.
This improves accuracy and prevents hallucinations.
"""

# Example 1: Sliding Window Chunker
window_chunker = SlidingWindowChunker(chunk_size=10, overlap=3)
result_window = window_chunker.chunk_with_timing(text)

print(f"Technique: {result_window.metadata['technique']}")
print(f"Chunks Count: {result_window.chunk_count}")
print(f"Processing Time: {result_window.metadata['processing_time']:.5f} seconds")
for i, chunk in enumerate(result_window.chunks):
    print(f"Chunk {i+1}: {chunk}")

# Example 2: Semantic Chunker
semantic_chunker = SemanticChunker(n_clusters=2)
result_semantic = semantic_chunker.chunk_with_timing(text)

print(f"\nTechnique: {result_semantic.metadata['technique']}")
print(f"Chunks Count: {result_semantic.chunk_count}")
for i, chunk in enumerate(result_semantic.chunks):
    print(f"Chunk {i+1}: {chunk}")
```
