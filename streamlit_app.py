"""
Chunking Playground — Streamlit App
Interactive frontend for all text chunking techniques.
Author: Rohith Sukka
"""

import streamlit as st
import sys
import os
import json
import importlib

# ── Path setup ─────────────────────────────────────────────────────────────────
CHUNKERS_DIR = os.path.join(os.path.dirname(__file__), "chunking_techniques")
sys.path.insert(0, CHUNKERS_DIR)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chunking Playground",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sample texts ───────────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Custom": "",
    "Technical Article": """\
What is Agentic AI?
Agentic AI gives autonomous agents the ability to plan, make decisions, and adapt as they pursue goals. But agentic potential only becomes reliable business execution when agents operate inside an orchestration layer that coordinates them with robots, APIs, documents, data, and people.

Agentic AI systems can:
Understand goals and make decisions based on context and available data.
Break down complex tasks into manageable step-by-step plans.
Use tools, applications, and external systems to complete work.
Maintain memory of relevant information and ongoing situations.
Collaborate with people or other agents when needed.
Learn from results to improve performance over time.

How agentic AI works
An agentic AI system is a continuous loop of perception, planning, action, and learning.

Perception
An AI-powered agent collects information from documents, applications, data sources, APIs, sensors, or other systems. It interprets this information to understand the current context.

Reasoning and planning
Based on what it perceives, the agent evaluates options and determines the steps required to achieve a goal.

Taking action
The agent performs tasks through applications, APIs, robots, or other agents. These actions can include updating records, generating content, retrieving information, or initiating workflow steps.

Reflecting and learning
After each action, the agent evaluates the outcome. It uses feedback and short- or long-term memory to refine its approach and improve future decisions.\
""",
    "Business Report": """\
Q2 2024 Business Performance Report

Executive Summary
The second quarter of 2024 demonstrated strong growth across all business units. Revenue increased by 23% year-over-year, driven primarily by expansion in the enterprise segment and successful launch of our new product line.

Financial Highlights
Total revenue reached $142M, up from $115M in Q2 2023. Gross margin improved to 68%, reflecting operational efficiencies and favorable product mix. Operating expenses were well-controlled at $87M, resulting in operating income of $55M.

Market Expansion
We successfully entered three new geographic markets: Southeast Asia, Eastern Europe, and Latin America. Customer acquisition in these regions exceeded initial projections by 40%, suggesting strong product-market fit.

Product Development
The engineering team shipped 47 new features and resolved 312 customer-reported issues. The new AI-powered analytics dashboard received exceptional customer feedback with a 4.8/5 satisfaction score.

Outlook
For Q3 2024, we project revenue between $155M and $165M. Key growth drivers will include the enterprise contract renewals scheduled for July and August, continued expansion in new markets, and the upcoming release of version 3.0 of our core platform.\
""",
    "Code Documentation": """\
BaseChunker Class
Abstract base class for all chunking techniques.

Overview
The BaseChunker provides a standardized interface for all text chunking implementations. Every chunker inherits from this class and must implement the chunk() method.

Installation
Install the package using pip: pip install chunking-playground

Quick Start
Import the desired chunker and instantiate it with optional parameters.
Call the chunk() method with your input text to get a ChunkResult object.
Use chunk_with_timing() to automatically measure processing time.

ChunkResult Properties
chunks: List of string chunks produced by the chunker.
chunk_count: Total number of chunks produced.
total_length: Total character count across all chunks.
avg_chunk_size: Average character count per chunk.
metadata: Dictionary containing method name and technique-specific parameters.

BaseChunker Methods
chunk(text): Abstract method that must be implemented by subclasses.
validate_text(text): Validates input is a non-empty string, returns True or raises an error.
get_info(): Returns a dictionary with name, description, use_cases, and parameters.
chunk_with_timing(text): Wraps chunk() and injects processing_time, technique, and parameters into metadata.

Error Handling
TypeError is raised when the input is not a string.
ValueError is raised when the input text is empty or contains only whitespace.\
""",
    "Mixed Content": """\
Introduction to RAG Systems

What is Retrieval-Augmented Generation?
RAG combines the power of large language models with the ability to retrieve relevant information from external knowledge bases. This approach grounds model responses in factual, up-to-date information.

<PAGE_BREAK>

Core Components of RAG

The Document Store
Documents are ingested, chunked, and stored as vector embeddings. Common storage solutions include Pinecone, Weaviate, Chroma, and FAISS. Each chunk is encoded using an embedding model such as text-embedding-3-small or all-MiniLM-L6-v2.

The Retriever
When a query arrives, it is embedded using the same model. Similarity search (cosine or dot-product) finds the top-k most relevant chunks. These chunks form the context window passed to the LLM.

<PAGE_BREAK>

Chunking Strategies for RAG

Choosing the right chunking strategy is critical for retrieval quality. Fixed-size chunking is simple but may break semantic units. Sentence-based chunking preserves grammatical structure. Sliding-window chunking maintains context across chunk boundaries. Semantic chunking groups thematically similar content together.

Evaluation Metrics
Retrieval recall measures how often the relevant chunk is retrieved. Answer faithfulness measures whether the LLM answer is grounded in the retrieved context. Context precision measures the relevance of retrieved chunks to the query.\
""",
}

SAMPLE_DESCRIPTIONS = {
    "Custom": "Write or paste any text to see how it gets split.",
    "Technical Article": "Agentic AI overview — good for testing paragraph and sentence chunkers.",
    "Business Report": "Business narrative — tests how chunkers handle headers and lists.",
    "Code Documentation": "Class reference with structured sections — great for page-based chunking.",
    "Mixed Content": "Contains embedded &lt;PAGE_BREAK&gt; markers — perfect for PageBased chunker.",
}

# ── Chunker registry ───────────────────────────────────────────────────────────
CHUNKER_REGISTRY = {
    "Basic": {
        "Naive":          {"module": "naive_chunker",           "class": "NaiveChunker",         "params": []},
        "FixedSize":      {"module": "fixed_size_chunking",     "class": "FixedSizeChunker",     "params": ["chunk_size"]},
        "SlidingWindow":  {"module": "sliding_window_chunking", "class": "SlidingWindowChunker", "params": ["chunk_size", "overlap"]},
        "SentenceBased":  {"module": "sentence_based_chunking", "class": "SentenceChunker",      "params": []},
        "ParagraphBased": {"module": "paragraph_based_Chunking","class": "ParagraphChunker",     "params": []},
        "PageBased":      {"module": "page_based_chunking",     "class": "PageChunker",          "params": ["page_separator"]},
    },
    "AI-Powered": {
        "Semantic": {"module": "semantic_chunker", "class": "SemanticChunker", "params": ["n_clusters"]},
    },
}

# Flat lookup used throughout
ALL_CHUNKERS = {k: v for cat in CHUNKER_REGISTRY.values() for k, v in cat.items()}

# ── Rich per-technique educational details ─────────────────────────────────────
TECHNIQUE_DETAILS = {
    "Naive": {
        "emoji": "🟦",
        "split_on": "Single newline (\\n)",
        "logic": "text.split('\\n') — each non-empty line becomes one chunk. The simplest possible split.",
        "parameters": "None — no configuration needed.",
        "speed": "⚡ Instant",
        "speed_color": "#16a34a",
        "understands_meaning": False,
        "learner_tip": "Best for structured line-by-line content (chat logs, lists, FAQs). Falls apart on prose because a paragraph is many lines.",
    },
    "FixedSize": {
        "emoji": "🟩",
        "split_on": "Word count — every N words",
        "logic": "Split all words into a flat list, then take slices of size chunk_size. No overlap — chunks never share words.",
        "parameters": "chunk_size (words per chunk, default 100).",
        "speed": "⚡ Instant",
        "speed_color": "#16a34a",
        "understands_meaning": False,
        "learner_tip": "Predictable and fast. The downside: it can cut right through a sentence. Use when uniform size matters more than readability.",
    },
    "SlidingWindow": {
        "emoji": "🟨",
        "split_on": "Word count with overlap between consecutive chunks",
        "logic": "step = chunk_size - overlap. Each chunk starts 'step' words after the previous one. The last 'overlap' words of chunk N are also the first 'overlap' words of chunk N+1.",
        "parameters": "chunk_size (default 100) · overlap (default 20 words shared between neighbours).",
        "speed": "⚡ Instant",
        "speed_color": "#16a34a",
        "understands_meaning": False,
        "learner_tip": "The go-to upgrade from FixedSize for RAG. Overlapping words mean a sentence that sits at a boundary appears in two chunks, so retrieval never misses it.",
    },
    "SentenceBased": {
        "emoji": "🟧",
        "split_on": "Sentence-ending punctuation (. ! ?)",
        "logic": "regex re.split(r'(?<=[.!?])\\s+', text) — splits immediately after a full stop, exclamation mark, or question mark followed by whitespace.",
        "parameters": "None — the punctuation rules are fixed.",
        "speed": "⚡ Instant",
        "speed_color": "#16a34a",
        "understands_meaning": False,
        "learner_tip": "Respects grammar — every chunk is a complete sentence. Chunks are small and numerous. Great for Q&A datasets or legal text where precision matters.",
    },
    "ParagraphBased": {
        "emoji": "🟥",
        "split_on": "Double newline (\\n\\n) — blank lines between paragraphs",
        "logic": "text.split('\\n\\n') — everything between blank lines is one chunk. Paragraphs can be very long.",
        "parameters": "None — depends entirely on blank lines being present in the text.",
        "speed": "⚡ Instant",
        "speed_color": "#16a34a",
        "understands_meaning": False,
        "learner_tip": "Often confused with Naive. Key difference: Naive splits on every \\n (line), Paragraph splits only on \\n\\n (blank lines). Produces much larger, more coherent chunks.",
    },
    "PageBased": {
        "emoji": "🟫",
        "split_on": "Custom page-break marker (e.g. <PAGE_BREAK>)",
        "logic": "text.split(page_separator) — everything between marker occurrences becomes one chunk. If the marker is absent, the whole text is one chunk.",
        "parameters": "page_separator — the exact string to split on (default <PAGE_BREAK>).",
        "speed": "⚡ Instant",
        "speed_color": "#16a34a",
        "understands_meaning": False,
        "learner_tip": "Only works if your text has explicit markers — common in PDFs converted to text. Try the 'Mixed Content' sample text which contains <PAGE_BREAK> markers.",
    },
    "Semantic": {
        "emoji": "🤖",
        "split_on": "Topic similarity — not text structure",
        "logic": "1) Split into sentences. 2) Encode each sentence with the all-MiniLM-L6-v2 transformer model. 3) Run K-Means clustering on the embeddings. 4) Sentences in the same cluster form one chunk — regardless of where they appear in the text.",
        "parameters": "n_clusters — how many topic groups (chunks) to produce (default 3).",
        "speed": "🐢 Seconds (ML model)",
        "speed_color": "#d97706",
        "understands_meaning": True,
        "learner_tip": "The only technique that understands meaning. Two sentences about the same topic get grouped together even if they're paragraphs apart. Ideal for RAG over mixed-topic documents.",
    },
}

CHUNK_PALETTE = [
    ("#6366f1", "rgba(99,102,241,0.09)"),
    ("#0ea5e9", "rgba(14,165,233,0.09)"),
    ("#10b981", "rgba(16,185,129,0.09)"),
    ("#f59e0b", "rgba(245,158,11,0.08)"),
    ("#ef4444", "rgba(239,68,68,0.08)"),
    ("#ec4899", "rgba(236,72,153,0.08)"),
    ("#8b5cf6", "rgba(139,92,246,0.09)"),
    ("#14b8a6", "rgba(20,184,166,0.09)"),
]


# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: #f8fafc !important;
    color: #1e293b !important;
}
[data-testid="stAppViewContainer"] > .main { background: transparent; }
[data-testid="block-container"] { padding-top: 0 !important; }

/* ── Fix Streamlit top header background to match app theme ── */
[data-testid="stHeader"] {
    background-color: #f8fafc !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
/* Hide only the colored decoration stripe at the very top */
#stDecoration { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* ── Creator card ── */
.creator-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 20px;
    color: rgba(255,255,255,0.92);
    font-size: 0.82rem;
    line-height: 1.7;
}
.creator-card strong { font-size: 1rem; color: #fff; display: block; margin-bottom: 2px; font-weight: 700; }
.creator-card .role { font-size: 0.76rem; color: rgba(255,255,255,0.72); display: block; margin-bottom: 10px; }
.creator-card .links { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
.creator-card a {
    background: rgba(255,255,255,0.2);
    color: #fff;
    text-decoration: none;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 500;
    transition: background 0.2s;
}
.creator-card a:hover { background: rgba(255,255,255,0.38); }

/* ── Sidebar how-to guide ── */
.how-to-card {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 16px;
}
.how-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #4338ca;
    margin-bottom: 10px;
}
.how-step { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 7px; font-size: 0.83rem; color: #3730a3; line-height: 1.4; }
.sn {
    width: 20px; height: 20px;
    background: #6366f1;
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 800;
    flex-shrink: 0; margin-top: 1px;
}

/* ── Sidebar footer ── */
.footer-badge {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.74rem;
    color: #94a3b8;
    line-height: 1.7;
}
.footer-badge b { color: #6366f1; }

/* ── Page header ── */
.cp-header {
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #0ea5e9 100%);
    border-radius: 18px;
    padding: 34px 44px 28px;
    margin-bottom: 32px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(99,102,241,0.2);
    position: relative;
    overflow: hidden;
}
.cp-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 15% 50%, rgba(255,255,255,0.12) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 50%, rgba(255,255,255,0.08) 0%, transparent 55%);
}
.cp-header h1 { font-size: 2.1rem; font-weight: 800; color: #fff; margin: 0 0 8px; letter-spacing: -0.5px; position: relative; }
.cp-header .subtitle { font-size: 0.97rem; color: rgba(255,255,255,0.87); margin: 0 0 4px; position: relative; }
.cp-header .tagline { font-size: 0.79rem; color: rgba(255,255,255,0.65); font-style: italic; position: relative; }

/* ── Step header (numbered badges) ── */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    margin-top: 8px;
}
.step-circle {
    width: 32px; height: 32px;
    background: #6366f1;
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 800;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(99,102,241,0.35);
}
.step-label { font-size: 1.05rem; font-weight: 700; color: #1e293b; }
.step-sub { font-size: 0.82rem; color: #64748b; font-weight: 400; margin-left: 4px; }

/* ── Section divider ── */
.section-divider { height: 1px; background: #e2e8f0; margin: 28px 0 26px; }

/* ── Sample description ── */
.sample-desc {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: #1d4ed8;
    line-height: 1.55;
    margin-top: 8px;
    margin-bottom: 4px;
}

/* ── Live counter ── */
.live-count { font-size: 0.77rem; color: #94a3b8; text-align: right; margin-top: 4px; }

/* ── Category label inside technique panel ── */
.cat-label {
    font-size: 0.73rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    margin: 14px 0 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.cat-label:first-child { margin-top: 0; }

/* ── Radio styled as pill buttons ── */
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    padding: 2px 0 4px !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    display: inline-flex !important;
    align-items: center !important;
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 7px 14px !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
    margin: 0 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: #eef2ff !important;
    border-color: #a5b4fc !important;
}
/* Selected pill — uses :has() (Chrome 105+, Firefox 121+, Safari 15.4+) */
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border-color: transparent !important;
    box-shadow: 0 3px 10px rgba(99,102,241,0.35) !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label p {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    margin: 0 !important;
    line-height: 1 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
}
/* Hide the radio circle indicator */
div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
/* Hide the top-level widget label (we use our own step labels) */
div[data-testid="stRadio"] > label { display: none !important; }

/* ── Last pill = Semantic (AI) — own row + amber gold accent ── */
div[data-testid="stRadio"] div[role="radiogroup"] label:last-child {
    flex-basis: 100% !important;          /* Force onto its own row */
    max-width: max-content !important;    /* Don't stretch to full width */
    margin-top: 28px !important;          /* Space above for the AI-Powered label */
    position: relative !important;
    border-color: #f59e0b !important;
    background: #fffbeb !important;
}
/* "🤖 AI-Powered" section label injected above the Semantic pill */
div[data-testid="stRadio"] div[role="radiogroup"] label:last-child::before {
    content: '🤖  AI-Powered';
    position: absolute;
    top: -21px;
    left: 0;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    white-space: nowrap;
    pointer-events: none;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:last-child p {
    color: #92400e !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:last-child:hover {
    background: #fef3c7 !important;
    border-color: #d97706 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:last-child:has(input:checked) {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    border-color: transparent !important;
    box-shadow: 0 3px 10px rgba(245,158,11,0.4) !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:last-child:has(input:checked) p {
    color: #ffffff !important;
}

/* ── "Currently selected" indicator ── */
.selected-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eef2ff;
    border: 1.5px solid #c7d2fe;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #4338ca;
    margin-bottom: 12px;
}

/* ── AI note ── */
.ai-note {
    background: #fefce8;
    border: 1px solid #fde047;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #713f12;
    margin-top: 8px;
    line-height: 1.5;
}

/* ── Param section ── */
.param-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #334155;
    margin-top: 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 5px;
}

/* ── Info card (rich detail card) ── */
.info-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* Header row: emoji + name + badges */
.ic-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
}
.ic-emoji {
    font-size: 2rem;
    line-height: 1;
    flex-shrink: 0;
}
.ic-name {
    font-size: 1.05rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 5px;
}
.ic-badges { display: flex; flex-wrap: wrap; gap: 5px; }

/* Badge styles */
.badge {
    display: inline-block;
    font-size: 0.71rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    border: 1px solid;
}
.badge-ai   { background: #f0fdf4; color: #16a34a; border-color: #86efac; }
.badge-rule { background: #f8fafc; color: #64748b;  border-color: #cbd5e1; }

/* Description */
.ic-desc {
    font-size: 0.87rem;
    color: #64748b;
    line-height: 1.65;
    margin-bottom: 14px;
    padding-bottom: 14px;
    border-bottom: 1px solid #f1f5f9;
}

/* Section rows */
.ic-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
    margin-top: 12px;
    margin-bottom: 4px;
}
.ic-value {
    font-size: 0.84rem;
    color: #334155;
    line-height: 1.6;
}
.ic-code {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 10px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.8rem;
    color: #4338ca;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}

/* When-to-use list */
.ic-list { margin: 4px 0 0; padding-left: 18px; }
.ic-list li { font-size: 0.84rem; color: #475569; line-height: 1.9; }

/* Learner tip */
.ic-tip {
    margin-top: 14px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 0.82rem;
    color: #78350f;
    line-height: 1.6;
}

/* ── Run button ── */
.stButton > button {
    background: #ffffff !important;
    color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.18s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.stButton > button:hover {
    background: #f5f3ff !important;
    border-color: #6366f1 !important;
    color: #6366f1 !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 20px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px);
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricValue"] { color: #6366f1 !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; font-weight: 500 !important; }

/* ── Results header ── */
.results-header { font-size: 1.05rem; font-weight: 700; color: #1e293b; margin-bottom: 16px; }
.results-header span { color: #6366f1; }

/* ── Chunk cards (single column, full background colour) ── */
.chunk-card {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border-left: 4px solid;
    line-height: 1.75;
    font-size: 0.88rem;
    color: #1e293b;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    word-break: break-word;
    white-space: pre-wrap;
}
.chunk-num {
    font-size: 0.69rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    opacity: 0.85;
}
.chunk-meta {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 10px;
    display: flex;
    gap: 18px;
    font-weight: 500;
}

/* ── Welcome / empty state ── */
.welcome-state {
    text-align: center;
    padding: 64px 24px;
    background: #ffffff;
    border: 2px dashed #e2e8f0;
    border-radius: 18px;
    margin-top: 8px;
}
.welcome-icon { font-size: 3rem; margin-bottom: 14px; }
.welcome-title { font-size: 1.2rem; font-weight: 700; color: #334155; margin-bottom: 8px; }
.welcome-desc { font-size: 0.9rem; color: #94a3b8; line-height: 1.65; max-width: 400px; margin: 0 auto; }

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stTextArea"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label {
    color: #334155 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
[data-testid="stTextInput"] input {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    outline: none !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: #6366f1 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrack"] > div { background: #6366f1 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] { color: #64748b !important; font-weight: 600 !important; font-size: 0.87rem !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #6366f1 !important; border-bottom-color: #6366f1 !important; }
[data-testid="stTabs"] { border-bottom: 1px solid #e2e8f0 !important; }

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
[data-testid="stExpander"] summary { color: #6366f1 !important; font-weight: 600 !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #f0fdf4 !important;
    border: 1.5px solid #bbf7d0 !important;
    color: #16a34a !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover { background: #dcfce7 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Chunker loading (properly cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_chunker_cached(name: str, params_frozen: tuple):
    """Cache chunker instances by name + params to avoid re-instantiation."""
    params = dict(params_frozen)
    cfg = ALL_CHUNKERS[name]
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class"])
    return cls(**params)


def load_chunker(name: str, params: dict):
    """Public loader — converts params dict to a hashable tuple for caching."""
    return _load_chunker_cached(name, tuple(sorted(params.items())))


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # ── Author card
        st.markdown("""
<div class="creator-card">
    <strong>👤 Rohith Sukka</strong>
    <span class="role">AI/ML Engineer &amp; Educator</span>
    <div class="links">
        <a href="https://github.com/rohithsukka" target="_blank">GitHub</a>
        <a href="https://linkedin.com/in/rohithsukka" target="_blank">LinkedIn</a>
        <a href="https://rohithsukka.github.io/portfolio/" target="_blank">Website</a>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── How to use (replaces fake navigation)
        st.markdown("""
<div class="how-to-card">
    <div class="how-title">📋 How to Use</div>
    <div class="how-step"><span class="sn">1</span>Pick a sample text or paste your own</div>
    <div class="how-step"><span class="sn">2</span>Choose a chunking technique from the list</div>
    <div class="how-step"><span class="sn">3</span>Adjust parameters if they appear</div>
    <div class="how-step"><span class="sn">4</span>Click <b>Run Chunking</b> &amp; explore the results</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        # ── App footer info
        st.markdown("""
<div class="footer-badge">
    <b>Chunking Playground</b><br>
    v1.0 · Built with Streamlit<br>
    Learn chunking for RAG &amp; LLMs.<br>
    All techniques run locally.
</div>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
<div class="cp-header">
    <h1>🧩 Chunking Playground</h1>
    <div class="subtitle">Explore Text Chunking Techniques for RAG &amp; LLM Systems</div>
    <div class="tagline">An interactive learning tool — by Rohith Sukka</div>
</div>
""", unsafe_allow_html=True)


# ── Parameters panel ───────────────────────────────────────────────────────────
def render_params(name: str) -> dict:
    param_keys = ALL_CHUNKERS[name]["params"]
    params = {}

    if not param_keys:
        st.markdown(
            "<div style='font-size:0.82rem;color:#94a3b8;padding:4px 0;'>"
            "No configurable parameters for this technique.</div>",
            unsafe_allow_html=True,
        )
        return params

    for p in param_keys:
        if p == "chunk_size":
            params["chunk_size"] = st.slider("Chunk Size (words)", 20, 500, 100, 10, key=f"cs_{name}")
        elif p == "overlap":
            params["overlap"] = st.slider("Overlap (words)", 0, 100, 20, 5, key=f"ov_{name}")
        elif p == "page_separator":
            params["page_separator"] = st.text_input(
                "Page Separator", value="<PAGE_BREAK>", key=f"sep_{name}"
            )
        elif p == "n_clusters":
            params["n_clusters"] = st.slider("Number of Clusters", 2, 10, 3, 1, key=f"nc_{name}")

    return params


# ── Chunk visualisation (single column, full text, coloured background) ───────
def render_chunks(chunks: list[str]):
    for idx, chunk in enumerate(chunks):
        color_border, color_bg = CHUNK_PALETTE[idx % len(CHUNK_PALETTE)]
        word_count = len(chunk.split())
        char_count = len(chunk)

        # Safely escape HTML special characters so raw text renders correctly
        safe_text = (
            chunk
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        st.markdown(f"""
<div class="chunk-card" style="background:{color_bg};border-left-color:{color_border};">
    <div class="chunk-num" style="color:{color_border};">Chunk {idx + 1}</div>
    <div>{safe_text}</div>
    <div class="chunk-meta">
        <span>📝 {word_count} words</span>
        <span>🔤 {char_count} chars</span>
    </div>
</div>""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    render_sidebar()
    render_header()

    # ── Session state initialisation
    defaults = {
        "selected_technique": "Naive",
        "result": None,
        "last_chunker": None,
        "last_params": {},
        "input_text": SAMPLE_TEXTS["Custom"],   # seed so text_area has no value= conflict
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def on_sample_change():
        """Clear results and update text area when a new sample is chosen."""
        new_key = st.session_state["sample_select"]
        st.session_state["input_text"] = SAMPLE_TEXTS[new_key]
        st.session_state.result = None

    def on_technique_change():
        """Single callback for all technique selections — no dual-radio state conflict."""
        st.session_state.selected_technique = st.session_state["radio_technique"]
        st.session_state.result = None

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Enter Your Text
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
<div class="step-header">
    <div class="step-circle">1</div>
    <div class="step-label">Enter Your Text</div>
</div>""", unsafe_allow_html=True)

    text_left, text_right = st.columns([1, 2], gap="large")

    with text_left:
        sample_choice = st.selectbox(
            "Choose a sample text:",
            list(SAMPLE_TEXTS.keys()),
            key="sample_select",
            on_change=on_sample_change,
        )
        desc_html = SAMPLE_DESCRIPTIONS.get(sample_choice, "")
        st.markdown(f'<div class="sample-desc">💡 {desc_html}</div>', unsafe_allow_html=True)

    with text_right:
        # NOTE: do NOT pass value= here — session_state["input_text"] drives the content.
        # on_sample_change() updates st.session_state["input_text"] so the textarea refreshes.
        input_text = st.text_area(
            "Text to chunk (edit freely):",
            height=210,
            key="input_text",
            placeholder="Paste or type your text here…",
        )
        w = len(input_text.split()) if input_text.strip() else 0
        c = len(input_text)
        st.markdown(
            f'<div class="live-count">📄 {w} words · {c} characters</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Choose Technique + See Description + Configure Params
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
<div class="step-header">
    <div class="step-circle">2</div>
    <div class="step-label">Choose a Chunking Technique</div>
    <span class="step-sub">— select one below, then tweak parameters if needed</span>
</div>""", unsafe_allow_html=True)

    tech_col, info_col = st.columns([1, 1.15], gap="large")

    with tech_col:
        # Build a flat ordered list: all Basic techniques then all AI techniques
        basic_opts = list(CHUNKER_REGISTRY["Basic"].keys())
        ai_opts    = list(CHUNKER_REGISTRY["AI-Powered"].keys())
        all_opts   = basic_opts + ai_opts          # e.g. [...6 basic..., "Semantic"]

        current_idx = (
            all_opts.index(st.session_state.selected_technique)
            if st.session_state.selected_technique in all_opts
            else 0
        )

        # ── Category header: Basic
        st.markdown('<div class="cat-label">⚡ Basic Methods</div>', unsafe_allow_html=True)

        # Single unified radio — avoids the dual-radio on_change conflict
        st.radio(
            "Choose technique",
            all_opts,
            index=current_idx,
            key="radio_technique",
            on_change=on_technique_change,
            label_visibility="collapsed",
        )

        # ── Category label for AI section (rendered as annotation below the pills)
        st.markdown(
            '<div class="cat-label" style="margin-top:8px;">🤖 AI-Powered</div>'
            '<div class="ai-note">⚠️ <b>Semantic</b> uses an ML model (all-MiniLM-L6-v2) '
            'and K-Means clustering — first run may take a few seconds to load the model.</div>',
            unsafe_allow_html=True,
        )

        # ── Parameters
        st.markdown('<div class="param-label">⚙️ Parameters</div>', unsafe_allow_html=True)
        params = render_params(st.session_state.selected_technique)

    with info_col:
        selected = st.session_state.selected_technique

        # Show which technique is currently active (fixes cross-category confusion)
        st.markdown(
            f'<div class="selected-indicator">✅ Currently selected: <span style="color:#6366f1;">{selected}</span></div>',
            unsafe_allow_html=True,
        )

        # ── Rich technique detail card
        td = TECHNIQUE_DETAILS.get(selected, {})
        try:
            safe_params: dict = {}
            for p in ALL_CHUNKERS[selected]["params"]:
                if p == "chunk_size":       safe_params["chunk_size"]       = 100
                elif p == "overlap":        safe_params["overlap"]          = 20
                elif p == "n_clusters":     safe_params["n_clusters"]       = 3
                elif p == "page_separator": safe_params["page_separator"]   = "<PAGE_BREAK>"

            dummy = load_chunker(selected, safe_params)
            info  = dummy.get_info()
            use_cases_html = "".join(f"<li>{uc}</li>" for uc in info.get("use_cases", []))

            meaning_badge = (
                '<span class="badge badge-ai">✅ Understands Meaning</span>'
                if td.get("understands_meaning")
                else '<span class="badge badge-rule">📏 Rule-Based</span>'
            )
            speed_html = f'<span class="badge" style="background:#f0fdf4;color:{td.get("speed_color","#16a34a")};border-color:{td.get("speed_color","#16a34a")};">⏱ {td.get("speed", "")} </span>'

            card_html = (
                f'<div class="info-card">'
                f'<div class="ic-header">'
                f'<span class="ic-emoji">{td.get("emoji", "🔧")}</span>'
                f'<div>'
                f'<div class="ic-name">{info["name"]}</div>'
                f'<div class="ic-badges">{meaning_badge} {speed_html}</div>'
                f'</div></div>'
                f'<div class="ic-desc">{info["description"]}</div>'
                f'<div class="ic-section-label">✂️ Splits On</div>'
                f'<div class="ic-value">{td.get("split_on", "—")}</div>'
                f'<div class="ic-section-label">⚙️ How It Works</div>'
                f'<div class="ic-value ic-code">{td.get("logic", "—")}</div>'
                f'<div class="ic-section-label">🎛️ Parameters</div>'
                f'<div class="ic-value">{td.get("parameters", "None")}</div>'
                f'<div class="ic-section-label">💡 When to Use</div>'
                f'<ul class="ic-list">{use_cases_html}</ul>'
                f'<div class="ic-tip">🎓 <b>Learner tip:</b> {td.get("learner_tip", "")}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load technique info: {e}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Run Chunking
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
<div class="step-header">
    <div class="step-circle">3</div>
    <div class="step-label">Run &amp; Explore Results</div>
</div>""", unsafe_allow_html=True)

    run_col, clear_col = st.columns([5, 1], gap="small")
    with run_col:
        run_btn = st.button(
            f"🚀  Run  ·  {st.session_state.selected_technique}",
            type="primary",
            use_container_width=True,
            key="run_btn",
        )
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
            st.session_state.result = None
            st.rerun()

    if run_btn:
        text_val = st.session_state.get("input_text", "").strip()
        if not text_val:
            st.error("⚠️ Please enter some text in Step 1 before running.")
        else:
            with st.spinner(f"Running **{st.session_state.selected_technique}** chunker…"):
                try:
                    chunker = load_chunker(st.session_state.selected_technique, params)
                    result = chunker.chunk_with_timing(text_val)
                    st.session_state.result = result
                    st.session_state.last_chunker = st.session_state.selected_technique
                    st.session_state.last_params = params
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")
                    st.session_state.result = None

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS (or welcome state)
    # ══════════════════════════════════════════════════════════════════════════
    result = st.session_state.result

    if result is None:
        # ── Welcome / onboarding empty state
        st.markdown("""
<div class="welcome-state">
    <div class="welcome-icon">🧩</div>
    <div class="welcome-title">Your chunks will appear here</div>
    <div class="welcome-desc">
        Choose a technique above, enter your text, and click
        <b>Run Chunking</b> to see how the text gets split.
    </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown(
            f'<div class="results-header">📊 Results — <span>{st.session_state.last_chunker}</span></div>',
            unsafe_allow_html=True,
        )

        # ── Stats row with REAL avg-words calculation (fix #8)
        proc_time = result.metadata.get("processing_time", 0)
        chunks = result.chunks
        real_avg_words = (
            sum(len(c.split()) for c in chunks) / len(chunks) if chunks else 0
        )

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Chunks",       result.chunk_count)
        s2.metric("Total Chars",        f"{result.total_length:,}")
        s3.metric("Avg Chunk Size",     f"{result.avg_chunk_size:.0f} chars")
        s4.metric("Avg Words / Chunk",  f"{real_avg_words:.1f}")   # real, not chars÷5
        s5.metric("⏱ Process Time",     f"{proc_time * 1000:.2f} ms")

        st.markdown("")

        # ── Chunk grid — single column, full text, coloured backgrounds (fixes #4, #7, #12)
        render_chunks(chunks)

        st.markdown("---")

        # ── Download + raw view
        dl_col, raw_col = st.columns(2)
        with dl_col:
            payload = {
                "technique":        st.session_state.last_chunker,
                "parameters":       st.session_state.last_params,
                "chunk_count":      result.chunk_count,
                "total_length":     result.total_length,
                "avg_chunk_size":   result.avg_chunk_size,
                "avg_words_per_chunk": round(real_avg_words, 2),
                "processing_time_ms": round(proc_time * 1000, 4),
                "chunks":           result.chunks,
            }
            st.download_button(
                "⬇️ Download Results (JSON)",
                data=json.dumps(payload, indent=2),
                file_name=f"{st.session_state.last_chunker}_chunks.json",
                mime="application/json",
                use_container_width=True,
            )
        with raw_col:
            with st.expander("🔍 View Raw Chunks List"):
                for i, chunk in enumerate(result.chunks, 1):
                    st.markdown(
                        f"**#{i}** `{len(chunk)} chars` — {chunk[:120]}{'…' if len(chunk) > 120 else ''}"
                    )


if __name__ == "__main__":
    main()
