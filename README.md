# 📚 Academic Semantic Search Engine

An end-to-end, multi-stage Retrieval-Augmented Generation (RAG) system engineered to ingest dense academic literature, mitigate the **"Lost in the Middle"** context degradation phenomenon, and synthesize cited research reviews using free-tier inference and local neural embeddings.

---

## 🎯 Problem Statement: The "Lost in the Middle" Phenomenon

Standard RAG architectures struggle when processing extensive scientific literature:
1. **Layout Garbling:** Naive PDF parsers interleave two-column text, corrupting mathematical formulas and citations.
2. **Keyword Fragility:** Pure dense vector embeddings often fail on exact academic terminology, hyperparameter values ($d_k = 64$), or author names.
3. **Contextual Degradation:** As established by Liu et al. (2023), Large Language Models exhibit a U-shaped performance curve—retrieval fidelity drops sharply when critical evidence is situated in the middle of long input contexts.

---

## 🏗️ System Architecture

This project resolves these failure modes via a modular 5-stage pipeline:

```text
[Academic PDFs]
       │
       ▼
[Layout-Aware PyMuPDF Ingestion] ──► [Metadata-Preserving Chunker]
                                                     │
                                                     ▼
[Query] ──► [Multi-Paper Abstract Router]   [Dense ChromaDB + Sparse BM25]
                    │                                       │
                    └───────────────┬───────────────────────┘
                                    ▼
                     [Reciprocal Rank Fusion (RRF)]
                                    │
                                    ▼
                  [FlashRank Cross-Encoder Re-ranker]
                                    │
                                    ▼
                 [MAP Phase: Async Parallel Fact Extraction (Groq)]
                                    │
                                    ▼
                 [REDUCE Phase: Synthesis with Grounded Citations (Gemini)]
```

### Key Technical Components:
- **Layout-Aware PDF Ingestion:** PyMuPDF (`pymupdf`) extracts blocks according to natural reading order, preserving 2-column boundaries, headings, and page coordinates.
- **Hybrid Retrieval (Dense + Sparse):** Merges semantic vector similarity (`sentence-transformers/all-MiniLM-L6-v2` via ChromaDB) with sparse keyword matching (`rank-bm25`) using **Reciprocal Rank Fusion (RRF)**.

$$\mathrm{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + \mathrm{rank}_m(d)}$$

- **Local Cross-Encoder Re-ranking:** Re-scores fused candidates using `FlashRank` (`ms-marco-TinyBERT-L-2-v2`) locally on CPU to prioritize semantic alignment before prompting.
- **Agentic Abstract Router:** Evaluates query intent against corpus abstracts to prune irrelevant documents before retrieval.
- **Hierarchical Map-Reduce Synthesis:**
  - **Map Step (Groq):** Concurrently extracts isolated evidence and metric snippets from top chunks using asynchronous inference.
  - **Reduce Step (Google Gemini 3.8 Flash):** Synthesizes extracted facts into a coherent literature review with strict bracket citations (`[paper.pdf, Page X]`).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Free API keys from [Groq Console](https://console.groq.com) and [Google AI Studio](https://aistudio.google.com)

### 2. Installation & Setup
```bash
# Clone the repository
git clone [https://github.com/your-username/academic-semantic-rag.git](https://github.com/your-username/academic-semantic-rag.git)
cd academic-semantic-rag

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
source venv/bin/activate       # On Linux/macOS

# Install dependencies
pip install -r requirements.txt