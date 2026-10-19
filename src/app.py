import os
import sys
import asyncio
from pathlib import Path

# Ensure project root is in sys.path when executed via 'streamlit run'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.vectorstore import AcademicVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import LocalReranker
from src.chains.router import PaperRouter
from src.chains.map_reduce import MapReduceRAG

# Streamlit Page Config
st.set_page_config(
    page_title="Academic Semantic Search Engine",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Academic Semantic Search & Literature Review")
st.caption("Addressing 'Lost in the Middle' with Hybrid Retrieval, Cross-Encoder Re-ranking, and Map-Reduce Synthesis.")


# Cache persistent components so they reload instantly
@st.cache_resource(show_spinner="Initializing retrieval index and embedding models...")
def init_system():
    vector_store = AcademicVectorStore()

    # Ingest default sample papers if collection is empty
    sample_dir = Path("data/sample_papers")
    if sample_dir.exists():
        existing_count = vector_store.collection.count()
        if existing_count == 0:
            for pdf_file in sample_dir.glob("*.pdf"):
                vector_store.index_pdf(pdf_file)

    # Initialize BM25 on all indexed texts
    raw_data = vector_store.collection.get(include=["documents", "metadatas"])
    docs = [{"text": d, "metadata": m} for d, m in zip(raw_data["documents"], raw_data["metadatas"])]
    bm25 = BM25Retriever()
    bm25.index_documents(docs)

    hybrid = HybridSearcher(vector_store=vector_store, bm25_retriever=bm25)
    reranker = LocalReranker()
    router = PaperRouter()
    rag = MapReduceRAG()

    return vector_store, bm25, hybrid, reranker, router, rag


vector_store, bm25, hybrid, reranker, router, rag = init_system()

# Sidebar: Corpus & Document Management
with st.sidebar:
    st.header("📄 Ingested Literature")
    raw_data = vector_store.collection.get(include=["metadatas"])
    sources = sorted(list({m["source"] for m in raw_data["metadatas"]})) if raw_data["metadatas"] else []

    if sources:
        for s in sources:
            st.markdown(f"- **{s}**")
    else:
        st.info("No papers found in collection.")

    st.divider()
    st.subheader("Upload New Paper")
    uploaded_file = st.file_uploader("Upload an academic PDF", type=["pdf"])
    if uploaded_file is not None:
        save_path = Path("data/sample_papers") / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner(f"Parsing & indexing {uploaded_file.name}..."):
            count = vector_store.index_pdf(save_path)
            # Refresh BM25 index
            updated_data = vector_store.collection.get(include=["documents", "metadatas"])
            docs = [{"text": d, "metadata": m} for d, m in zip(updated_data["documents"], updated_data["metadatas"])]
            bm25.index_documents(docs)
        st.success(f"Indexed {count} new chunks from {uploaded_file.name}!")
        st.rerun()

    st.divider()
    st.markdown("### Architecture Pipeline")
    st.markdown("""
    1. **Paper Router:** Query-to-paper filtering
    2. **Hybrid Search:** Dense + BM25 (RRF)
    3. **Cross-Encoder:** FlashRank Neural Re-ranking
    4. **Map Stage:** Async Groq Fact Extraction
    5. **Reduce Stage:** Gemini Synthesis with Citations
    """)

# Main Query Interface
query = st.text_input(
    "Enter an academic research question:",
    value="How does context position affect language model retrieval performance, and what did the authors observe?",
)

col1, col2 = st.columns([1, 1])
with col1:
    top_k_retrieve = st.slider("Passages to retrieve (Hybrid)", min_value=4, max_value=20, value=10)
with col2:
    top_k_rerank = st.slider("Passages to re-rank (Cross-Encoder)", min_value=2, max_value=8, value=5)

if st.button("Run Literature Synthesis", type="primary"):
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Step 1: Routing query to target literature..."):
            papers_metadata = [{"filename": s, "description": "Academic PDF"} for s in sources]
            selected_papers = router.route_query(query, papers_metadata)
            st.info(f"Target Papers Identified: {', '.join(selected_papers)}")

        with st.spinner("Step 2: Performing Hybrid Search & Cross-Encoder Re-ranking..."):
            candidates = hybrid.search(query=query, fused_top_k=top_k_retrieve)
            filtered = [c for c in candidates if c["metadata"]["source"] in selected_papers]
            top_passages = reranker.rerank(query=query, candidates=filtered, top_n=top_k_rerank)

        with st.spinner("Step 3: Running Parallel Map Extraction & Reduce Synthesis..."):
            result = asyncio.run(rag.synthesize(query=query, retrieved_chunks=top_passages))

        st.subheader("💡 Synthesized Literature Review")
        st.markdown(result["answer"])

        with st.expander(f"🔍 View Distilled Evidence Blocks ({result['evidence_count']} extracted)"):
            for i, ev in enumerate(result["raw_evidence"], start=1):
                st.markdown(f"**Evidence Block {i}:**")
                st.text(ev)
                st.divider()