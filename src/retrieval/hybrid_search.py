from typing import List, Dict, Any
from src.retrieval.vectorstore import AcademicVectorStore
from src.retrieval.bm25_retriever import BM25Retriever


class HybridSearcher:
    """
    Fuses dense semantic search (ChromaDB) and sparse keyword search (BM25)
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_store: AcademicVectorStore, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    @staticmethod
    def _doc_key(doc: Dict[str, Any]) -> str:
        meta = doc["metadata"]
        return f"{meta['source']}_p{meta['page']}_c{meta.get('chunk_id', 0)}"

    def search(
        self,
        query: str,
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        rrf_k: int = 60,
        fused_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Executes dense and sparse queries, then aggregates ranks using RRF:
        RRF_Score(d) = SUM( 1 / (k + rank(d)) )
        """
        dense_results = self.vector_store.query(query, top_k=dense_top_k)
        sparse_results = self.bm25_retriever.query(query, top_k=sparse_top_k)

        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        # 1. Score dense rankings
        for rank, item in enumerate(dense_results, start=1):
            key = self._doc_key(item)
            doc_store[key] = item
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Score sparse rankings
        for rank, item in enumerate(sparse_results, start=1):
            key = self._doc_key(item)
            if key not in doc_store:
                doc_store[key] = item
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        # 3. Sort by fused RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        final_results = []
        for key in sorted_keys[:fused_top_k]:
            item = doc_store[key]
            final_results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "rrf_score": round(rrf_scores[key], 6),
            })

        return final_results


if __name__ == "__main__":
    print("HybridSearcher loaded successfully.")