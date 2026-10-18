from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest


class LocalReranker:
    """
    Ultra-lightweight, local cross-encoder re-ranking using FlashRank.
    Runs entirely on CPU with zero GPU / API requirements.
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        # Uses the default compact ~4MB TinyBERT model
        self.ranker = Ranker(model_name=model_name, cache_dir="data/cache")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Takes candidate chunks and produces deep cross-attention relevance scores.
        """
        if not candidates:
            return []

        # FlashRank expects documents formatted as dicts with an 'id' and 'text'
        passages = [
            {"id": idx, "text": doc["text"], "metadata": doc["metadata"]}
            for idx, doc in enumerate(candidates)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_passages = self.ranker.rerank(rerank_req)

        results = []
        for p in ranked_passages[:top_n]:
            results.append({
                "text": p["text"],
                "metadata": p["metadata"],
                "rerank_score": round(float(p["score"]), 4),
            })
        return results


if __name__ == "__main__":
    print("LocalReranker loaded successfully.")