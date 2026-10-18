import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    In-memory sparse keyword retriever using the Okapi BM25 ranking algorithm.
    """

    def __init__(self):
        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Standard alphanumeric lowercased word tokenization
        return re.findall(r"\w+", text.lower())

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Indexes a list of documents where each item is a dict with 'text' and 'metadata'.
        """
        self.corpus = documents
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries BM25 and returns top_k results.
        """
        if not self.bm25 or not self.corpus:
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Sort indices by descending BM25 score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:  # Only return matches with positive term overlap
                results.append({
                    "text": self.corpus[idx]["text"],
                    "metadata": self.corpus[idx]["metadata"],
                    "bm25_score": round(float(scores[idx]), 4),
                })
        return results


if __name__ == "__main__":
    print("BM25Retriever loaded successfully.")