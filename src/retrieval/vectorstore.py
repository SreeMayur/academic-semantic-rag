from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from src.ingestion.parser import AcademicPDFParser
from src.ingestion.chunker import AcademicChunker


class AcademicVectorStore:
    """
    Manages local document embeddings and vector search using ChromaDB
    and SentenceTransformers (all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "academic_papers",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 1. Initialize persistent local Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # 2. Free, local CPU embedding function
        self.embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        )

        # 3. Create or get existing collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_pdf(self, pdf_path: str | Path) -> int:
        """
        Parses, chunks, and indexes a single PDF document into ChromaDB.
        Returns the number of indexed chunks.
        """
        parser = AcademicPDFParser(pdf_path)
        blocks = parser.extract_blocks()

        chunker = AcademicChunker()
        chunks = chunker.chunk_blocks(blocks)

        if not chunks:
            return 0

        # Prepare payloads for Chroma batch insertion
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{c['metadata']['source']}_p{c['metadata']['page']}_c{c['metadata']['chunk_id']}"
            for c in chunks
        ]

        # Chroma handles batches automatically
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        return len(chunks)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs dense semantic search and returns top_k relevant chunks with similarity score.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity = 1.0 - dist
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                })

        return formatted_results


if __name__ == "__main__":
    print("AcademicVectorStore loaded successfully.")