from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class AcademicChunker:
    """
    Chunks extracted text blocks while preserving source document metadata
    and preventing cross-paragraph fragmentation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes raw blocks from AcademicPDFParser and yields granular chunks
        with enriched metadata (chunk index, page, source).
        """
        all_chunks = []
        global_chunk_id = 0

        for block in blocks:
            text = block["text"]
            base_metadata = block["metadata"]

            # If the block is already smaller than chunk_size, keep it intact
            if len(text) <= self.chunk_size:
                sub_texts = [text]
            else:
                sub_texts = self.splitter.split_text(text)

            for sub_text in sub_texts:
                if len(sub_text.strip()) > 30:  # Drop tiny noise fragments
                    all_chunks.append({
                        "text": sub_text.strip(),
                        "metadata": {
                            "source": base_metadata["source"],
                            "page": base_metadata["page"],
                            "chunk_id": global_chunk_id,
                            "char_count": len(sub_text.strip()),
                        }
                    })
                    global_chunk_id += 1

        return all_chunks


if __name__ == "__main__":
    print("AcademicChunker loaded successfully.")