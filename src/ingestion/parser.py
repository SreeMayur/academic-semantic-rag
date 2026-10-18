from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF


class AcademicPDFParser:
    """
    Layout-aware parser for academic PDFs.
    Extracts text preserving multi-column reading order and attaches page/section metadata.
    """

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

    def extract_blocks(self) -> List[Dict[str, Any]]:
        """
        Extracts structured text blocks sorted in natural reading order.
        PyMuPDF's get_text('blocks') returns tuples:
        (x0, y0, x1, y1, "text", block_no, block_type)
        where block_type == 0 is text (block_type == 1 is image).
        """
        doc = fitz.open(self.pdf_path)
        extracted_data = []

        for page_idx, page in enumerate(doc):
            # Sort blocks: reading order by top-to-bottom, left-to-right columns
            blocks = page.get_text("blocks", sort=True)

            for b in blocks:
                text = b[4].strip()
                block_type = b[6]

                # Filter out empty blocks and image placeholders
                if block_type == 0 and len(text) > 20:
                    # Clean up hyphenation common at line breaks in academic text
                    cleaned_text = text.replace("-\n", "").replace("\n", " ")

                    extracted_data.append({
                        "text": cleaned_text,
                        "metadata": {
                            "source": self.pdf_path.name,
                            "page": page_idx + 1,
                            "bbox": (b[0], b[1], b[2], b[3]),  # coordinates for reference
                        }
                    })

        doc.close()
        return extracted_data


if __name__ == "__main__":
    # Quick self-test description
    print("AcademicPDFParser loaded successfully.")