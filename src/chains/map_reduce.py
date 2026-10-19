import os
import asyncio
from typing import List, Dict, Any
from groq import AsyncGroq
from google import genai
from dotenv import load_dotenv

load_dotenv()


class MapReduceRAG:
    """
    Overcomes the 'Lost in the Middle' problem by running parallel extraction
    on individual chunks (Map step via Groq) before final cited synthesis
    (Reduce step via Google Gemini).
    """

    def __init__(
            self,
            groq_model: str = "openai/gpt-oss-20b",
            gemini_model: str = "gemini-3.8-flash",
        ):
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.groq_model = groq_model
        self.gemini_model = gemini_model

    async def _map_chunk(self, query: str, chunk: Dict[str, Any]) -> str:
        """
        Map step: Extract only evidence and facts directly answering the query
        from a single passage.
        """
        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]
        text = chunk["text"]

        prompt = f"""You are an academic research analyst.
Examine this passage from '{source}' (Page {page}) and extract all specific facts, metrics, findings, or mechanisms that help answer the query.

QUERY: "{query}"

PASSAGE:
\"\"\"{text}\"\"\"

INSTRUCTIONS:
- Extract all direct evidence, hyperparameter values, formulas, or conceptual explanations.
- If this passage contains NO relevant information, output: NONE.
- Keep extraction concise and tag each bullet with [{source}, Page {page}].
"""

        try:
            res = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250,
            )
            content = res.choices[0].message.content.strip()
            return "" if "NONE" in content.upper() and len(content) < 20 else content
        except Exception as e:
            return f"[Error processing chunk {source} p{page}: {e}]"

    async def synthesize(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes parallel Map across all chunks, then invokes Gemini for Reduce synthesis.
        """
        # 1. Parallel Map Step
        tasks = [self._map_chunk(query, c) for c in retrieved_chunks]
        map_results = await asyncio.gather(*tasks)

        # Filter out empty extractions
        valid_evidence = [r for r in map_results if r.strip()]

        if not valid_evidence:
            return {
                "answer": "No relevant evidence found in the retrieved context to answer the query.",
                "evidence_count": 0,
                "raw_evidence": [],
            }

        distilled_context = "\n\n".join(valid_evidence)

        # 2. Reduce Step (Gemini synthesis)
        reduce_prompt = f"""You are a senior academic researcher conducting a rigorous literature review.
Answer the following query using ONLY the distilled evidence below.

QUERY:
"{query}"

DISTILLED EVIDENCE FROM RETRIEVED LITERATURE:
{distilled_context}

REQUIREMENTS:
1. Provide a comprehensive, technical, and coherent synthesis directly answering the query.
2. Every major claim or metric MUST cite its source and page in brackets, e.g., [attention_is_all_you_need.pdf, Page 5].
3. If different papers provide contrasting viewpoints or complementary components, explicitly compare them.
4. Conclude with a brief 'Key Takeaway' section.
"""

        response = self.google_client.models.generate_content(
            model=self.gemini_model,
            contents=reduce_prompt,
        )

        return {
            "answer": response.text,
            "evidence_count": len(valid_evidence),
            "raw_evidence": valid_evidence,
        }


if __name__ == "__main__":
    print("MapReduceRAG loaded successfully.")