import os
import json
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class PaperRouter:
    """
    Analyzes user queries against registered paper metadata
    to select relevant documents before retrieval.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def route_query(
        self, query: str, available_papers: List[Dict[str, str]]
    ) -> List[str]:
        """
        Takes candidate papers [{'filename': '...', 'description': '...'}]
        and returns a list of target filenames.
        """
        papers_context = "\n".join(
            [f"- {p['filename']}: {p.get('description', 'Academic Paper')}" for p in available_papers]
        )

        prompt = f"""You are an academic research assistant routing a literature query to relevant research papers.

AVAILABLE PAPERS:
{papers_context}

USER QUERY:
"{query}"

INSTRUCTIONS:
Return ONLY a valid JSON list containing the filenames of papers that are directly relevant to answering this query.
If all papers could be relevant or in doubt, include them.
Do NOT include markdown formatting, backticks, or explanatory text. Return strictly raw JSON.

Example output:
["attention_is_all_you_need.pdf"]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            # Clean possible markdown fence wrappers
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            selected = json.loads(cleaned)
            return [f for f in selected if any(p["filename"] == f for p in available_papers)]
        except Exception:
            # Fallback: include all papers if routing call fails
            return [p["filename"] for p in available_papers]


if __name__ == "__main__":
    print("PaperRouter loaded successfully.")