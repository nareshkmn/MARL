# gemini_search.py
import google.generativeai as genai
from typing import List, Dict
from config import Config

genai.configure(api_key=Config.GEMINI_API_KEY)

MODEL = "gemini-2.0-flash"  # cheap & fast

def gemini_web_search(query: str, n_results: int = 5) -> List[str]:
    prompt = f"""
Search the web for the query: "{query}"

Return ONLY a list of {n_results} relevant links,
ranked by relevance, without commentary.
"""
    response = genai.GenerativeModel(MODEL).generate_content(prompt)
    text = response.text.strip()
    return [line.strip("- • ") for line in text.split("\n") if line.startswith(("-", "•"))]


def gemini_extract_paper_metadata(url: str) -> Dict:
    prompt = f"""
Given the following URL, extract paper metadata:

URL: {url}

Return a JSON with:
- title
- authors
- year
- venue if available
- summary (3–5 sentences)
- key findings (bullet points)
- methodology summary
- limitations
"""
    response = genai.GenerativeModel(MODEL).generate_content(prompt)
    import json
    try:
        return json.loads(response.text)
    except:
        return {"raw": response.text}
