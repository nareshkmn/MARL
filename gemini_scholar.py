# gemini_scholar.py
from typing import List, Dict, Any
import json
from io import BytesIO

import google.generativeai as genai
import requests
from pypdf import PdfReader

from config import Config

# Configure Gemini
genai.configure(api_key=Config.GEMINI_API_KEY)
MODEL_NAME = Config.GEMINI_SCHOLAR_MODEL


def _model():
    return genai.GenerativeModel(MODEL_NAME)


# ---------- 1. Paper search ----------

def search_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Use Gemini to search for relevant research papers.
    We ask it to return a JSON list of paper metadata.
    """
    prompt = f"""
You are a scholarly search assistant.

Search the *current* research literature and web for papers about:

  "{query}"

Return STRICTLY valid JSON of the form:
{{
  "papers": [
    {{
      "title": "...",
      "authors": ["..."],
      "year": 2024,
      "venue": "...",
      "url": "...",
      "pdf_url": "...",
      "abstract": "...",
      "key_topics": ["..."]
    }},
    ...
  ]
}}

- Prefer arXiv / open-access links when possible.
- Only include up to {max_results} papers.
- Do NOT include any text outside the JSON.
"""
    resp = _model().generate_content(prompt)
    txt = resp.text.strip()
    try:
        data = json.loads(txt)
        papers = data.get("papers", [])
        if isinstance(papers, list):
            return papers[:max_results]
        return []
    except json.JSONDecodeError:
        return [{"raw_response": txt}]


# ---------- 2. Download and parse PDFs ----------

def download_pdf_text(pdf_url: str, max_pages: int = 20) -> str:
    """
    Download a PDF and extract text (first max_pages pages).
    """
    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"[PDF download failed: {e}]"

    try:
        reader = PdfReader(BytesIO(resp.content))
        pages = reader.pages[:max_pages]
        texts = [(p.extract_text() or "") for p in pages]
        return "\n\n".join(texts)
    except Exception as e:
        return f"[PDF parsing failed: {e}]"


# ---------- 3. Deep analysis: sections, equations, tables ----------

def analyze_paper_full(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given paper metadata, optionally download PDF and ask Gemini
    to produce a rich structured analysis (sections, equations, tables).
    """
    pdf_url = paper.get("pdf_url")
    text_source = ""

    if pdf_url:
        pdf_text = download_pdf_text(pdf_url)
        text_source = pdf_text
    else:
        abstract = paper.get("abstract", "")
        text_source = abstract or "[No PDF and no abstract available]"

    if len(text_source) > 15000:
        text_source = text_source[:15000]

    meta_str = json.dumps(paper, indent=2)

    prompt = f"""
You are an expert research assistant.

We have the following paper metadata:

{meta_str}

And the following extracted text from the paper (possibly partial):

\"\"\"{text_source}\"\"\"


From this, produce STRICT JSON with the following structure:

{{
  "title": "...",
  "core_contribution": "...",
  "problem_statement": "...",
  "methods": {{
    "high_level": "...",
    "details": ["..."]
  }},
  "datasets_or_setup": ["..."],
  "results": ["..."],
  "limitations": ["..."],
  "future_work": ["..."],
  "sections": [
    {{
      "heading": "...",
      "summary": "..."
    }}
  ],
  "important_equations": [
    {{
      "symbolic_form": "LaTeX-like equation",
      "meaning": "intuitive explanation"
    }}
  ],
  "important_tables": [
    {{
      "description": "...",
      "key_numbers": ["..."]
    }}
  ]
}}

Rules:
- Fill fields as best as you can.
- If something is unknown, use an empty list or empty string.
- DO NOT include commentary or any text outside the JSON.
"""
    resp = _model().generate_content(prompt)
    txt = resp.text.strip()
    try:
        data = json.loads(txt)
        return data
    except json.JSONDecodeError:
        return {
            "metadata": paper,
            "raw_analysis": txt,
        }


def build_corpus(query: str, max_papers: int) -> List[Dict[str, Any]]:
    """
    End-to-end: search -> (maybe) PDF -> deep analysis.
    """
    papers = search_papers(query, max_results=max_papers)
    corpus = []
    for p in papers:
        analysis = analyze_paper_full(p)
        corpus.append(analysis)
    return corpus
