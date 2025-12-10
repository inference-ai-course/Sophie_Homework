# tool_registry.py
from tools import search_arxiv, calculate
from qa_tool import search_interview_qa

TOOL_REGISTRY = {
    "search_arxiv": search_arxiv,
    "calculate": calculate,
    "search_interview_qa": search_interview_qa  # ✅ NEW TOOL
}