# tools.py
from sympy import sympify

def search_arxiv(query: str) -> str:
    """
    Simulate an arXiv search or return a dummy passage.
    """
    return f"[arXiv snippet related to '{query}': Recent studies explore theoretical foundations and experimental results.]"

def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression using sympy.
    """
    try:
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Math Error: {e}"
