# fastapi_app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from search import hybrid_search

app = FastAPI(title="Hybrid Search API")

class Result(BaseModel):
    chunk_id: int
    doc_id: int = None
    score: float
    content: str
    docmeta: List = None

class ResponseModel(BaseModel):
    query: str
    results: List[Result]

""" @app.get("/hybrid_search")
def hybrid_search_api(
    query: str,
    k: int = 3,
    alpha: float = 0.6,
    use_bm25: bool = True
):
    out = hybrid_search(query, k=k, alpha=alpha, use_bm25=use_bm25)
    results = []
    for r in out["merged"]:
        docmeta = list(r["docmeta"]) if r.get("docmeta") else []
        results.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "score": r["score"],
            "content": r["content"],
            "docmeta": docmeta
        })
    return {"query": query, "results": results} """

@app.get("/hybrid_search")
def hybrid_search_api(q: str, k: int = 3, alpha: float = 0.6):
    try:
        out = hybrid_search(q, k=k, alpha=alpha, use_bm25=True)
        return {"query": q, "results": out["merged"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting Hybrid Search API server...")
    print("Access the API at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# Run with:
# uvicorn fastapi_app:app --reload --port 8000
# then GET http://127.0.0.1:8000/hybrid_search?query=lead%20time&k=3&alpha=0.6
#, response_model=ResponseModel