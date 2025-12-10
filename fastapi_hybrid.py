from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any
from search import hybrid_search
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Search API",
    description="Week 5: FAISS Vector Search + SQLite Keyword Search Hybrid Retrieval",
    version="1.0.0"
)

# Initialize searcher
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
searcher = hybrid_search(
    faiss_index_path="embeddings/faiss.index",
    db_path="arxiv_hybrid_final.db", 
    embedding_model=embedding_model
)

# Response models
class SearchResult(BaseModel):
    chunk_id: int
    content: str
    paper_title: str
    authors: str
    year: int
    arxiv_id: str
    source_pdf: str
    hybrid_score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    method: str

class ComparisonResponse(BaseModel):
    query: str
    vector_only: List[str]
    keyword_only: List[str]
    hybrid: List[str]

@app.get("/", summary="API Root")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hybrid Search API",
        "version": "1.0.0",
        "endpoints": {
            "/hybrid_search": "Hybrid vector + keyword search",
            "/vector_search": "Vector-only semantic search", 
            "/keyword_search": "Keyword-only exact match search",
            "/compare_methods": "Compare all three search methods"
        }
    }

@app.get("/hybrid_search", response_model=SearchResponse, summary="Hybrid Search")
async def hybrid_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Perform hybrid search combining FAISS vector similarity and SQLite keyword matching.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both search methods.
    """
    results = searcher.hybrid_search(q, k=k)
    
    return SearchResponse(
        query=q,
        results=results,
        total_results=len(results),
        method="hybrid_rrf"
    )

@app.get("/vector_search", response_model=SearchResponse, summary="Vector-Only Search")
async def vector_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Perform vector-only semantic search using FAISS.
    """
    vector_results = searcher.vector_search(q, k)
    # Convert to full document details
    chunk_ids = [r['chunk_id'] for r in vector_results]
    full_results = searcher._get_document_details(chunk_ids)
    
    # Add vector scores
    for result, vector_result in zip(full_results, vector_results):
        result['hybrid_score'] = vector_result['vector_score']
    
    return SearchResponse(
        query=q,
        results=full_results,
        total_results=len(full_results),
        method="vector_only"
    )

@app.get("/keyword_search", response_model=SearchResponse, summary="Keyword-Only Search")  
async def keyword_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results to return", ge=1, le=20)
):
    """
    Perform keyword-only exact match search using SQLite FTS5.
    """
    keyword_results = searcher.db.keyword_search(q, k)
    # Convert to full document details
    chunk_ids = [r['chunk_id'] for r in keyword_results]
    full_results = searcher._get_document_details(chunk_ids)
    
    # Add keyword scores
    for result, keyword_result in zip(full_results, keyword_results):
        result['hybrid_score'] = keyword_result['keyword_score']
    
    return SearchResponse(
        query=q,
        results=full_results,
        total_results=len(full_results),
        method="keyword_only"
    )

@app.get("/compare_methods", response_model=ComparisonResponse, summary="Compare Search Methods")
async def compare_methods(
    q: str = Query(..., description="Search query"),
    k: int = Query(3, description="Number of results to return", ge=1, le=10)
):
    """
    Compare results from all three search methods (vector, keyword, hybrid).
    Returns content previews for easy comparison.
    """
    # Vector search
    vector_results = searcher.vector_search(q, k)
    vector_chunk_ids = [r['chunk_id'] for r in vector_results]
    vector_docs = searcher._get_document_details(vector_chunk_ids)
    
    # Keyword search
    keyword_results = searcher.db.keyword_search(q, k)
    keyword_chunk_ids = [r['chunk_id'] for r in keyword_results]
    keyword_docs = searcher._get_document_details(keyword_chunk_ids)
    
    # Hybrid search
    hybrid_docs = searcher.hybrid_search(q, k)
    
    return ComparisonResponse(
        query=q,
        vector_only=[doc['content'][:200] + "..." for doc in vector_docs],
        keyword_only=[doc['content'][:200] + "..." for doc in keyword_docs],
        hybrid=[doc['content'][:200] + "..." for doc in hybrid_docs]
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting Hybrid Search API server...")
    print("Access the API at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
