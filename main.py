# main.py
from fastapi import FastAPI, Query
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------
# Config / file paths
# ------------------------
OUTPUT_DIR = "output"
INDEX_FILE = f"{OUTPUT_DIR}/faiss.index"
META_FILE = f"{OUTPUT_DIR}/meta.pkl"

# ------------------------
# Create FastAPI app
# ------------------------
app = FastAPI(title="RAG PDF Search API")

# ------------------------
# Global objects (will be loaded at startup)
# ------------------------
faiss_index = None
metadata = None
model = None

# ------------------------
# Startup event to load index, metadata, and model
# ------------------------
@app.on_event("startup")
def startup_event():
    global faiss_index, metadata, model
    # Load FAISS index
    faiss_index = faiss.read_index(INDEX_FILE)
    print(f"Loaded index with {faiss_index.ntotal} vectors")
    
    # Load metadata
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    print(f"Loaded {len(metadata)} metadata items")
    
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loaded embedding model")

# ------------------------
# /search endpoint
# ------------------------
@app.get("/search")
async def search(
    q: str = Query(..., description="Query text"), 
    k: int = Query(3, description="Number of top results to return")
):
    """
    Search for top-k relevant chunks for a query string.
    """
    global faiss_index, metadata, model
    
    if faiss_index is None or metadata is None or model is None:
        return {"error": "Server not ready. Index or model not loaded."}

    # Embed query
    query_vector = model.encode([q]).astype("float32")  # shape: (1, dim)

    # Perform FAISS search
    distances, indices = faiss_index.search(query_vector, k)

    # Collect results with metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        item = metadata[idx].copy()
        item["distance"] = float(dist)
        results.append(item)

    return {"query": q, "results": results}
