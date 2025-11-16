"""
rag_pipeline.py
Full pipeline:
1) Extract text from PDFs (using PyMuPDF)
2) Chunk text (sliding window)
3) Embed chunks (sentence-transformers)
4) Create FAISS index and save it
5) Save chunk metadata to JSON/pickle
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF
from tqdm import tqdm
import argparse
import pickle

# ---------------------------
# Config (change as needed)
# ---------------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-d model
OUTPUT_DIR = Path("rag_project/output")
CHUNKS_JSON = OUTPUT_DIR / "chunks.json"
INDEX_FILE = OUTPUT_DIR / "faiss.index"
META_FILE = OUTPUT_DIR / "meta.pkl"   # metadata for chunks (page, source, text)
DIM = 384

# ---------------------------
# Utilities
# ---------------------------
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Return list of page texts (one element per PDF page)."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages

def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """
    Simple whitespace tokenizer-based chunking.
    You can replace with a tokenizer-based approach (tiktoken) for better token control.
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# ---------------------------
# Pipeline
# ---------------------------
def build_index_from_pdfs(pdf_paths: List[str],
                          model_name: str = MODEL_NAME,
                          max_tokens: int = 512,
                          overlap: int = 50):
    """
    Given list of pdfs, build embeddings, create faiss index, and save metadata.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = SentenceTransformer(model_name)
    all_chunks = []
    metadata = []  # list of dicts: {source, page, chunk_id, text}

    for pdf_path in pdf_paths:
        pages = extract_text_from_pdf(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = chunk_text(page_text, max_tokens=max_tokens, overlap=overlap)
            for i, chunk in enumerate(page_chunks):
                # store metadata
                meta = {
                    "source": str(pdf_path),
                    "page": page_num,
                    "chunk_id": len(all_chunks),  # index into all_chunks
                    "text": chunk
                }
                metadata.append(meta)
                all_chunks.append(chunk)

    print(f"[+] Extracted {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")

    # embed in batches
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch = all_chunks[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    # create faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[+] Built FAISS Index with {index.ntotal} vectors (dim={dim})")

    # Save index and metadata
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    # Save chunk texts as JSON (optional)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump([m["text"] for m in metadata], f, ensure_ascii=False, indent=2)

    print(f"[+] Saved index -> {INDEX_FILE}")
    print(f"[+] Saved metadata -> {META_FILE}")
    print(f"[+] Saved chunks json -> {CHUNKS_JSON}")

    return index, metadata

def load_index_and_meta(index_path: str = INDEX_FILE, meta_path: str = META_FILE):
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve(query: str, model: SentenceTransformer, index, metadata, k: int = 3):
    qvec = model.encode([query]).astype("float32")
    distances, indices = index.search(qvec, k)
    results = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx].copy()
        meta["distance"] = float(distances[0][i])
        results.append(meta)
    return results

# ---------------------------
# Command-line interface
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG pipeline")
    parser.add_argument("--pdfs", nargs="+", help="PDF files to process", required=False)
    parser.add_argument("--build", action="store_true", help="Build index from provided PDFs")
    parser.add_argument("--query", type=str, help="Run a test query against saved index", required=False)
    args = parser.parse_args()

    if args.build:
        if not args.pdfs:
            raise ValueError("Provide --pdfs file1.pdf file2.pdf ...")
        build_index_from_pdfs(args.pdfs)
    elif args.query:
        model = SentenceTransformer(MODEL_NAME)
        index, metadata = load_index_and_meta()
        results = retrieve(args.query, model, index, metadata, k=3)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("Use --build to build an index, or --query to test a query")

if __name__ == "__main__":
    main()
