# search.py
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import math

DB_PATH = "hybrid_search.db"
FAISS_INDEX_PATH = "faiss.index"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load model and index on import (or call init)
model = SentenceTransformer(MODEL_NAME)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

def get_embedding(text):
    emb = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return emb[0].astype('float32')

def faiss_search(query, k=10):
    q_emb = get_embedding(query).reshape(1, -1)
    D, I = faiss_index.search(q_emb, k)  # D: similarity scores (inner product)
    scores = D[0].tolist()
    indices = I[0].tolist()
    # map faiss index pos to chunk_id and chunk info
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    results = []
    for faiss_pos, score in zip(indices, scores):
        if faiss_pos < 0:
            continue
        cur.execute("SELECT chunk_id, doc_id, content FROM chunks WHERE chunk_id = (SELECT chunk_id FROM faiss_mapping WHERE faiss_id = ?)", (faiss_pos,))
        row = cur.fetchone()
        if row:
            chunk_id, doc_id, content = row
            cur.execute("SELECT title, author, year, keywords FROM documents WHERE doc_id = ?", (doc_id,))
            docmeta = cur.fetchone()
            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": content,
                "docmeta": docmeta,
                "score": float(score)
            })
    conn.close()
    return results

def fts_search(query, k=10):
    # Use simple MATCH against doc_chunks; return chunk rows and rely on SQLite ordering by rank
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT rowid, content
        FROM doc_chunks
        WHERE doc_chunks MATCH ?
        LIMIT ?
    """, (query, k))
    rows = cur.fetchall()
    results = []
    for row in rows:
        chunk_id, content = row
        # fetch doc_id, title
        cur2 = conn.cursor()
        cur2.execute("SELECT doc_id FROM chunks WHERE chunk_id = ?", (chunk_id,))
        doc_row = cur2.fetchone()
        doc_id = doc_row[0] if doc_row else None
        cur2.execute("SELECT title, author, year, keywords FROM documents WHERE doc_id = ?", (doc_id,))
        docmeta = cur2.fetchone()
        results.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "docmeta": docmeta,
            "score": None  # SQLite does not easily expose score by default here
        })
    conn.close()
    return results

def build_bm25_corpus():
    # loads all chunks into memory and build BM25 index
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, content FROM chunks ORDER BY chunk_id")
    rows = cur.fetchall()
    conn.close()
    docs = [r[1] for r in rows]
    chunk_ids = [r[0] for r in rows]
    tokenized = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunk_ids, docs

# Build BM25 index once (call at startup in real app)
BM25_INDEX, BM25_IDS, BM25_DOCS = build_bm25_corpus()

def bm25_search(query, k=10):
    tokenized = query.split()
    scores = BM25_INDEX.get_scores(tokenized)
    top_n_idx = np.argsort(scores)[::-1][:k]
    results = []
    conn = sqlite3.connect(DB_PATH)
    for i in top_n_idx:
        chunk_id = BM25_IDS[i]
        content = BM25_DOCS[i]
        cur = conn.cursor()
        cur.execute("SELECT doc_id FROM chunks WHERE chunk_id = ?", (chunk_id,))
        doc_row = cur.fetchone()
        doc_id = doc_row[0] if doc_row else None
        cur.execute("SELECT title, author, year, keywords FROM documents WHERE doc_id = ?", (doc_id,))
        docmeta = cur.fetchone()
        results.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            "docmeta": docmeta,
            "score": float(scores[i])
        })
    conn.close()
    return results

# -------------------
# Hybrid merging
# -------------------

def normalize_scores(res_list, score_key="score"):
    # Normalize numbers to 0..1 across results
    scores = [r.get(score_key, 0.0) for r in res_list]
    if not scores:
        return res_list
    min_s, max_s = min(scores), max(scores)
    for r in res_list:
        s = r.get(score_key, 0.0)
        if max_s == min_s:
            norm = 1.0
        else:
            norm = (s - min_s) / (max_s - min_s)
        r["_norm_score"] = norm
    return res_list

def merge_weighted(faiss_res, key_res, alpha=0.6, k=10):
    # Build maps by chunk_id
    for r in faiss_res:
        if r.get("score") is None:
            r["score"] = 0.0
    for r in key_res:
        if r.get("score") is None:
            r["score"] = 0.0

    normalize_scores(faiss_res, "score")
    normalize_scores(key_res, "score")
    key_map = {r["chunk_id"]: r for r in key_res}
    combined = []
    seen = set()
    # include all candidates from both sets
    for r in faiss_res + key_res:
        cid = r["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        v = next((x for x in faiss_res if x["chunk_id"] == cid), {"_norm_score": 0.0})
        kscore = key_map.get(cid, {"_norm_score": 0.0})
        score = alpha * v.get("_norm_score", 0.0) + (1 - alpha) * kscore.get("_norm_score", 0.0)
        combined.append((cid, score, v, kscore))
    combined.sort(key=lambda x: x[1], reverse=True)
    # return enriched results (combine metadata)
    out = []
    for cid, score, v, k in combined[:k]:
        # Use vector content if exists else key content
        content = v.get("content") or k.get("content")
        docmeta = v.get("docmeta") or k.get("docmeta")
        out.append({
            "chunk_id": cid,
            "score": score,
            "content": content,
            "docmeta": docmeta
        })
    return out

def reciprocal_rank_fusion(rank_lists, k=10, c=60):
    # rank_lists: list of lists, each list is [chunk_id ordered by rank]
    # RRF score: sum_{lists} 1 / (c + rank)
    # We'll accept lists where items are dicts with 'chunk_id'
    scores = {}
    for lst in rank_lists:
        for rank, item in enumerate(lst):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + (rank + 1))  # rank 1 -> +1/(c+1)
    sorted_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    # fetch metadata
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    results = []
    for cid, s in sorted_ids:
        cur.execute("SELECT content, doc_id FROM chunks WHERE chunk_id = ?", (cid,))
        row = cur.fetchone()
        if row:
            content, doc_id = row
            cur.execute("SELECT title, author, year, keywords FROM documents WHERE doc_id = ?", (doc_id,))
            docmeta = cur.fetchone()
            results.append({
                "chunk_id": cid,
                "score": float(s),
                "content": content,
                "docmeta": docmeta
            })
    conn.close()
    return results

# Example helper to run a hybrid search
def hybrid_search(query, k=5, alpha=0.6, use_bm25=False):
    faiss_res = faiss_search(query, k=k)
    if use_bm25:
        key_res = bm25_search(query, k=k)
    else:
        key_res = fts_search(query, k=k)
    merged = merge_weighted(faiss_res, key_res, alpha=alpha, k=k)
    return {
        "faiss": faiss_res,
        "keyword": key_res,
        "merged": merged
    }

if __name__ == "__main__":
    q = "example query here"
    out = hybrid_search(q, k=5, alpha=0.6, use_bm25=True)
    import pprint; pprint.pp(out["merged"])
