# build_index.py
import sqlite3
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

DB_PATH = "hybrid_search.db"
FAISS_INDEX_PATH = "faiss.index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast & small
EMBED_DIM = 384  # all-MiniLM-L6-v2 dim; adjust if you change model

# Example: documents input
# Option A: folder of JSON files, each with {doc_id, title, author, year, keywords, text}
# Option B: one CSV with columns
# For brevity, this script expects a JSONL input file of docs:
DOCS_JSONL = "docs.jsonl"

def create_schema(conn):
    cur = conn.cursor()
    # main doc metadata
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY,
        title TEXT,
        author TEXT,
        year INTEGER,
        keywords TEXT
    )
    """)
    # chunk storage (we store full chunk text here)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER,
        content TEXT,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    )
    """)
    # FTS5 virtual table referencing chunks.content
    # Use content='' and explicit inserts into FTS (safer cross-platform)
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(content, content='')")
    conn.commit()

def chunk_text(text, chunk_size=400, overlap=50):
    # naive whitespace-based chunking
    tokens = text.split()
    i = 0
    chunks = []
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def load_documents(jsonl_path):
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def build(db_path=DB_PATH, docs_jsonl=DOCS_JSONL):
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    cur = conn.cursor()

    docs = load_documents(docs_jsonl)

    # Initialize embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    all_embeddings = []
    chunk_ids = []

    for doc in tqdm(docs, desc="Insert docs & chunks"):
        doc_id = doc.get("doc_id")
        title = doc.get("title")
        author = doc.get("author")
        year = doc.get("year")
        keywords = doc.get("keywords", "")

        cur.execute("INSERT OR REPLACE INTO documents(doc_id, title, author, year, keywords) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, title, author, year, keywords))

        # chunk
        text = doc.get("text", "")
        chunks = chunk_text(text)
        # insert each chunk and insert into FTS doc_chunks as well
        for chunk in chunks:
            cur.execute("INSERT INTO chunks(doc_id, content) VALUES (?, ?)", (doc_id, chunk))
            # get last inserted chunk_id
            chunk_id = cur.lastrowid
            # Insert the chunk into FTS index
            cur.execute("INSERT INTO doc_chunks(rowid, content) VALUES (?, ?)", (chunk_id, chunk))
            chunk_ids.append(chunk_id)
    conn.commit()

    # Now compute embeddings for all chunk texts (do batch)
    cur.execute("SELECT chunk_id, content FROM chunks ORDER BY chunk_id")
    rows = cur.fetchall()
    texts = [r[1] for r in rows]
    ids = [r[0] for r in rows]

    print(f"Computing embeddings for {len(texts)} chunks...")
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embed batches"):
        batch_texts = texts[i:i+batch_size]
        emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float32')

    # Build FAISS index (IndexFlatIP for cosine when embeddings are normalized)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    # Save mapping from faiss index positions to chunk_id in sqlite table
    # We'll create a mapping table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS faiss_mapping (
        faiss_id INTEGER PRIMARY KEY,  -- position in the index
        chunk_id INTEGER
    )
    """)
    cur.executemany("INSERT INTO faiss_mapping(faiss_id, chunk_id) VALUES (?, ?)",
                    [(i, ids[i]) for i in range(len(ids))])
    conn.commit()

    conn.close()
    print("Build complete.")

if __name__ == "__main__":
    build()
