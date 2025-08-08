#Embedding and Pinecone Logic will bee here..
# Embedding and Pinecone Logic will be here..

# embedding.py
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
embedder = SentenceTransformer(EMBED_MODEL)

FAISS_STORE = "vector_store.pkl"
INDEXES = {}

def load_index(namespace: str):
    if namespace in INDEXES:
        return INDEXES[namespace]
    if os.path.exists(f"{namespace}_{FAISS_STORE}"):
        with open(f"{namespace}_{FAISS_STORE}", "rb") as f:
            index, meta = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(384)
        meta = []
    INDEXES[namespace] = (index, meta)
    return index, meta

def save_index(namespace: str):
    if namespace in INDEXES:
        index, meta = INDEXES[namespace]
        with open(f"{namespace}_{FAISS_STORE}", "wb") as f:
            pickle.dump((index, meta), f)

def embed_texts(texts: List[str]) -> np.ndarray:
    return np.array(embedder.encode(texts, normalize_embeddings=True))

def upsert_chunks(chunks: List[Dict], namespace: str):
    index, meta = load_index(namespace)
    vecs = embed_texts([c["text"] for c in chunks])
    for i, c in enumerate(chunks):
        meta.append(c)
    index.add(vecs)
    INDEXES[namespace] = (index, meta)
    save_index(namespace)

def retrieve_for_question(question: str, namespace: str, top_k: int = 8):
    index, meta = load_index(namespace)
    if index.ntotal == 0:
        return []
    q_vec = embed_texts([question])
    scores, idxs = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < len(meta):
            r = meta[idx].copy()
            r["score"] = float(score)
            r["id"] = f"{namespace}:{r['chunk_id']}"
            results.append(r)
    return results
