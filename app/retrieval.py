#Clause matching....

from typing import List, Dict
from embedding import embed_texts, index

TOP_K = 5
SCORE_THRESHOLD = 0.5  # Lower for MiniLM

def pinecone_query(question: str, namespace: str, top_k: int = TOP_K):
    q_emb = embed_texts([question])[0]
    resp = index.query(q_emb, top_k=top_k, include_metadata=True, namespace=namespace)
    matches = []
    for m in resp.get("matches", []):
        matches.append({
            "id": m["id"],
            "score": m.get("score", 0.0),
            "text": m["metadata"].get("text"),
            "page": m["metadata"].get("page")
        })
    return matches

def retrieve_for_question(question: str, namespace: str, min_score: float = SCORE_THRESHOLD):
    matches = pinecone_query(question, namespace)
    if not matches:
        return []
    filtered = [m for m in matches if m["score"] >= min_score]
    if not filtered:
        return matches[:3]
    return filtered
