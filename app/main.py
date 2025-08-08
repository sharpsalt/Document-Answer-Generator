# main.py
import os
import asyncio
import hashlib
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from typing import List, Dict, Any
from pydantic import BaseModel
from document_parser import parse_document
from embedding import upsert_chunks, retrieve_for_question
from logic import synthesize_answer
from utils import setup_logger

EXPECTED_BEARER = os.getenv(
    "TEAM_BEARER_TOKEN",
    "2eb381aba879111710d913f8ae4963621bfb6f29350adf5d2b210bd726fb21c6"
)
INDEX_NAMESPACE_PREFIX = "doc"

logger = setup_logger("hackrx-main")
app = FastAPI(title="HackRx /hackrx/run")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerItem(BaseModel):
    question: str
    answer_text: str
    confidence: float
    evidence: List[Dict[str, Any]]
    rules_fired: List[str]
    llm_rationale: str = None

class QueryResponse(BaseModel):
    answers: List[AnswerItem]

def namespace_for_url(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{INDEX_NAMESPACE_PREFIX}-{h}"

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(req: QueryRequest, authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split()[1]
    if token != EXPECTED_BEARER:
        raise HTTPException(status_code=403, detail="Invalid token")

    if not req.documents or not req.questions:
        raise HTTPException(status_code=400, detail="documents and questions required")

    ns = namespace_for_url(req.documents)
    logger.info(f"Processing document: {req.documents} -> namespace={ns}")

    chunks = await asyncio.get_event_loop().run_in_executor(None, parse_document, req.documents)
    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to parse document")

    await asyncio.get_event_loop().run_in_executor(None, upsert_chunks, chunks, ns)

    async def process_question(q: str):
        retrieved = await asyncio.get_event_loop().run_in_executor(None, retrieve_for_question, q, ns)
        if not retrieved:
            return {
                "question": q,
                "answer_text": "Information not available in provided document.",
                "confidence": 0.0,
                "evidence": [],
                "rules_fired": [],
                "llm_rationale": ""
            }
        synthesized = await asyncio.get_event_loop().run_in_executor(None, synthesize_answer, q, retrieved)
        return {
            "question": q,
            "answer_text": synthesized.get("answer"),
            "confidence": float(synthesized.get("confidence", 0.0)),
            "evidence": synthesized.get("evidence", []),
            "rules_fired": synthesized.get("rules_fired", []),
            "llm_rationale": synthesized.get("rationale", "")
        }

    results = await asyncio.gather(*[process_question(q) for q in req.questions])
    return {"answers": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
