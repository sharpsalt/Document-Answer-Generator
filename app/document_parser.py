# Used for parsing PDF/DOCX/Email Parsing# Used for parsing PDF/DOCX/Email Parsing

import io
import re
import requests
from typing import List, Dict
import pdfplumber
import docx

def download_bytes(url: str, timeout: int = 15) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def _split_semantic(text: str, min_chunk_chars: int = 400, max_chunk_chars: int = 1200) -> List[Dict]:
    text = re.sub(r'\r\n?', '\n', text).strip()
    parts = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for p in paragraphs:
        subparts = re.split(r'(?m)(?=^\s*\d+[\.\)]\s+|^Clause\s+\d+:|^\s*[-•\*]\s+)', p)
        for s in subparts:
            s = s.strip()
            if s:
                parts.append(s)
    chunks = []
    cur = ""
    cid = 0
    for part in parts:
        if not cur:
            cur = part
        elif len(cur) + len(part) <= max_chunk_chars:
            cur = cur + "\n\n" + part
        else:
            if len(cur) < min_chunk_chars and chunks:
                chunks[-1]["text"] += "\n\n" + cur
            else:
                chunks.append({"chunk_id": f"c{cid}", "text": cur})
                cid += 1
            cur = part
    if cur:
        if len(cur) < min_chunk_chars and chunks:
            chunks[-1]["text"] += "\n\n" + cur
        else:
            chunks.append({"chunk_id": f"c{cid}", "text": cur})
    return chunks

def parse_pdf_bytes(content: bytes) -> List[Dict]:
    chunks = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            page_chunks = _split_semantic(text)
            for chunk in page_chunks:
                chunk["page"] = pno
                chunks.append(chunk)
    return chunks

def parse_docx_bytes(content: bytes) -> List[Dict]:
    doc = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs)
    chunks = _split_semantic(text)
    for c in chunks:
        c.setdefault("page", None)
    return chunks

def parse_document(url: str) -> List[Dict]:
    content = download_bytes(url)
    lurl = url.lower()
    if lurl.endswith(".pdf") or b"%PDF" in content[:10]:
        return parse_pdf_bytes(content)
    elif lurl.endswith(".docx"):
        return parse_docx_bytes(content)
    else:
        text = content.decode(errors="ignore")
        chunks = _split_semantic(text)
        for c in chunks:
            c.setdefault("page", None)
        return chunks
