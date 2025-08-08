#Domain specific LOgic yaha pe hoga...

# logic.py
import re
from typing import List, Dict
from llm import synthesize

def _extract_numeric_and_unit(text: str):
    m = re.search(r'(\d+(?:\.\d+)?)(\s*(days?|months?|years?|%|percent|% of the Sum Insured|per cent))', text, flags=re.I)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None

def synthesize_answer(question: str, retrieved: List[Dict]) -> Dict:
    rules = []
    evidence = []
    for r in retrieved:
        txt = r.get("text", "")
        val, unit = _extract_numeric_and_unit(txt)
        if val:
            rules.append("numeric_extraction")
            evidence.append({
                "doc_id": r.get("id"),
                "page": r.get("page"),
                "text": txt,
                "char_start": r.get("char_start"),
                "char_end": r.get("char_end"),
                "score": r.get("score")
            })
            return {
                "answer": f"{val} {unit}".strip(),
                "confidence": 0.95,
                "evidence": evidence,
                "rationale": "Found numeric clause via regex in retrieved snippets.",
                "rules_fired": rules
            }
    llm_result = synthesize(question, retrieved)
    answer = llm_result.get("answer")
    conf = llm_result.get("confidence", 0.5)
    rationale = llm_result.get("rationale", "")
    evidence = [{
        "doc_id": r.get("id"),
        "page": r.get("page"),
        "text": r.get("text"),
        "char_start": r.get("char_start"),
        "char_end": r.get("char_end"),
        "score": r.get("score")
    } for r in retrieved[:3]]
    rules.append("llm_synthesis")
    return {
        "answer": answer,
        "confidence": conf,
        "evidence": evidence,
        "rationale": rationale,
        "rules_fired": rules
    }
