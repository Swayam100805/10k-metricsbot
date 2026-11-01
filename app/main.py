import sys, re, hashlib
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE / "src"))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from retrieval import Retriever
from qa_compare import extractive_answer, summarize     # loads the QA + FLAN-T5 models once at startup

DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models" / "risk_classifier"

# ---------- Risk classifier (Exp 04) ----------
clf_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
clf_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()

def classify_risk(text):
    enc = clf_tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = clf_model(**enc).logits[0].softmax(-1).tolist()
    scores = {clf_model.config.id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    top = max(scores, key=scores.get)
    return top, scores

def mostly_numbers(text):
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(c.isdigit() for c in chars) / len(chars) > 0.2

# ---------- Retriever cache (building the index for a 200k-word filing takes a few seconds) ----------
_cache = {}

def get_retriever(req):
    if req.filing_name:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", req.filing_name):
            raise HTTPException(400, "filing_name may only contain letters, digits, _ and -")
        path = DATA_DIR / f"{req.filing_name}.txt"
        if not path.exists():
            raise HTTPException(404, f"No filing named '{req.filing_name}' in data/")
        key, load = req.filing_name, (lambda: path.read_text(encoding="utf-8"))
    else:
        key, load = hashlib.md5(req.filing_text.encode("utf-8")).hexdigest(), (lambda: req.filing_text)
    if key not in _cache:
        if len(_cache) >= 5:
            _cache.pop(next(iter(_cache)))
        _cache[key] = Retriever(load())
    return _cache[key]

# ---------- API ----------
app = FastAPI(
    title="10K-MetricsBot",
    description="Numeric question answering and risk classification for SEC 10-K filings. "
                "Numbers come only from an extractive model (exact text from the filing); "
                "the generative model is used only for the short summary.",
    version="1.0.0")

class FilingQuery(BaseModel):
    question: str = Field(..., min_length=5,
                          examples=["What were Apple's total net sales in fiscal 2025?"])
    filing_name: Optional[str] = Field(None, description="A filing stored in data/ (e.g. tech_10k or bank_10k)",
                                       examples=["tech_10k"])
    filing_text: Optional[str] = Field(None, description="Or paste the filing text here instead")

class RiskQuery(BaseModel):
    text: str = Field(..., min_length=20,
                      examples=["We may be unable to raise capital on acceptable terms when our obligations come due."])

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "app" / "static" / "index.html")

@app.get("/filings")
def list_filings():
    return {"filings": sorted(p.stem for p in DATA_DIR.glob("*.txt"))}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify-risk")
def classify_endpoint(req: RiskQuery):
    label, scores = classify_risk(req.text)
    return {"risk_category": label, "scores": scores}

@app.post("/query-filing")
def query_filing(req: FilingQuery):
    if bool(req.filing_name) == bool(req.filing_text):
        raise HTTPException(400, "Provide exactly one of filing_name or filing_text")
    if req.filing_text and len(req.filing_text.split()) < 30:
        raise HTTPException(400, "filing_text is too short to search")

    retriever = get_retriever(req)
    ans = extractive_answer(req.question, retriever)
    if not ans["answer"]:
        raise HTTPException(422, "No answer found in the filing")

    passage = ans["context"]
    if mostly_numbers(passage):
        risk, risk_scores, risk_note = None, None, "Source passage is mostly table numbers, so it was not classified."
    else:
        risk, risk_scores = classify_risk(passage)
        risk_note = None

    return {
        "question": req.question,
        "answer": ans["answer"],
        "answer_is_exact_text_from_filing": True,
        "confidence": round(ans["score"], 4),
        "source_passage": passage,
        "risk_category": risk,
        "risk_scores": risk_scores,
        "risk_note": risk_note,
        "passage_summary": summarize(passage, ans["answer"]),
        "warning": "Check the answer against the source passage. Financial tables can be misread (for example, wrong year column).",
    }