import json, re, os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


class Invoice(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("currency")
    @classmethod
    def cur(cls, v):
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError("bad currency")
        return v

    @field_validator("date")
    @classmethod
    def dt(cls, v):
        m = re.search(r"\d{4}-\d{2}-\d{2}", v)
        if not m:
            raise ValueError("bad date")
        return m.group(0)


SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}


def regex_extract(text: str) -> dict:
    # currency
    cur = re.search(r"\b(USD|EUR|GBP)\b", text, re.I)
    currency = cur.group(1).upper() if cur else next(
        (c for s, c in SYMBOLS.items() if s in text), "USD")

    # date: prefer one near "due"
    dates = list(re.finditer(r"\d{4}-\d{2}-\d{2}", text))
    date = ""
    for m in dates:
        ctx = text[max(0, m.start() - 40):m.start()].lower()
        if "due" in ctx or "pay" in ctx:
            date = m.group(0)
            break
    if not date and dates:
        date = dates[-1].group(0)

    # amount: prefer one near total/due/amount, else any number tied to currency
    num = r"(\d[\d,]*(?:\.\d+)?)"
    amount = None
    for pat in [
        rf"(?:total|amount|balance)[^\d\n]{{0,30}}(?:USD|EUR|GBP|[$€£])?\s*{num}",
        rf"(?:USD|EUR|GBP|[$€£])\s*{num}",
        rf"{num}\s*(?:USD|EUR|GBP)",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            amount = float(m.group(1).replace(",", ""))
            break

    # vendor
    vendor = ""
    m = re.search(
        r"([A-Z][\w&\-]*(?:\s+[A-Z][\w&\-]*){0,3}?\s+"
        r"(?:Industries|Ltd|Inc|LLC|Corp|GmbH|Co|Group|Holdings|Solutions)\b\.?"
        r"(?:\s+(?:Ltd|Inc|LLC)\.?)?)", text)
    if m:
        vendor = m.group(1).strip()
    else:
        m = re.search(r"(?:from|vendor|billed by|supplier)\s*[:\-]?\s*([^\n,;]+)", text, re.I)
        if m:
            vendor = m.group(1).strip()

    return {"vendor": vendor, "amount": amount or 0.0,
            "currency": currency, "date": date}


async def llm_extract(text: str) -> dict | None:
    prompt = (
        "Extract invoice fields. Reply with JSON only, keys: vendor (string), "
        "amount (number, total due), currency (3-letter uppercase code), "
        "date (payment due date, YYYY-MM-DD).\n\nInvoice:\n" + text
    )
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(OLLAMA_URL, json={
                "model": MODEL, "prompt": prompt,
                "format": "json", "stream": False,
                "options": {"temperature": 0}})
            r.raise_for_status()
            return json.loads(r.json()["response"])
    except Exception:
        return None


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=Invoice)
async def extract(request: Request):
    # Parse manually so a valid body can never trigger a signature-mismatch 422
    try:
        body = await request.json()
        text = body.get("text") if isinstance(body, dict) else None
    except Exception:
        text = None
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=422, content={"detail": "text is required"})

    fallback = regex_extract(text)
    llm = await llm_extract(text)

    # Merge: trust the LLM only if its values are grounded in the text
    result = dict(fallback)
    if isinstance(llm, dict):
        v = str(llm.get("vendor", "")).strip()
        if v and v.lower() in text.lower():
            result["vendor"] = v
        try:
            a = float(str(llm.get("amount")).replace(",", ""))
            if f"{a:g}" in text.replace(",", "") or f"{a:.2f}" in text.replace(",", ""):
                result["amount"] = a
        except Exception:
            pass
        c = str(llm.get("currency", "")).upper()
        if c in ("USD", "EUR", "GBP") and c in text.upper():
            result["currency"] = c
        d = str(llm.get("date", ""))
        if d in text:
            result["date"] = d

    try:
        return Invoice(**result)
    except Exception:
        return JSONResponse(status_code=422, content={"detail": "could not extract"})
