# quiz.py
"""
LLM Quiz Solver - FastAPI endpoint.

This file implements /solve which verifies a secret, opens the provided quiz URL
in a headless browser (Playwright), extracts data and attempts to submit an answer.
Keep this file free of any README/markdown text — only valid Python code.
"""

import os
import asyncio
import json
import re
import base64
import tempfile
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Playwright
from playwright.async_api import async_playwright

# PDF parsing & tables
import pdfplumber
import pandas as pd

# Configuration from environment
QUIZ_SECRET = os.environ.get("QUIZ_SECRET")
STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL") or "student@example.com"

# Timeouts
OVERALL_TIMEOUT_SECONDS = int(os.environ.get("OVERALL_TIMEOUT_SECONDS", "170"))  # < 180s (3 min)
NAV_TIMEOUT_MS = int(os.environ.get("NAV_TIMEOUT_MS", "30000"))

app = FastAPI(title="LLM Quiz Solver Endpoint")


class SolveRequest(BaseModel):
    email: str
    secret: str
    url: str


async def fetch_page_content(playwright, url: str) -> Dict[str, Any]:
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
    await asyncio.sleep(0.5)
    html = await page.content()
    text = await page.inner_text("body") if await page.query_selector("body") else html
    anchors = await page.eval_on_selector_all('a', 'els => els.map(a => ({href: a.href, text: a.innerText}))')
    pres = await page.eval_on_selector_all('pre, code', 'els => els.map(e => e.innerText)')
    submit_url = None
    m = re.search(r"https?://[\w\-\./?=&%:]+/submit[\w\-\./?=&%:]*", text)
    if m:
        submit_url = m.group(0)
    file_links = [a["href"] for a in anchors if a.get("href")]
    await browser.close()
    return {"html": html, "text": text, "anchors": anchors, "pres": pres, "submit_url": submit_url, "file_links": file_links}


async def download_file(url: str, client: httpx.AsyncClient) -> Optional[bytes]:
    try:
        r = await client.get(url, timeout=30.0)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def parse_sum_value_from_html(html: str) -> Optional[float]:
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if "value" in cols:
            try:
                s = df.iloc[:, cols.index("value")].dropna().astype(float).sum()
                return float(s)
            except Exception:
                continue
    return None


def parse_text_for_instructions(text: str) -> Dict[str, Any]:
    lower = text.lower()
    result = {"question": None, "operation": None}
    if "sum" in lower and ('"value"' in lower or "value column" in lower):
        result["operation"] = "sum_value_column"
    if "download" in lower and ("file" in lower or "pdf" in lower):
        result["question"] = text
    return result


def parse_pdf_for_table_sum(pdf_bytes: bytes) -> Optional[float]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            path = tmp.name
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for table in tables:
                    try:
                        df = pd.DataFrame(table)
                        df.columns = df.iloc[0]
                        df = df[1:]
                    except Exception:
                        try:
                            df = pd.DataFrame(table)
                        except Exception:
                            continue
                    cols = [str(c).lower() for c in df.columns]
                    if "value" in cols:
                        try:
                            s = pd.to_numeric(df.iloc[:, cols.index("value")], errors="coerce").dropna().sum()
                            return float(s)
                        except Exception:
                            continue
        return None
    except Exception:
        return None


async def solve_quiz_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = payload.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")

    async with async_playwright() as p:
        page_data = await fetch_page_content(p, url)

    text = page_data.get("text", "")
    html = page_data.get("html", "")
    anchors = page_data.get("anchors", [])
    pres = page_data.get("pres", [])
    submit_url = page_data.get("submit_url")
    file_links = page_data.get("file_links", [])

    instr = parse_text_for_instructions(text + "\n" + "\n".join(pres))
    answer = None
    if instr.get("operation") == "sum_value_column":
        s = parse_sum_value_from_html(html)
        if s is not None:
            answer = s

    async with httpx.AsyncClient() as client:
        for link in file_links:
            if not link.lower().startswith("http"):
                continue
            content = await download_file(link, client)
            if not content:
                continue
            if link.lower().endswith(".pdf") or content[:4] == b"%PDF":
                s = parse_pdf_for_table_sum(content)
                if s is not None:
                    answer = s
                    break
            if b"<table" in content[:10000]:
                try:
                    s = parse_sum_value_from_html(content.decode("utf-8", errors="ignore"))
                    if s is not None:
                        answer = s
                        break
                except Exception:
                    pass
            try:
                txt = content.decode("utf-8", errors="ignore")
                m = re.search(r'atob\(`([A-Za-z0-9+/=\n ]+)`\)', txt)
                if m:
                    dec = base64.b64decode(m.group(1))
                    try:
                        j = json.loads(dec)
                        if isinstance(j, dict) and "answer" in j:
                            answer = j["answer"]
                            break
                    except Exception:
                        pass
            except Exception:
                pass

    if answer is None:
        m = re.search(r"answer\W*[:=]\W*([\d,\.]+)", text, re.IGNORECASE)
        if m:
            try:
                answer = float(m.group(1).replace(",", ""))
            except Exception:
                pass

    if answer is None:
        return {"success": False, "reason": "Could not determine answer from page or linked files", "submit_url": submit_url}

    if not submit_url:
        for a in anchors:
            href = a.get("href")
            if href and "/submit" in href:
                submit_url = href
                break

    if not submit_url:
        return {"success": False, "reason": "No submit URL found on the page", "answer": answer}

    submit_payload = {
        "email": payload.get("email", STUDENT_EMAIL),
        "secret": payload.get("secret"),
        "url": payload.get("url"),
        "answer": answer,
    }

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(submit_url, json=submit_payload, timeout=30.0)
            r.raise_for_status()
            try:
                j = r.json()
            except Exception:
                j = {"status_code": r.status_code, "text": r.text}
            return {"success": True, "submit_response": j, "answer": answer}
        except Exception as e:
            return {"success": False, "reason": f"Submission failed: {e}", "answer": answer}



@app.get("/")
async def root():
    return {
        "service": "LLM Quiz Solver",
        "status": "running",
        "endpoints": {
            "solve (POST)": "/solve"
        },
        "note": "POST to /solve with JSON {email, secret, url}"
    }

@app.get("/health")
async def health():
    # simple health-check: responds 200 when app is up
    return {"status": "ok"}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root_html():
    return """
    <html>
      <head><title>LLM Quiz Solver</title></head>
      <body>
        <h1>LLM Quiz Solver</h1>
        <p>Service is running. Use <code>POST /solve</code> with JSON <code>{email, secret, url}</code>.</p>
      </body>
    </html>
    """




@app.post("/solve")
async def solve_endpoint(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if "secret" not in payload or "email" not in payload or "url" not in payload:
        raise HTTPException(status_code=400, detail="Missing required fields (email, secret, url)")

    if QUIZ_SECRET is None:
        raise HTTPException(status_code=500, detail="Server not configured with QUIZ_SECRET environment variable")

    if payload["secret"] != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        result = await asyncio.wait_for(solve_quiz_task(payload), timeout=OVERALL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Solver timed out")

    return JSONResponse(content=result)
