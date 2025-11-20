# LLM Quiz Solver — Project scaffold

This canvas contains a ready-to-run Python server that implements the required POST endpoint which:

* Verifies the `secret` provided in incoming payloads.
* Loads the target quiz URL in a headless browser (Playwright) so JavaScript is executed.
* Attempts to parse the quiz page (tables, PDF links, file downloads, explicit instructions) and compute answers using heuristics.
* Submits answers to the submit endpoint provided on the quiz page.
* Returns the result to the original caller.

---

## Files included (in this single document)

### `main.py`

```python
"""
Simple FastAPI app that accepts POST requests with a quiz task and tries to solve it.

Usage (example):

$ export QUIZ_SECRET="your_secret_here"
$ export STUDENT_EMAIL="student@example.com"
$ uvicorn main:app --host 0.0.0.0 --port 8000

POST /solve with JSON: { "email": "student@example.com", "secret": "your_secret_here", "url": "https://example.com/quiz-834" }

Note: This implementation uses Playwright and pdfplumber for PDF parsing.
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
from playwright.async_api import async_playwright, Page

# PDF parsing
import pdfplumber
import pandas as pd

# Configuration from environment
QUIZ_SECRET = os.environ.get("QUIZ_SECRET")
STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL") or "student@example.com"
# optional: keep a GitHub URL, prompts etc. as env vars (set them from the Google Form)
GITHUB_URL = os.environ.get("GITHUB_URL")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT")
USER_PROMPT = os.environ.get("USER_PROMPT")

# Timeouts
OVERALL_TIMEOUT_SECONDS = int(os.environ.get("OVERALL_TIMEOUT_SECONDS", "170"))  # < 180s (3 min)
NAV_TIMEOUT_MS = int(os.environ.get("NAV_TIMEOUT_MS", "30000"))

app = FastAPI(title="LLM Quiz Solver Endpoint")

class SolveRequest(BaseModel):
    email: str
    secret: str
    url: str
    # allow extra fields


async def fetch_page_content(playwright, url: str) -> Dict[str, Any]:
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)

    # Wait a short while for client-side scripts to render
    await asyncio.sleep(0.5)

    html = await page.content()
    text = await page.inner_text("body") if await page.query_selector("body") else html
    # capture anchors and pre blocks
    anchors = await page.eval_on_selector_all('a', 'els => els.map(a => ({href: a.href, text: a.innerText}))')
    pres = await page.eval_on_selector_all('pre, code', 'els => els.map(e => e.innerText)')

    # find any obvious submit URL in the page text
    submit_url = None
    # look for https://.../submit or /submit
    m = re.search(r"https?://[\w\-./?=&%:]+/submit[\w\-./?=&%:]*", text)
    if m:
        submit_url = m.group(0)

    # gather any downloadable file links (hrefs)
    file_links = [a['href'] for a in anchors if a.get('href')]

    await browser.close()
    return {"html": html, "text": text, "anchors": anchors, "pres": pres, "submit_url": submit_url, "file_links": file_links}


async def download_file(url: str, client: httpx.AsyncClient) -> Optional[bytes]:
    try:
        r = await client.get(url, timeout=30.0)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"download_file error for {url}: {e}")
        return None


def parse_sum_value_from_html(html: str) -> Optional[float]:
    # Try pandas read_html to find tables
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if "value" in cols:
            # sum the column
            try:
                s = df.iloc[:, cols.index("value")].dropna().astype(float).sum()
                return float(s)
            except Exception:
                continue
    return None


def parse_text_for_instructions(text: str) -> Dict[str, Any]:
    # naive heuristics to extract question & expected operation
    lower = text.lower()
    result = {"question": None, "operation": None}
    # detect sum of "value" column
    if "sum" in lower and '"value"' in lower or "value column" in lower:
        result["operation"] = "sum_value_column"
    # detect download instruction
    if "download" in lower and ("file" in lower or "pdf" in lower):
        result["question"] = text
    return result


def parse_pdf_for_table_sum(pdf_bytes: bytes) -> Optional[float]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            path = tmp.name
        # try pdfplumber
        with pdfplumber.open(path) as pdf:
            # search pages for tables
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for table in tables:
                    # table is list of rows (list of lists). Try to make DataFrame
                    try:
                        df = pd.DataFrame(table)
                        # assume first row is headers if strings
                        df.columns = df.iloc[0]
                        df = df[1:]
                    except Exception:
                        try:
                            df = pd.DataFrame(table)
                        except Exception:
                            continue
                    cols = [str(c).lower() for c in df.columns]
                    if 'value' in cols:
                        try:
                            s = pd.to_numeric(df.iloc[:, cols.index('value')], errors='coerce').dropna().sum()
                            return float(s)
                        except Exception:
                            continue
        return None
    except Exception as e:
        print(f"parse_pdf_for_table_sum error: {e}")
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

    # parse instructions
    instr = parse_text_for_instructions(text + "\n" + "\n".join(pres))

    # attempt 1: parse HTML tables
    answer = None
    if instr.get("operation") == "sum_value_column":
        s = parse_sum_value_from_html(html)
        if s is not None:
            answer = s

    # attempt 2: download files and inspect them (prefer PDF)
    async with httpx.AsyncClient() as client:
        for link in file_links:
            # skip javascript links
            if not link.lower().startswith("http"):
                continue
            content = await download_file(link, client)
            if not content:
                continue
            # if PDF
            if link.lower().endswith('.pdf') or (content[:4] == b'%PDF'):
                s = parse_pdf_for_table_sum(content)
                if s is not None:
                    answer = s
                    break
            # try HTML as fallback
            if b"<table" in content[:10000]:
                try:
                    s = parse_sum_value_from_html(content.decode('utf-8', errors='ignore'))
                    if s is not None:
                        answer = s
                        break
                except Exception:
                    pass
            # check if content contains base64 encoded JSON with submit payload (some quizzes embed pre>base64)
            try:
                txt = content.decode('utf-8', errors='ignore')
                # common pattern: atob(`base64...`)
                m = re.search(r'atob\(`([A-Za-z0-9+/=\n ]+)`\)', txt)
                if m:
                    dec = base64.b64decode(m.group(1))
                    # try to parse JSON
                    try:
                        j = json.loads(dec)
                        # check for sample answer
                        if isinstance(j, dict) and 'answer' in j:
                            answer = j['answer']
                            break
                    except Exception:
                        pass
            except Exception:
                pass

    # attempt 3: naive number extraction if still missing
    if answer is None:
        # look for patterns like "sum is: 12345" or "answer: 12345"
        m = re.search(r"answer\W*[:=]\W*([\d,\.]+)", text, re.IGNORECASE)
        if m:
            try:
                answer = float(m.group(1).replace(',', ''))
            except Exception:
                pass

    # if still None, return an instructive response
    if answer is None:
        return {"success": False, "reason": "Could not determine answer from page or linked files", "submit_url": submit_url}

    # prepare submission
    if not submit_url:
        # try to extract typical submit endpoints from anchors
        for a in anchors:
            href = a.get('href')
            if href and '/submit' in href:
                submit_url = href
                break

    if not submit_url:
        return {"success": False, "reason": "No submit URL found on the page", "answer": answer}

    submit_payload = {
        "email": payload.get('email', STUDENT_EMAIL),
        "secret": payload.get('secret'),
        "url": payload.get('url'),
        "answer": answer
    }

    # submit and return the response
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


@app.post("/solve")
async def solve_endpoint(request: Request):
    # Validate JSON body
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # verify required fields
    if 'secret' not in payload or 'email' not in payload or 'url' not in payload:
        raise HTTPException(status_code=400, detail="Missing required fields (email, secret, url)")

    if QUIZ_SECRET is None:
        raise HTTPException(status_code=500, detail="Server not configured with QUIZ_SECRET environment variable")

    if payload['secret'] != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # run the solver with a global timeout (must respond within ~3 minutes)
    try:
        result = await asyncio.wait_for(solve_quiz_task(payload), timeout=OVERALL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Solver timed out")

    return JSONResponse(content=result)
```

### `requirements.txt`

```
fastapi
uvicorn[standard]
playwright
httpx
pandas
pdfplumber
python-multipart
pydantic
```

### `README.md` (usage notes)

````markdown
# LLM Quiz Solver

## Overview
This server exposes `/solve` which accepts POST requests with JSON payloads `{email, secret, url}`. It checks the secret and, if valid, opens the provided `url` in a headless browser, attempts to read the quiz instructions and data, computes an answer using several heuristics (HTML tables, downloadable PDFs, embedded JSON), and POSTs the answer to the submit endpoint found on the page.

## Setup

1. Create a Python 3.10+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Install Playwright browsers: `python -m playwright install`
4. Set required environment variables:

```bash
export QUIZ_SECRET="the_secret_you_put_in_the_google_form"
export STUDENT_EMAIL="your_email@example.com"
# optional:
export GITHUB_URL="https://github.com/you/repo"
export SYSTEM_PROMPT="..."
export USER_PROMPT="..."
````

5. Run the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

6. POST a JSON body to `/solve` from your evaluator. The server returns JSON describing success/failure and the remote submit response.

## Notes & limitations

* This project uses heuristics; depending on the variety of quiz pages, you will likely need to extend the parsing logic for each new quiz type (image OCR, non-tabular PDF parsing, audio transcription, API-auth headers, geo analysis, etc.).
* For more robust PDF table extraction, consider adding `camelot` (requires ghostscript) or `tabula-py` (Java).
* The solver is synchronous and limited by `OVERALL_TIMEOUT_SECONDS`. Make sure your server/container has enough CPU and network access.
* Keep your GitHub repo public with an MIT license when submitting for evaluation.

```

---

## Next steps & suggestions

- Add additional parsers (OCR via Tesseract, speech-to-text for audio, specialised CSV/Excel parsing).
- Hardening: sandbox downloads, validate input sizes, limit external network targets.
- Logging & observability: structured logs, traces to help debug quizzes you fail to parse.

---

*I created the scaffold files above (main.py, requirements.txt, README). You can copy them into your repository, add an MIT license file, and deploy.*

```
