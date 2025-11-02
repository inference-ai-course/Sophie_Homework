"""
arxiv_scraper.py
- Query arXiv for latest N papers in a category (e.g. cs.CL)
- For each paper: fetch /abs/ page, extract with trafilatura; screenshot with Playwright and OCR via pytesseract
- Save JSON: [{url, title, abstract, authors, date}, ...]
- Ensure output file <= 1MB by truncating abstracts if necessary

Requires:
  pip install feedparser requests trafilatura pytesseract playwright pillow
  python -m playwright install  # to install browser binaries
Also requires system Tesseract installed and accessible (or set pytesseract.pytesseract.tesseract_cmd).
"""

import os
import time
import json
import math
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Dict

import feedparser
import requests
import trafilatura
from PIL import Image
import pytesseract

# Playwright (sync API)
try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    raise RuntimeError("playwright not installed or browsers not installed. Run: pip install playwright && python -m playwright install") from e

# ---------- Config ----------
ARXIV_API = "http://export.arxiv.org/api/query"
CATEGORY = "cs.CL"                 # change as needed
MAX_RESULTS = 200                  # fetch this many entries via API
REQUESTS_DELAY = 0.5               # polite delay between requests (seconds)
USER_AGENT = "arXivAbstractScraper/1.0 (mailto:jademtl88@gmail.com)"  # change email
OUTPUT_FILE = "arxiv_clean.json"
TMP_DIR = Path("tmp_arxiv")
TESSERACT_CMD = r"C:\Users\FeiFei\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"  # change if necessary

# Max output bytes (1MB)
MAX_BYTES = 1 * 1024 * 1024

# Set Tesseract executable (if needed)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------- Helper functions ----------
def fetch_arxiv_feed(category: str, max_results: int = 200) -> List[dict]:
    """Use arXiv API to fetch latest papers in category. Returns feedparser entries."""
    q = f"search_query=cat:{category}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    url = f"{ARXIV_API}?{q}"
    headers = {"User-Agent": USER_AGENT}
    logging.info("Fetching arXiv feed: %s", url)
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    entries = feed.entries
    logging.info("Got %d entries", len(entries))
    return entries


def fetch_abs_html(abs_url: str) -> str:
    """Fetch /abs/ HTML content via requests."""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(abs_url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def extract_abstract_trafilatura(html: str) -> str:
    """Use trafilatura to extract main text, then try to isolate the abstract.
    trafilatura may return a long text — we attempt to find 'Abstract' section if possible."""
    extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not extracted:
        return ""
    # common pattern: "Abstract" followed by text; we attempt to locate it
    lower = extracted.lower()
    idx = lower.find("abstract")
    if idx != -1:
        # take substring after 'abstract' word
        candidate = extracted[idx:]
        # remove section headers after the abstract if any (e.g., "Introduction")
        for sep in ["introduction", "keywords", "1.", "1 Introduction"]:
            sep_idx = candidate.lower().find(sep)
            if sep_idx > 20:  # ensure abstract not very short
                candidate = candidate[:sep_idx]
                break
        # cleanup
        candidate = candidate.replace("Abstract", "", 1).strip()
        return candidate
    # fallback: return first ~500 chars of extracted text as a guess
    return extracted.strip()[:2000]


def screenshot_page_and_ocr(playwright, abs_url: str, out_path: Path) -> str:
    """Open page with Playwright, take full-page screenshot, run pytesseract on it, return text."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(user_agent=USER_AGENT)
    page = context.new_page()
    try:
        logging.debug("Navigating to %s", abs_url)
        page.goto(abs_url, wait_until="networkidle", timeout=30000)
        time.sleep(0.3)  # let JS render if needed
        # full page screenshot
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out_path), full_page=True)
        # OCR
        image = Image.open(str(out_path))
        text = pytesseract.image_to_string(image)
        return text.strip()
    finally:
        context.close()
        browser.close()


def enforce_max_size(records: List[Dict], max_bytes: int) -> List[Dict]:
    """Ensure JSON of records <= max_bytes by truncating abstracts proportionally if needed."""
    json_bytes = json.dumps(records, ensure_ascii=False).encode("utf-8")
    if len(json_bytes) <= max_bytes:
        return records
    logging.info("Output %d bytes exceeds %d — truncating abstracts", len(json_bytes), max_bytes)
    # compute total abstract length
    abstracts = [r.get("abstract", "") for r in records]
    total_len = sum(len(a) for a in abstracts) or 1
    # target bytes for JSON metadata (approx)
    # We'll iteratively reduce abstracts until size fits
    # Simple proportional truncation:
    scale = 0.5
    new_records = records
    for attempt in range(20):
        # proportionally truncate each abstract
        new_records = []
        for r in records:
            a = r.get("abstract", "")
            new_len = max(80, int(len(a) * scale))  # keep at least 80 chars
            r2 = r.copy()
            if len(a) > new_len:
                r2["abstract"] = a[:new_len].rstrip() + "…"
            new_records.append(r2)
        b = json.dumps(new_records, ensure_ascii=False).encode("utf-8")
        logging.info("Attempt %d: size %d bytes (scale=%.3f)", attempt + 1, len(b), scale)
        if len(b) <= max_bytes:
            return new_records
        scale *= 0.8  # reduce more
    # If still too big, keep only top N items fitting
    logging.info("Proportional truncation didn't fit. Falling back to keeping fewer records.")
    # Sort by abstract length ascending and pick as many as fit
    sorted_by_len = sorted(new_records, key=lambda x: len(x.get("abstract","")))
    out = []
    for item in sorted_by_len:
        candidate = out + [item]
        if len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) > max_bytes:
            break
        out.append(item)
    return out


# ---------- Main workflow ----------
def main(category: str = CATEGORY, max_results: int = MAX_RESULTS):
    TMP_DIR.mkdir(exist_ok=True)
    entries = fetch_arxiv_feed(category=category, max_results=max_results)

    results = []
    with sync_playwright() as p:
        for i, e in enumerate(entries):
            try:
                # Basic metadata from feed
                url = e.get("id", "").replace("http://", "https://")
                title = (e.get("title") or "").replace("\n", " ").strip()
                authors = [a.name for a in e.get("authors", [])] if e.get("authors") else []
                date = e.get("published", "")  # ISO-like string from feed

                logging.info("[%d/%d] %s", i + 1, len(entries), title[:80])

                # Fetch HTML
                try:
                    html = fetch_abs_html(url)
                    trafilatura_abstract = extract_abstract_trafilatura(html)
                except Exception as ex:
                    logging.warning("Failed to fetch/extract HTML: %s", ex)
                    trafilatura_abstract = ""

                # Screenshot + OCR
                screenshot_path = TMP_DIR / f"{i+1}_abs.png"
                try:
                    ocr_text = screenshot_page_and_ocr(p, url, screenshot_path)
                except Exception as ex:
                    logging.warning("Screenshot/OCR failed: %s", ex)
                    ocr_text = ""

                # Choose best abstract: prefer trafilatura if non-empty & reasonably long, else OCR
                chosen_abstract = ""
                if trafilatura_abstract and len(trafilatura_abstract) >= 50:
                    chosen_abstract = trafilatura_abstract
                elif ocr_text and len(ocr_text) >= 50:
                    chosen_abstract = ocr_text
                else:
                    # try to locate arXiv summary from feed (some feed entries include summary)
                    feed_summary = (e.get("summary") or "").strip()
                    if feed_summary:
                        chosen_abstract = feed_summary
                    else:
                        chosen_abstract = trafilatura_abstract or ocr_text or ""

                result = {
                    "url": url,
                    "title": title,
                    "abstract": chosen_abstract.strip(),
                    "authors": authors,
                    "date": date,
                }
                results.append(result)

                time.sleep(REQUESTS_DELAY)
            except Exception as e:
                logging.exception("Error processing entry: %s", e)
                # keep going

    # Enforce size limit (1MB)
    final_records = enforce_max_size(results, MAX_BYTES)
    logging.info("Final records count: %d", len(final_records))
    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
    logging.info("Saved results to %s (%.1f KB)", OUTPUT_FILE, os.path.getsize(OUTPUT_FILE) / 1024.0)


if __name__ == "__main__":
    main()
