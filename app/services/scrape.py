"""
On-demand scraping: run a portal's scraper to fetch a tender that isn't ingested
yet. The scrapers live one level up (c:\\anshul\\MVP\\<source>\\<script>.py) and run
on **system Python 3.14** (where their deps — requests/fitz/pytesseract/playwright
— are installed), not this app's venv. We shell out, then read the exact output
JSON path the scraper prints in its summary ("json     : <path>").
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

from app.config import settings

log = logging.getLogger("tender_rag.scrape")


class ScrapeError(RuntimeError):
    pass


# source -> scraper script (relative to scrapers_root)
SCRIPTS = {
    "etenders": "etenders/tender_ocr.py",
    "transnet": "transnet/transnet.py",
    "sadc": "sadc/sadc.py",
    "zppa": "zppa/zppa.py",
    "ppadb": "ppadb/ppadb.py",
    "randwater": "randwater/randwater.py",
    "capetown": "capetown/capetown.py",
    "cpbn": "cpbn/cpbn.py",
    # nra is intentionally absent: it needs a headed browser + reCAPTCHA and
    # cannot be scraped unattended.
}

# extra CLI args (saved creds/details) some portals need.
_EXTRA = {
    "cpbn": ["--details-file", "cpbn/details.json"],
    "ppadb": ["--credentials-file", "ppadb/credentials.json"],
    "capetown": ["--credentials-file", "capetown/credentials.json"],
}

# dirs the scrapers need on PATH (OCR + Office conversion).
_PATH_DIRS = [r"C:\Program Files\Tesseract-OCR", r"C:\Program Files\LibreOffice\program"]


def can_scrape(source: str) -> bool:
    return source in SCRIPTS


def _env() -> dict:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(_PATH_DIRS) + os.pathsep + env.get("PATH", "")
    return env


def scrape_tender(source: str, tender_id: str) -> Path:
    """Run the scraper for one tender and return the JSON file it wrote."""
    if source == "nra":
        raise ScrapeError(
            "nra can't be auto-scraped (reCAPTCHA + headed browser). Fetch it "
            "manually: python nra/nra.py <slug> --browser --headed --details-file "
            "nra/details.json --accept-policy")
    script = SCRIPTS.get(source)
    if not script:
        raise ScrapeError(f"auto-scrape not supported for source {source!r}; "
                          f"known: {', '.join(SCRIPTS)}")

    cmd = [settings.scraper_python, script, tender_id, *_EXTRA.get(source, [])]
    log.info("scraping %s/%s: %s", source, tender_id, " ".join(cmd))
    try:
        p = subprocess.run(cmd, cwd=str(settings.scrapers_root), env=_env(),
                           capture_output=True, text=True,
                           timeout=settings.scrape_timeout)
    except subprocess.TimeoutExpired as e:
        raise ScrapeError(f"scraping {source}/{tender_id} timed out after "
                          f"{settings.scrape_timeout}s") from e
    except OSError as e:
        raise ScrapeError(f"could not launch the {source} scraper "
                          f"({settings.scraper_python}): {e}") from e

    # The scraper prints "json     : <abs path>" on success.
    m = re.search(r"json\s*:\s*(.+?\.json)\s*$", p.stdout or "", re.MULTILINE)
    if not m:
        tail = (p.stderr or p.stdout or "").strip()[-400:]
        raise ScrapeError(f"the {source} scraper did not produce a tender "
                          f"(exit {p.returncode}). Output tail: {tail}")
    path = Path(m.group(1).strip())
    if not path.exists():
        raise ScrapeError(f"scraper reported {path} but the file is missing")
    log.info("scraped %s/%s -> %s", source, tender_id, path.name)
    return path
