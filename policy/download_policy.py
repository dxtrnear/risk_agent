"""Download the AXA "Mon Auto" conditions générales PDF."""

import urllib.request
from pathlib import Path

import config

URL = "https://media.axa.fr/content/dam/axa-fr/image/particuliers/auto/documents-informations/auto/pdf-mon-auto-cg.pdf"


def download() -> Path:
    """Fetch the policy PDF and save it to policy/raw/. Returns the saved path."""
    config.POLICY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = config.POLICY_RAW_DIR / "pdf-mon-auto-cg.pdf"

    with urllib.request.urlopen(URL, timeout=30) as response:
        pdf_path.write_bytes(response.read())
    return pdf_path


if __name__ == "__main__":
    path = download()
    print(f"Saved {path}")
