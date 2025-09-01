from __future__ import annotations
import uuid, pathlib, logging
from typing import List, Dict, Any
from pypdf import PdfReader
import trafilatura
from .utils import Doc, normalize_text

# Silence noisy pypdf warnings from malformed PDFs
logging.getLogger("pypdf").setLevel(logging.ERROR)

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def read_any(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext in [".txt", ".md"]:
        return read_txt(path)
    elif ext in [".pdf"]:
        return read_pdf(path)
    else:
        return read_txt(path)

def fetch_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    return trafilatura.extract(downloaded) or ""

def split_to_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += step
    return chunks or [text]

def guess_coin(label: str) -> str:
    low = label.lower()
    if "bitcoin" in low or "btc" in low: return "bitcoin"
    if "ethereum" in low or "eth" in low: return "ethereum"
    return ""

def build_docs_from_paths(paths: List[str], source_label: str = "local") -> List[Doc]:
    docs: List[Doc] = []
    for p in paths or []:
        raw = read_any(p)
        if not raw: 
            continue
        coin = guess_coin(p)
        for i, chunk in enumerate(split_to_chunks(raw)):
            docs.append(Doc(
                id=f"{uuid.uuid4()}",
                text=normalize_text(chunk),
                metadata={"source": source_label, "path": p, "chunk": i, "coin": coin}
            ))
    return docs

def build_docs_from_urls(urls: List[str], source_label: str = "web") -> List[Doc]:
    docs: List[Doc] = []
    for u in urls or []:
        raw = fetch_url(u)
        if not raw: 
            continue
        coin = guess_coin(u)
        for i, chunk in enumerate(split_to_chunks(raw)):
            docs.append(Doc(
                id=f"{uuid.uuid4()}",
                text=normalize_text(chunk),
                metadata={"source": source_label, "url": u, "chunk": i, "coin": coin}
            ))
    return docs
