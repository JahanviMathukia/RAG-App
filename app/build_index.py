"""
This script:
- Reads all .pdf in data/docs
- Splits them into overlapping chunks
- Embeds chunks with SentenceTransformers (free & local)
- Saves embeddings/index.pkl for fast retrieval (caching)`
"""
import os
import glob
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_PATH = os.path.join(EMBED_DIR, "index.pkl")

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 200   # overlap between chunks
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class DocChunk:
    id: int
    source_path: str
    chunk_index: int
    text: str

def read_txt_or_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)

def load_documents() -> List[Tuple[str, str]]:
    """
    Returns list of (path, text) for all supported documents in data/docs.
    """
    paths_txt = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    paths_md = glob.glob(os.path.join(DOCS_DIR, "*.md"))
    paths_pdf = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))

    all_paths = paths_txt + paths_md + paths_pdf

    if not all_paths:
        raise RuntimeError(
            f"No .txt/.md/.pdf files found in {DOCS_DIR}. "
            f"Add at least one document there."
        )

    documents: List[Tuple[str, str]] = []

    for p in all_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in [".txt", ".md"]:
            text = read_txt_or_md(p)
        elif ext == ".pdf":
            print(f"Reading PDF: {os.path.basename(p)}")
            text = read_pdf(p)
        else:
            continue

        text = text.replace("\r", " ").strip()
        documents.append((p, text))

    return documents

def chunk_text(text: str, source_path: str, start_chunk_id: int) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    start = 0
    chunk_index = 0
    current_id = start_chunk_id

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(
                DocChunk(
                    id=current_id,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            current_id += 1
            chunk_index += 1

        # Move start forward with overlap
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0

        # If not progressing, break (safety)
        if start >= len(text):
            break

    return chunks

def build_index() -> None:
    os.makedirs(EMBED_DIR, exist_ok=True)

    print(f"Loading docs from {DOCS_DIR} ...")
    docs = load_documents()

    all_chunks: List[DocChunk] = []
    next_id = 0

    for path, text in docs:
        print(f"Chunking file: {os.path.basename(path)}")
        file_chunks = chunk_text(text, path, start_chunk_id=next_id)
        all_chunks.extend(file_chunks)
        next_id += len(file_chunks)

    if not all_chunks:
        raise RuntimeError("No text content available to create chunks.")

    print(f"Created {len(all_chunks)} chunks in total.")
    print(f"Embedding with {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c.text for c in all_chunks]
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True
    )

    index: Dict[str, Any] = {
        "embed_model_name": EMBED_MODEL_NAME,
        "chunks": [asdict(c) for c in all_chunks],
        "embeddings": embeddings,
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"Saved index to {INDEX_PATH}")

if __name__ == "__main__":
    build_index()