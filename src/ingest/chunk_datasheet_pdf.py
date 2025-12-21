# pip install pymupdf

import re
import json
from pathlib import Path
import fitz  # PyMuPDF

REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / "docs_raw" / "acuvim-3-datasheet.pdf"
OUT_DIR = REPO_ROOT / "src" / "ingest" / "data" / "chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_ID = "acuvim_3_datasheet"
MIN_CHARS = 60
CHUNK_CHARS = 500
OVERLAP_CHARS = 75
DEBUG_DIAGRAM_HEURISTIC = False

HEADING_ALLOWLIST = {
    "DESCRIPTION","FEATURES","KEY FEATURES","APPLICATIONS","SPECIFICATIONS",
    "FUNCTION LIST","DIMENSIONS","WIRING DIAGRAMS","ORDERING INFORMATION"
}

def is_heading(line: str) -> bool:
    s = line.strip()
    return s in HEADING_ALLOWLIST

def normalize_text_line(line: str) -> str:
    # collapse weird spacing while preserving table-ish alignment somewhat
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line

def page_is_diagram_heavy(page, text: str) -> bool:
    # Heuristic: low text area/word count + lots of images/drawings.
    if not text.strip():
        return True

    page_area = float(page.rect.width * page.rect.height)
    text_area = 0.0
    text_blocks = 0
    word_count = 0
    try:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox")
            if bbox:
                x0, y0, x1, y1 = bbox
                text_area += max(0.0, (x1 - x0) * (y1 - y0))
            text_blocks += 1
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    word_count += len(span.get("text", "").split())
    except Exception:
        word_count = len(text.split())

    try:
        drawing_count = len(page.get_drawings())
    except Exception:
        drawing_count = 0

    if page_area <= 0:
        return False

    text_ratio = text_area / page_area
    low_text = text_ratio < 0.12 and word_count < 200
    heavy_graphics = drawing_count >= 400

    is_diagram = low_text and heavy_graphics
    if DEBUG_DIAGRAM_HEURISTIC:
        print(f"Page {page.number + 1}: text_ratio={text_ratio:.3f}, "
              f"word_count={word_count}, text_blocks={text_blocks}, "
              f"drawing_count={drawing_count} => diagram={is_diagram}")
    return is_diagram

def extract_chunks(pdf_path: Path):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        raw = doc[page_idx].get_text("text") or ""
        lines = [normalize_text_line(l) for l in raw.splitlines()]
        lines = [l for l in lines if l]  # drop blanks

        # Skip diagram-heavy pages.
        diagram = page_is_diagram_heavy(doc[page_idx], raw)

        if diagram:
            continue

        # Text pages: split by headings
        current_section = "UNKNOWN"
        buf = []

        def flush():
            nonlocal buf, current_section
            if not buf:
                buf = []
                return
            text = "\n".join(buf).strip()
            if len(text) < MIN_CHARS:
                buf = []
                return
            parts = split_text_by_chars(
                text,
                chunk_chars=CHUNK_CHARS,
                overlap_chars=OVERLAP_CHARS,
            )
            for part in parts:
                chunks.append({
                    "chunk_id": f"{DOC_ID}:p{page_num}:{slugify_section(current_section)}:{len(chunks)}",
                    "doc_id": DOC_ID,
                    "page": page_num,
                    "section": current_section,
                    "text": part
                })
            buf = []

        for line in lines:
            if is_heading(line):
                # start a new chunk at heading boundaries
                flush()
                current_section = line
                # include heading as first line (helps retrieval)
                buf.append(line)
            else:
                buf.append(line)

        flush()

    return chunks

def slugify_section(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "section"

def split_text_by_chars(text: str, chunk_chars: int, overlap_chars: int):
    if chunk_chars <= 0:
        return [text]
    if overlap_chars < 0:
        overlap_chars = 0

    chunks = []
    start = 0
    text_len = len(text)
    step = max(chunk_chars - overlap_chars, 1)

    while start < text_len:
        end = min(start + chunk_chars, text_len)
        part = text[start:end]
        if part.strip():
            chunks.append(part.strip())
        start += step

    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHARS:
        chunks[-2] = f"{chunks[-2]}{chunks[-1]}"
        chunks.pop()

    return chunks

def write_jsonl(chunks, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    chunks = extract_chunks(PDF_PATH)
    write_jsonl(chunks, OUT_DIR / f"{DOC_ID}_chunks.jsonl")
    print(f"Wrote {len(chunks)} chunks to {OUT_DIR / f'{DOC_ID}_chunks.jsonl'}")
