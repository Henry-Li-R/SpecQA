# pip install pymupdf

import re
import json
from pathlib import Path
import fitz  # PyMuPDF

REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / "docs_raw" / "acuvim-3-manual.pdf"
OUT_DIR = REPO_ROOT / "src" / "ingest" / "data" / "chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_ID = "acuvim_3_manual"
MIN_CHARS = 60
CHUNK_CHARS = 500
OVERLAP_CHARS = 75
DEBUG_DIAGRAM_HEURISTIC = False

HEADING_ALLOWLIST = {
  "Chapter 1: Introduction",
  "1.1 Acuvim 3 Overview",
  "1.1.1 Revenue Grade Energy Measurement",
  "1.1.2 Power Quality Analysis",
  "1.1.3 Synchrophasor",
  "1.2 Areas of Application",
  "1.3 Accuracy",

  "Chapter 2: Hardware Installation",
  "2.1 Appearance and Dimensions",
  "2.2 Installation Methods",
  "2.2.1 DIN Rail Installation",
  "2.2.2 Panel Installation",
  "2.3 Wiring",
  "2.3.1 Terminals",
  "2.3.2 Safety Earth Connection",
  "2.3.3 Power Requirement",
  "2.3.4 Voltage Input Wiring",
  "2.3.5 Current Input Wiring",
  "2.3.6 Common Wiring Methods",
  "2.4 Communications Interface",
  "2.4.1 Serial RS485 Communications",
  "2.4.2 USB Communications",
  "2.4.3 Ethernet Communications",
  "2.4.4 Wi-Fi Communications",
  "2.5 On-board Input/Output Ports",
  "2.5.1 Digital Input",
  "2.5.2 Digital Output",

  "Chapter 3: Extended Modules",
  "3.1 Input/Output Modules",
  "3.1.1 Appearance and Dimensions",
  "3.1.2 I/O Functionality",
  "3.1.3 Installation Method",
  "3.1.4 I/O Module Wiring",

  "Chapter 4: Site Map and Metering",
  "4.1 Site Map",
  "4.2 About",
  "4.2.1 Meter Information",
  "4.2.2 Installation Record",
  "4.2.3 Inspection Record",
  "4.2.4 Nameplate",
  "4.3 Metering",
  "4.3.1 Realtime Webpage",
  "4.3.2 Fundamental Webpage",
  "4.3.3 Energy and Demand Webpage",
  "4.3.4 Min/Max Webpage",
  "4.3.5 THD and Flicker Webpage",
  "4.3.6 Harmonics Webpage",
  "4.3.7 Sequence Webpage",
  "4.3.8 I/O Webpage",
  "4.3.9 I/O Settings",
  "4.3.10 TOU Energy Webpage",
  "4.3.11 Revenue and Energy TOU Setting",
  "4.4 Logs",
  "4.4.1 SOE Log",
  "4.4.2 Trend Log",
  "4.4.3 Trend Log Management",
  "4.4.4 Data Log",
  "4.4.5 Event Log",
  "4.5 General Settings",
  "4.5.1 General Configuration",
  "4.5.2 HMI",

  "Chapter 5: Acuvim 3 Display Screen",
  "5.1 Acuvim 3 Screen Overview",
  "5.2 Metering",
  "5.2.1 Realtime Screen",
  "5.2.2 Unbalance Screen",
  "5.2.3 THD Screen",
  "5.2.4 Harmonics Screen",
  "5.2.5 Max/Min Screen",
  "5.3 Energy/Demand",
  "5.3.1 Import/Export Screen",
  "5.3.2 Quadrant Screen",
  "5.3.3 TOU Energy Screen",
  "5.3.4 Demand Screen",
  "5.4 Visualization",
  "5.4.1 Realtime Diagrams",
  "5.4.2 Harmonic Histogram",
  "5.5 Trend",
  "5.5.1 Realtime Trend Log",
  "5.5.2 Energy Trend Log",
  "5.6 Waveform",
  "5.7 Power Quality",
  "5.7.1 PQ Event",
  "5.7.2 ITIC",
  "5.7.3 Alarm Status",
  "5.7.4 Alarm Log",
  "5.8 Input Output",
  "5.8.1 I/O Configuration",
  "5.8.2 SOE Log",
  "5.9 Dashboard",
  "5.10 User Center",
  "5.10.1 Installation",
  "5.10.2 Communication",
  "5.10.3 About",
  "5.10.4 Operation",
  "5.10.5 Event Log",
  "5.11 User Management",

  "Chapter 6: Power Quality Measurements",
  "6.1 Power Quality Event",
  "6.1.1 Voltage Sag Detection",
  "6.1.2 Voltage Swell Detection",
  "6.1.3 Voltage Interruption Detection",
  "6.1.4 Unbalanced Voltage Detection",
  "6.1.5 Transient Voltage Detection",
  "6.1.6 Current Sag Detection",
  "6.1.7 Current Swell Detection",
  "6.1.8 Unbalanced Current Detection",
  "6.1.9 Power Quality Event General Configuration",
  "6.2 Waveform and Fastlog",
  "6.2.1 Waveform and Fastlog Settings",
  "6.2.2 Waveform and Fastlog Data Post Settings",
  "6.2.3 Waveform and Fastlog HTTP/HTTPS Settings",
  "6.2.4 Waveform and Fastlog FTP/SFTP Settings",
  "6.3 Email Notification",
  "6.4 Power Quality Event Analysis",
  "6.4.1 Power Quality Event",
  "6.4.2 Waveform Capture",
  "6.4.3 Fast Log",
  "6.4.4 Transient Voltage Log",
  "6.4.5 Mains Signaling Voltage Log",
  "6.4.6 Mains Signaling Voltage Record",
  "6.5 Alarm",
  "6.5.1 Alarm Configuration",
  "6.5.2 Alarm Status",
  "6.5.3 Alarm Log",
  "6.6 Power Quality Report",
  "6.6.1 EN50160 Compliant Report",
  "6.6.2 IEEE519 Compliant Report",
  "6.6.3 ITIC/CBEMA Curve Report",
  "6.6.4 SEMI Curve Report",
  "6.7 Power Quality Logging",
  "6.7.1 IEC 61000-4-30 Compliant Aggregation Logging",
  "6.7.2 EN50160 Report Logging",
  "6.7.3 IEEE519 Report Logging",
  "6.8 DI Trigger",

  "Chapter 7: Communications",
  "7.1 RS485 and USB Settings",
  "7.2 Network",
  "7.2.1 RSTP",
  "7.2.2 IPv4 Ethernet",
  "7.2.3 IPv4 Wi-Fi",
  "7.2.4 IPv6 Ethernet",
  "7.2.5 HTTP Proxy",
  "7.3 Access Control",
  "7.4 Remote Access",
  "7.5 Webpage Interface",
  "7.5.1 HTTP/HTTPS",
  "7.5.2 Certificate Management",
  "7.6 Time/Date",
  "7.6.1 NTP & SNTP",
  "7.6.2 PTP",
  "7.6.3 IRIG-B",
  "7.6.4 Manual",
  "7.7 SMTP Email",
  "7.8 Modbus",
  "7.9 BACnet",
  "7.9.1 BACnet/IP",
  "7.9.2 BACnet MS/TP",
  "7.10 SNMP",
  "7.10.1 SNMP V2C",
  "7.10.2 SNMP V3",
  "7.10.3 Email Traps",
  "7.11 DNP",
  "7.12 IEC 61850",
  "7.13 EtherNet/IP",
  "7.14 PMU",
  "7.14.1 Message Settings",
  "7.14.2 Transmission Settings",

  "Chapter 8: Data Log and Post",
  "8.1 Data Log",
  "8.1.1 Log File Setting",
  "8.1.2 Log Parameter Options",
  "8.1.3 SFTP Backup",
  "8.2 Data Post",
  "8.2.1 HTTP/HTTPS Settings",
  "8.2.2 FTP Settings",
  "8.2.3 SFTP Settings",
  "8.2.4 Email Settings",
  "8.3 AcuCloud",

  "Chapter 9: User Management",
  "9.1 User Configuration",
  "9.2 Role Configuration",
  "9.2.1 Reading Permissions",
  "9.2.2 Configuration Permission",
  "9.2.3 Maintenance Permission",
  "9.2.4 User Configuration Permission",
  "9.3 Password Policy",
  "9.4 Password Configuration",
  "9.5 API Token Management",

  "Chapter 10: Maintenance and Management",
  "10.1 Operation",
  "10.1.1 Debug Diagnostic",
  "10.2 Configuration Management",
  "10.3 Network Diagnostic",
  "10.3.1 Network Status",
  "10.3.2 Host Lookup",
  "10.3.3 Connection Test",
  "10.4 Firmware"
}

CHAPTER_ALLOWLIST = {h for h in HEADING_ALLOWLIST if h.startswith("Chapter ")}

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

    # Text pages: split by headings
    current_section = "UNKNOWN"
    current_chapter = "UNKNOWN"

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        raw = doc[page_idx].get_text("text") or ""
        lines = [normalize_text_line(l) for l in raw.splitlines()]
        lines = [l for l in lines if l]  # drop blanks

        # Skip diagram-heavy pages.
        diagram = page_is_diagram_heavy(doc[page_idx], raw)

        if diagram:
            continue

        buf = []

        def flush():
            nonlocal buf, current_section, current_chapter
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
                    "chunk_id": f"{DOC_ID}:p{page_num}:{slugify_section(current_chapter)}:{slugify_section(current_section)}:{len(chunks)}",
                    "doc_id": DOC_ID,
                    "page": page_num,
                    "chapter": current_chapter,
                    "section": current_section,
                    "text": part
                })
            buf = []

        for line in lines:
            if is_heading(line):
                # start a new chunk at heading boundaries
                flush()
                current_section = line
                if line in CHAPTER_ALLOWLIST:
                    current_chapter = line
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
