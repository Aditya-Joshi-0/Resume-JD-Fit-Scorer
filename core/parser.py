import pdfplumber
import re
import io
import numpy as np
from typing import Optional

# PaddleOCR is imported lazily (only when needed) to keep cold-start fast
_ocr_engine = None

SECTION_HEADERS = [
    "experience", "work experience", "employment", "professional experience",
    "education", "academic background",
    "skills", "technical skills", "core competencies",
    "projects", "certifications", "summary", "objective",
    "publications", "achievements", "awards",
]

# If pdfplumber extracts fewer than this many characters, treat as scanned
SCANNED_CHAR_THRESHOLD = 100


def _get_ocr_engine():
    """Lazy-load PaddleOCR so it doesn't slow down the initial app load."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        # lang='en', use_angle_cls detects rotated text (common in scanned docs)
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_engine


def _ocr_pdf(file) -> str:
    """
    OCR fallback for scanned PDFs.
    Converts each page to an image via pdfplumber, then runs PaddleOCR.
    """
    import pdfplumber
    from PIL import Image

    ocr = _get_ocr_engine()
    full_text = []

    # Reset file pointer in case it was read before
    file.seek(0)

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # Render page to PIL image at 200 DPI (good balance of speed vs accuracy)
            img = page.to_image(resolution=200).original
            img_array = np.array(img)

            result = ocr.ocr(img_array, cls=True)

            if result and result[0]:
                page_lines = [
                    word_info[1][0]          # [1][0] = the recognised text string
                    for line in result
                    for word_info in line
                    if word_info[1][1] > 0.5  # confidence threshold
                ]
                full_text.append(" ".join(page_lines))

    return "\n".join(full_text).strip()


def extract_text_from_pdf(file) -> tuple[str, str]:
    """
    Extract text from a PDF, with automatic OCR fallback for scanned documents.

    Returns:
        (text, method) where method is 'digital' or 'ocr'
    """
    # --- Try digital extraction first ---
    text = ""
    file.seek(0)
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    text = text.strip()

    if len(text) >= SCANNED_CHAR_THRESHOLD:
        return text, "digital"

    # --- Fall back to OCR ---
    text = _ocr_pdf(file)
    return text, "ocr"


def detect_section(line: str) -> Optional[str]:
    """Return the section name if the line looks like a section header."""
    cleaned = line.strip().lower().rstrip(":")
    if cleaned in SECTION_HEADERS:
        return cleaned
    return None


def parse_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections.
    Falls back to a single 'full_text' key if no headers are detected.
    """
    sections = {}
    current_section = "summary"
    buffer = []

    for line in text.splitlines():
        section = detect_section(line)
        if section:
            if buffer:
                sections[current_section] = "\n".join(buffer).strip()
            current_section = section
            buffer = []
        else:
            if line.strip():
                buffer.append(line)

    if buffer:
        sections[current_section] = "\n".join(buffer).strip()

    if not sections:
        sections["full_text"] = text

    return sections


def chunk_resume(sections: dict[str, str], chunk_size: int = 300) -> list[dict]:
    """
    Break each section into overlapping text chunks for embedding.
    Returns list of {text, section, chunk_id}.
    """
    chunks = []
    chunk_id = 0

    for section, content in sections.items():
        words = content.split()
        stride = chunk_size // 2

        if len(words) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "section": section,
                "text": content,
            })
            chunk_id += 1
        else:
            for i in range(0, len(words), stride):
                chunk_words = words[i: i + chunk_size]
                chunks.append({
                    "chunk_id": chunk_id,
                    "section": section,
                    "text": " ".join(chunk_words),
                })
                chunk_id += 1
                if i + chunk_size >= len(words):
                    break

    return chunks
