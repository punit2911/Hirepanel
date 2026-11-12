"""Resume file -> plain text, with strict validation (never trust an upload)."""
from __future__ import annotations

import io
import re

ALLOWED = {".pdf", ".docx", ".txt", ".md"}
MAX_CHARS = 30_000


class ExtractError(ValueError):
    pass


def _ext(filename: str) -> str:
    m = re.search(r"(\.[A-Za-z0-9]+)$", filename or "")
    return m.group(1).lower() if m else ""


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractError("this PDF is password protected")
        return "\n".join((p.extract_text() or "") for p in reader.pages[:15])
    except ExtractError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as e:
        raise ExtractError(f"could not read this PDF ({type(e).__name__})") from e


def _docx(data: bytes) -> str:
    import docx
    from docx.opc.exceptions import PackageNotFoundError

    try:
        d = docx.Document(io.BytesIO(data))
    except (PackageNotFoundError, ValueError, KeyError, OSError) as e:
        raise ExtractError("could not read this Word file") from e
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def _plain(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    raise ExtractError("could not decode this text file")


def extract_text(filename: str, data: bytes) -> str:
    ext = _ext(filename)
    if ext not in ALLOWED:
        raise ExtractError(f"unsupported file type '{ext or 'unknown'}'. Use PDF, DOCX or TXT")
    if not data:
        raise ExtractError("the file is empty")
    if ext == ".pdf":
        if not data.startswith(b"%PDF"):
            raise ExtractError("this file is not a real PDF")
        text = _pdf(data)
    elif ext == ".docx":
        if not data.startswith(b"PK"):
            raise ExtractError("this file is not a real Word document")
        text = _docx(data)
    else:
        text = _plain(data)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(re.sub(r"\s", "", text)) < 80:
        raise ExtractError("no readable text found (scanned image PDFs are not supported)")
    return text[:MAX_CHARS]
