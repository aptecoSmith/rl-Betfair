"""DOCX extractor (python-docx). Segments by Heading styles; no Office automation."""
from __future__ import annotations

from pathlib import Path

from base import ExtractResult, ExtractorUnavailable, Segment, to_paragraph_segments


def _load():
    try:
        import docx
        return docx
    except ImportError as e:
        raise ExtractorUnavailable(
            "python-docx not installed (pip install -r extractors/requirements.txt)") from e


def extract(path, granularity="structural"):
    docx = _load()
    d = docx.Document(str(path))
    source = Path(path).name
    segs = []
    cur_title = "Body"
    cur_text = []
    idx = 0

    def flush():
        nonlocal idx, cur_text
        if cur_text:
            idx += 1
            segs.append(Segment(f"s{idx}", cur_title, "\n\n".join(cur_text).strip(),
                                {"source": source, "section": cur_title}))
            cur_text = []

    for para in d.paragraphs:
        style = (para.style.name if para.style else "") or ""
        txt = para.text.strip()
        if style.startswith("Heading") and txt:
            flush()
            cur_title = txt
        elif txt:
            cur_text.append(txt)
    flush()
    if granularity == "paragraph":
        segs = to_paragraph_segments(segs)
    return ExtractResult(Path(path).stem, "docx", segs)
