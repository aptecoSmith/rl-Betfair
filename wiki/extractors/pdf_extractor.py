"""PDF extractor with structure-aware segmentation (pypdf).

Segmentation strategy:
  1. PDF outline (bookmarks) -> one segment per outline entry (chapter/section).
  2. else -> one segment per page (page-window fallback).
Paragraph granularity further explodes each segment by paragraph.
"""
from __future__ import annotations

from pathlib import Path

from base import ExtractResult, ExtractorUnavailable, Segment, to_paragraph_segments


def _load():
    try:
        import pypdf
        return pypdf
    except ImportError as e:
        raise ExtractorUnavailable(
            "pypdf not installed (pip install -r extractors/requirements.txt)") from e


def _walk_outline(reader, item, out):
    if isinstance(item, list):
        for sub in item:
            _walk_outline(reader, sub, out)
        return
    try:
        out.append((item.title, reader.get_destination_page_number(item)))
    except Exception:
        pass


def read_pages(path):
    pypdf = _load()
    reader = pypdf.PdfReader(str(path))
    pages = [(pg.extract_text() or "") for pg in reader.pages]
    outline = []
    try:
        _walk_outline(reader, reader.outline, outline)
    except Exception:
        outline = []
    return pages, outline


def segment_pages(pages, outline, source_name, granularity="structural"):
    """Pure segmentation logic - testable without a real PDF."""
    segs = []
    valid = [(t, p) for (t, p) in outline if isinstance(p, int) and 0 <= p < len(pages)]
    if valid:
        ol = sorted(valid, key=lambda x: x[1])
        for i, (title, start) in enumerate(ol):
            end = ol[i + 1][1] if i + 1 < len(ol) else len(pages)
            end = max(end, start + 1)
            text = "\n\n".join(pages[start:end]).strip()
            segs.append(Segment(f"s{i + 1}", title.strip() or f"Section {i+1}", text,
                                {"source": source_name, "pages": [start + 1, end]}))
    else:
        for i, text in enumerate(pages):
            segs.append(Segment(f"p{i + 1}", f"Page {i + 1}", text.strip(),
                                {"source": source_name, "page": i + 1}))
    if granularity == "paragraph":
        segs = to_paragraph_segments(segs)
    return segs


def extract(path, granularity="structural"):
    pages, outline = read_pages(path)
    segs = segment_pages(pages, outline, Path(path).name, granularity)
    return ExtractResult(Path(path).stem, "pdf", segs)
