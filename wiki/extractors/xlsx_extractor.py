"""XLSX extractor (openpyxl). One segment per worksheet."""
from __future__ import annotations

from pathlib import Path

from base import ExtractResult, ExtractorUnavailable, Segment


def _load():
    try:
        import openpyxl
        return openpyxl
    except ImportError as e:
        raise ExtractorUnavailable(
            "openpyxl not installed (pip install -r extractors/requirements.txt)") from e


def extract(path, granularity="structural"):
    openpyxl = _load()
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    source = Path(path).name
    segs = []
    for i, ws in enumerate(wb.worksheets, 1):
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                rows.append(" | ".join(cells))
        segs.append(Segment(f"sheet{i}", ws.title, "\n".join(rows),
                            {"source": source, "sheet": ws.title}))
    wb.close()
    return ExtractResult(Path(path).stem, "xlsx", segs)
