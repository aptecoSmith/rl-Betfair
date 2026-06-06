"""Dispatcher + CLI for extractors.

    python extractors/extract.py <file> [--granularity structural|paragraph] [--json]

Routes by extension to the right extractor. Markdown/text pass through as a single segment.
Missing optional libs or legacy binary formats raise ExtractorUnavailable so callers can fall
back to the agent's own file skills.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import ExtractResult, ExtractorUnavailable, Segment  # noqa: E402

LEGACY = {".doc", ".ppt", ".xls"}


def extract(path, granularity="structural") -> ExtractResult:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        import pdf_extractor
        return pdf_extractor.extract(path, granularity)
    if suffix == ".docx":
        import docx_extractor
        return docx_extractor.extract(path, granularity)
    if suffix == ".pptx":
        import pptx_extractor
        return pptx_extractor.extract(path, granularity)
    if suffix == ".xlsx":
        import xlsx_extractor
        return xlsx_extractor.extract(path, granularity)
    if suffix in (".md", ".markdown", ".txt"):
        text = p.read_text(encoding="utf-8", errors="replace")
        return ExtractResult(p.stem, suffix.lstrip("."),
                             [Segment("s1", p.stem, text, {"source": p.name})])
    if suffix in LEGACY:
        # Opt-in: convert via LibreOffice to OOXML, then re-dispatch. Raises ExtractorUnavailable
        # (caller falls back to the agent's file skill) if LibreOffice isn't installed.
        import shutil as _sh
        import tempfile as _tf

        import legacy
        tmp = _tf.mkdtemp(prefix="llmwiki-legacy-")
        try:
            converted = legacy.convert(p, out_dir=tmp)
            return extract(converted, granularity)
        finally:
            _sh.rmtree(tmp, ignore_errors=True)
    raise ExtractorUnavailable(f"no extractor for {suffix}")


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="extract")
    ap.add_argument("file")
    ap.add_argument("--granularity", choices=["structural", "paragraph"], default="structural")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--chunks-dir", help="also write each segment as a durable .md chunk in this dir "
                                         "(resumable/inspectable for very large docs)")
    args = ap.parse_args(argv)
    try:
        result = extract(args.file, args.granularity)
    except ExtractorUnavailable as e:
        print(f"unavailable: {e}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps({
            "title": result.title, "content_type": result.content_type,
            "page_count": result.page_count(),
            "segments": [{"id": s.seg_id, "title": s.title,
                          "chars": len(s.text), "provenance": s.provenance}
                         for s in result.segments]}, ensure_ascii=False, indent=2))
    else:
        print(f"{result.title} [{result.content_type}] - {len(result.segments)} segment(s), "
              f"~{result.page_count()} page(s)")
        for s in result.segments:
            preview = s.text[:60].replace("\n", " ")
            print(f"  {s.seg_id:8} {s.title[:40]:40} {preview}")
    if args.chunks_dir:
        from base import write_chunks
        paths = write_chunks(result, args.chunks_dir)
        print(f"wrote {len(paths)} chunk(s) to {args.chunks_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
