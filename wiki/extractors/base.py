"""Shared types and helpers for extractors.

An extractor turns a source file into an ExtractResult: a title + an ordered list of
Segments, each carrying provenance (page/section/slide/sheet). The ingest engine then
processes segments independently (the anti-shortcut flow). If an optional library is
missing, an extractor raises ExtractorUnavailable so the caller can fall back to the
agent's own file skills.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


class ExtractorUnavailable(Exception):
    """Raised when the dependency for a format isn't installed, or the format is unsupported."""


@dataclass
class Segment:
    seg_id: str
    title: str
    text: str
    provenance: dict = field(default_factory=dict)


@dataclass
class ExtractResult:
    title: str
    content_type: str
    segments: list

    def page_count(self) -> int:
        pages = set()
        for s in self.segments:
            p = s.provenance
            if "page" in p:
                pages.add(p["page"])
            elif "pages" in p:
                for n in range(p["pages"][0], p["pages"][1] + 1):
                    pages.add(n)
        return len(pages) or len(self.segments)


def split_paragraphs(text: str):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def to_paragraph_segments(segments):
    """Explode each segment into one-segment-per-paragraph (paragraph granularity)."""
    out = []
    for seg in segments:
        paras = split_paragraphs(seg.text)
        if len(paras) <= 1:
            out.append(seg)
            continue
        for i, para in enumerate(paras, 1):
            prov = dict(seg.provenance)
            prov["paragraph"] = i
            out.append(Segment(f"{seg.seg_id}.p{i}", seg.title, para, prov))
    return out


def _slug(text, maxlen=50):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:maxlen] or "segment"


def write_chunks(result, out_dir):
    """Persist each segment of an ExtractResult as a durable markdown chunk file, plus an index.

    For very large/interruptible docs this gives resumable, inspectable per-chapter artifacts on disk
    (the agent turns one chunk into notes at a time and can see what's left), while in-memory
    segmentation stays the default. Writes <out_dir>/NNN-<slug>.md per segment plus index.md, and
    returns the chunk paths. Idempotent: stable names, identical content on re-run.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segs = result.segments
    width = max(2, len(str(len(segs))))
    paths = []
    index = [f"# Chunks: {result.title}", "",
             f"- type: {result.content_type}",
             f"- segments: {len(segs)}",
             f"- pages (approx): {result.page_count()}", ""]
    for i, seg in enumerate(segs, 1):
        name = f"{str(i).zfill(width)}-{_slug(seg.title)}.md"
        chunk = (f"---\nseg_id: {seg.seg_id}\ntitle: {seg.title}\n"
                 f"provenance: {json.dumps(seg.provenance, ensure_ascii=False)}\n---\n\n"
                 f"# {seg.title}\n\n{seg.text}\n")
        (out / name).write_text(chunk, encoding="utf-8")
        paths.append(out / name)
        prov = ", ".join(f"{k}={v}" for k, v in seg.provenance.items())
        index.append(f"- [{seg.seg_id}] [{seg.title}]({name}) — {prov} — {len(seg.text)} chars")
    (out / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return paths
