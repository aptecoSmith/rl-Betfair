"""Read-only, privacy-aware access layer for the wiki - the logic behind the MCP server.

No `mcp` dependency here, so it stays unit-testable without the optional SDK. It reuses wiki_tool for
note discovery, frontmatter, BM25 search and the source registry.

PRIVACY (the key v2 adaptation): only the **shared** cloud is exposed to external consumers. Notes in
`personal/` and anything tagged `home`/`personal` are withheld unless `include_personal=True` - a
deliberate, local-only opt-in (the server passes it only when started with --include-personal).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wiki_tool as wt  # noqa: E402

PUBLIC_CLOUD = "shared"
PRIVATE_TAGS = {"home", "personal"}
SNIPPET_RADIUS = 120


def _root() -> Path:
    return Path(wt.ROOT)


def _rel(path: Path) -> str:
    return str(path.relative_to(_root())).replace("\\", "/")


def is_public(note, include_personal: bool = False) -> bool:
    """Exposable to external consumers iff in the shared cloud and not tagged home/personal."""
    if include_personal:
        return True
    if note.cloud != PUBLIC_CLOUD:
        return False
    return not ({str(t).lower() for t in note.as_list("tags")} & PRIVATE_TAGS)


def _public_notes(include_personal: bool = False):
    return [n for n in wt.find_notes() if is_public(n, include_personal)]


def _safe_path(page_path: str) -> Path:
    """Resolve a wiki path safely: reject escapes, require a known cloud + .md, must exist."""
    raw = page_path.strip().replace("\\", "/").lstrip("/")
    if not raw:
        raise ValueError("Path cannot be empty")
    root = _root().resolve()
    candidate = (root / raw).resolve()
    if root not in candidate.parents:
        raise ValueError("Path escapes the wiki root")
    if candidate.suffix.lower() != ".md":
        candidate = candidate.with_suffix(".md")
    parts = candidate.relative_to(root).parts
    if not parts or parts[0] not in wt.CLOUDS:
        raise ValueError(f"Path must be under a cloud {wt.CLOUDS}")
    if not candidate.exists():
        raise FileNotFoundError(f"Wiki page not found: {_rel(candidate)}")
    return candidate


def _note_for(page_path: str, include_personal: bool = False):
    note = wt.Note(_safe_path(page_path))
    if not is_public(note, include_personal):
        raise PermissionError(f"{_rel(note.path)} is private (personal cloud or home/personal tag)")
    return note


def _snippet(text: str, query: str) -> str:
    low = text.lower()
    pos = next((i for qt in wt._tokens(query) for i in [low.find(qt)] if i != -1), -1)
    if pos == -1:
        return re.sub(r"\s+", " ", text).strip()[: SNIPPET_RADIUS * 2]
    start, end = max(0, pos - SNIPPET_RADIUS), min(len(text), pos + SNIPPET_RADIUS)
    s = re.sub(r"\s+", " ", text[start:end]).strip()
    return ("... " + s if start > 0 else s) + (" ..." if end < len(text) else "")


# --------------------------------------------------------------------------- #
# the read-only API (each backs one MCP tool)
# --------------------------------------------------------------------------- #
def get_index(cloud: str = PUBLIC_CLOUD, include_personal: bool = False) -> str:
    if cloud != PUBLIC_CLOUD and not include_personal:
        raise PermissionError("personal index is private (server not started with --include-personal)")
    p = _root() / cloud / "index.md"
    if not p.exists():
        raise FileNotFoundError(f"No index for cloud '{cloud}'")
    return p.read_text(encoding="utf-8", errors="replace")


def read_page(page_path: str, include_personal: bool = False) -> str:
    return _note_for(page_path, include_personal).path.read_text(encoding="utf-8", errors="replace")


def page_metadata(page_path: str, include_personal: bool = False) -> dict:
    n = _note_for(page_path, include_personal)
    return {"path": _rel(n.path), "title": n.title, "type": n.meta.get("type"), "cloud": n.cloud,
            "frontmatter": n.meta, "links": [wt.link_target_stem(l) for l in n.links]}


def list_section(section: str = PUBLIC_CLOUD, recursive: bool = False,
                 include_personal: bool = False) -> list[str]:
    raw = section.strip().replace("\\", "/").lstrip("/") or PUBLIC_CLOUD
    root = _root().resolve()
    base = (root / raw).resolve()
    if root not in base.parents and base != root:
        raise ValueError("Path escapes the wiki root")
    if not base.is_dir():
        raise ValueError("Expected a section (directory) path")
    out = []
    for p in sorted(base.rglob("*.md") if recursive else base.glob("*.md")):
        try:
            if is_public(wt.Note(p), include_personal):
                out.append(_rel(p))
        except Exception:
            continue
    return out


def search(query: str, k: int = 10, include_personal: bool = False) -> list[dict]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    by, docs = {}, []
    for n in _public_notes(include_personal):
        rel = _rel(n.path)
        blob = (n.title + " " + " ".join(n.as_list("aliases")) + " "
                + " ".join(n.as_list("tags")) + " " + n.body)
        docs.append((rel, wt._tokens(blob)))
        by[rel] = n
    ranked = wt.bm25_rank(query, docs)[: max(1, k)]
    return [{"path": rel, "title": by[rel].title, "score": round(score, 3),
             "snippet": _snippet(by[rel].body, query)} for rel, score in ranked]


def resolve_links(page_path: str, include_missing: bool = False,
                  include_personal: bool = False) -> list[dict]:
    note = _note_for(page_path, include_personal)
    pub = {n.name: n for n in _public_notes(include_personal)}
    out = []
    for link in note.links:
        target = pub.get(wt.link_target_stem(link))
        if target:
            out.append({"link": link, "path": _rel(target.path), "title": target.title,
                        "content": target.path.read_text(encoding="utf-8", errors="replace")})
        elif include_missing:
            out.append({"link": link, "missing": True})
    return out


def get_source(src_id: str) -> dict:
    """Safe source-registry lookup: identity + presence, but NOT raw machine file paths."""
    entry = wt.load_registry().get(src_id)
    if not entry:
        raise FileNotFoundError(f"Unknown source {src_id}")
    present = any(loc.get("present") for loc in entry.get("locations", {}).values())
    return {"id": entry["id"], "title": entry.get("title"),
            "content_type": entry.get("content_type"), "sha256": entry.get("sha256", ""),
            "first_ingested": entry.get("first_ingested"), "present_somewhere": present}


# --------------------------------------------------------------------------- #
# structured query over the projection (Phase D) — same engine as the CLI, public-filtered.
# Read-only canned queries only (no raw SQL over MCP); the `public=True` view restricts to the shared
# cloud (home/personal withheld), matching the privacy model above.
# --------------------------------------------------------------------------- #
def list_queries() -> dict:
    import projection
    return {name: meta[2] for name, meta in projection.CANNED.items()}


def query(name: str, source=None, from_=None, type=None, of=None, tag=None, note=None) -> list:
    """Run a canned structured query against the projection, restricted to the public (shared) view."""
    import projection
    params = {"source": source, "from": from_, "type": type, "of": of, "tag": tag, "note": note}
    return projection.run_canned(name, params, public=True)


# --------------------------------------------------------------------------- #
# generated views (Phase H) — rendered from the projection, public-filtered, LLM off.
# --------------------------------------------------------------------------- #
def list_views() -> dict:
    import views
    return {name: meta[2] for name, meta in views.VIEWS.items()}


def render_view(name: str, of=None, fmt="md") -> str:
    """Render a view (markdown by default) from the projection, restricted to the public view."""
    import views
    return views.render(name, params={"of": of}, public=True, fmt=fmt)
