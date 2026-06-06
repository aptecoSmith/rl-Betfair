"""FastMCP server exposing the v2 wiki read-only to other agents.

Logic lives in core.py (no mcp dependency, unit-tested); this module is the thin MCP wiring. By
default only the shared cloud is served; pass --include-personal for local/trusted use to also expose
personal notes.

    python -m mcp_server                      # stdio (default)
    python -m mcp_server --transport sse      # network
    python -m mcp_server --include-personal   # also expose personal/ + home-tagged notes (local only)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

logger = logging.getLogger("llm-wiki-v2-mcp")
mcp = FastMCP("llm-wiki-v2")

# set by main(); default is the safe (shared-only) view
INCLUDE_PERSONAL = False


@mcp.tool()
def get_index() -> str:
    """Return the shared cloud's index.md - the catalogue/hub to start navigation from."""
    return core.get_index(include_personal=INCLUDE_PERSONAL)


@mcp.tool()
def read_page(page_path: str) -> str:
    """Read a wiki note's full markdown by path, e.g. shared/concepts/tdd.md.

    Args:
        page_path: Path under a cloud (shared/...). Personal notes are withheld unless the server
                   was started with --include-personal.
    """
    return core.read_page(page_path, include_personal=INCLUDE_PERSONAL)


@mcp.tool()
def get_page_metadata(page_path: str) -> dict[str, Any]:
    """Return a note's frontmatter (type/cloud/status/tags/sources), title and outgoing links.

    Args:
        page_path: Path under a cloud, e.g. shared/entities/vix.md
    """
    return core.page_metadata(page_path, include_personal=INCLUDE_PERSONAL)


@mcp.tool()
def list_section(section: str = "shared", recursive: bool = False) -> list[str]:
    """List wiki note paths under a section (e.g. shared, shared/concepts, shared/entities).

    Args:
        section: Directory under a cloud. Defaults to the shared cloud root.
        recursive: Recurse into child directories.
    """
    return core.list_section(section, recursive=recursive, include_personal=INCLUDE_PERSONAL)


@mcp.tool()
def search_pages(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """BM25 full-text search across shared wiki notes. Returns path, title, score and a snippet.

    Args:
        query: Search text.
        max_results: Maximum matches to return.
    """
    return core.search(query, k=max_results, include_personal=INCLUDE_PERSONAL)


@mcp.tool()
def resolve_links(page_path: str, include_missing: bool = False) -> list[dict[str, Any]]:
    """Resolve a note's [[wiki-links]] and return the linked notes' content.

    Args:
        page_path: The source note's path.
        include_missing: Also report links that don't resolve to a (public) note.
    """
    return core.resolve_links(page_path, include_missing=include_missing,
                              include_personal=INCLUDE_PERSONAL)


@mcp.tool()
def get_source(src_id: str) -> dict[str, Any]:
    """Look up a registered source by id (e.g. src-23573d): title, type, hash, presence.

    Raw machine file paths are deliberately NOT returned.
    """
    return core.get_source(src_id)


@mcp.tool()
def list_queries() -> dict[str, str]:
    """List the structured queries available over the wiki projection, and what each answers.

    These run with the LLM off, against the derived SQLite projection (shared cloud only). Use
    query_wiki(name, ...) to run one.
    """
    return core.list_queries()


@mcp.tool()
def query_wiki(name: str, source: str = "", from_id: str = "", type: str = "",
               of: str = "", tag: str = "", note: str = "") -> list[dict[str, Any]]:
    """Run a canned structured query over the wiki projection (shared-cloud only, read-only).

    Same engine and results as the CLI `wiki_tool query --public`. Answers questions like
    "unverified claims from source X", "who is X's sibling", "what supersedes Z" — from data, no LLM.

    Args:
        name: one of list_queries() — claims-by-source, unverified-claims, relations, supersedes,
              contradictions, sources, tag, note-claims.
        source: source id (for claims-by-source / unverified-claims).
        from_id: note id or name (for relations).
        type: edge type filter (for relations) or content_type (for sources).
        of: note id/name (for supersedes — what supersedes this).
        tag: tag name (for tag).
        note: note id/path (for note-claims).
    """
    return core.query(name, source=source or None, from_=from_id or None, type=type or None,
                      of=of or None, tag=tag or None, note=note or None)


@mcp.tool()
def list_views() -> dict[str, str]:
    """List the generated views available over the projection, and what each shows (shared cloud only)."""
    return core.list_views()


@mcp.tool()
def render_view(name: str, of: str = "", fmt: str = "md") -> str:
    """Render a viewer-independent view from the projection (read-only, shared cloud, LLM off).

    Args:
        name: one of list_views() — verified-only, unverified-claims, dialog-trail,
              supersession-timeline, draft-vs-stable.
        of: conclusion/claim/event id (required for dialog-trail).
        fmt: 'md' (default) or 'html'.
    """
    return core.render_view(name, of=of or None, fmt=fmt)


def main() -> None:
    global INCLUDE_PERSONAL
    parser = argparse.ArgumentParser(description="Run the llm-wiki-v3 MCP server (read-only).")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default=None, help="bind host for sse (e.g. 0.0.0.0 in a container)")
    parser.add_argument("--port", type=int, default=None, help="bind port for sse")
    parser.add_argument("--include-personal", action="store_true",
                        help="also expose personal/ + home-tagged notes (local/trusted use only)")
    args = parser.parse_args()
    INCLUDE_PERSONAL = args.include_personal
    if args.host is not None:
        mcp.settings.host = args.host
    if args.port is not None:
        mcp.settings.port = args.port
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting llm-wiki-v3 MCP server (%s, personal=%s)",
                args.transport, INCLUDE_PERSONAL)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
