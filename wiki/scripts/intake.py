"""Intake front doors (the testable ones): inbox drops, point-at folder/repo, URLs doc, bookmarks.

Sources are referenced, not copied (D4) - registration records a location, it does not duplicate
bytes. Dropped inbox files are the one exception: they live in the inbox until processed, then move
to processed/. Network scrapers and Google Drive live in scrape.py / a skill (they need network or
the agent's connector).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiki_tool as wt  # noqa: E402

DOC_EXTS = {".md", ".markdown", ".txt", ".rst", ".pdf", ".docx", ".pptx", ".xlsx", ".csv"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".obsidian"}
URL_RE = re.compile(r"https?://[^\s\)\"'<>]+")
HREF_RE = re.compile(r'href=["\'](https?://[^"\']+)["\']', re.I)


# --------------------------------------------------------------------------- #
# point-at folder / repo
# --------------------------------------------------------------------------- #
def register_folder(folder, recursive=True, exts=DOC_EXTS, dry_run=False):
    """Register every doc file under a folder, in place (no copy). Returns list of (path, sid).

    dry_run=True returns (path, None) without writing the registry - used to preview a migration.
    """
    base = Path(folder)
    out = []
    walker = base.rglob("*") if recursive else base.glob("*")
    for p in sorted(walker):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in exts:
            continue
        sid = None if dry_run else wt.register_source(str(p.resolve()), is_file=True)
        out.append((str(p), sid))
    return out


# --------------------------------------------------------------------------- #
# inbox drops
# --------------------------------------------------------------------------- #
def scan_inbox(cloud="inbox"):
    files_dir = wt.ROOT / cloud / "pending" / "Files"
    if not files_dir.exists():
        return []
    return [p for p in sorted(files_dir.rglob("*"))
            if p.is_file() and p.name != ".gitkeep"]


def move_to_processed(file_path, cloud="inbox"):
    src = Path(file_path)
    dest_dir = wt.ROOT / cloud / "processed"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    src.replace(dest)
    return dest


def scan_static(cloud="inbox"):
    """Files in the static (append-only) drop folder - referenced in place, snapshotted not moved."""
    d = wt.ROOT / cloud / "static"
    if not d.exists():
        return []
    return [p for p in sorted(d.rglob("*")) if p.is_file() and p.name != ".gitkeep"]


def scan_clippings():
    """Markdown the Obsidian Web Clipper dropped in the root Clippings/ folder (pending ingestion)."""
    d = wt.ROOT / "Clippings"
    if not d.exists():
        return []
    return [p for p in sorted(d.glob("*.md")) if p.name != ".gitkeep"]


# --------------------------------------------------------------------------- #
# urls doc + bookmarks
# --------------------------------------------------------------------------- #
def existing_urls(reg=None):
    reg = reg if reg is not None else wt.load_registry()
    urls = set()
    for e in reg.values():
        for loc in e.get("locations", {}).values():
            if "url" in loc:
                urls.add(loc["url"])
    return urls


def parse_urls_doc(path):
    urls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s+(https?://\S+)", line)
        if m:
            urls.append(m.group(1))
    return urls


def parse_bookmarks(path):
    """Accept a browser HTML export or a plain URL list; return de-duplicated URLs (in-file)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if path.lower().endswith((".html", ".htm")):
        found = HREF_RE.findall(text)
    else:
        found = URL_RE.findall(text)
    seen, out = set(), []
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def new_urls(urls, reg=None):
    """Filter out URLs already in the registry (dedup against ingested)."""
    have = existing_urls(reg)
    return [u for u in urls if u not in have]


def register_urls(urls):
    return [wt.register_source(u, is_file=False) for u in urls]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="intake")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("folder"); s.add_argument("path"); s.add_argument("--no-recursive", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("inbox"); s.add_argument("--personal", action="store_true")
    s = sub.add_parser("bookmarks"); s.add_argument("path")
    args = ap.parse_args(argv)

    if args.cmd == "folder":
        res = register_folder(args.path, recursive=not args.no_recursive, dry_run=args.dry_run)
        for path, sid in res:
            print(f"{sid or '(dry-run)'}  {path}")
        print(f"{'would register' if args.dry_run else 'registered'} {len(res)} file(s)")
    elif args.cmd == "inbox":
        cloud = "inbox_personal" if args.personal else "inbox"
        files = scan_inbox(cloud)
        for f in files:
            print(f"pending: {f}")
        statics = scan_static(cloud)
        for f in statics:
            print(f"static:  {f}")
        clips = scan_clippings()
        for f in clips:
            print(f"clip:    {f}")
        print(f"{len(files)} pending, {len(statics)} static, {len(clips)} clippings")
    elif args.cmd == "bookmarks":
        urls = new_urls(parse_bookmarks(args.path))
        for u in urls:
            print(u)
        print(f"{len(urls)} new url(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
