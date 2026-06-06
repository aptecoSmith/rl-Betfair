"""tag_superseded.py - mark stale notes with the `superseded` tag (deterministic).

A note that a `supersedes` edge points at is stale knowledge (the pre-change reference). This script
derives that set from the supersedes edges (via `wiki_tool.py relations --stale`) and ensures each such
note carries the `superseded` context tag, so the Obsidian graph can colour stale notes distinctly
(see wiki/.obsidian/graph.json). Idempotent: re-run after each ingest wave; a note already tagged is
left untouched, and the tag is only ADDED (never removed - supersession is monotonic in practice; if an
edge is deleted, drop the tag by hand).

This is structural, not editorial (v3 invariant 4: deterministic code extracts structure), so the tag
is applied by code, not by the ingesting LLM. Run: `python wiki/scripts/tag_superseded.py [--dry-run]`.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # wiki/
TAG = "superseded"
_STALE_RE = re.compile(r"^\s*STALE\s+(\S+)\s+superseded by", re.MULTILINE)
_TAGS_RE = re.compile(r"^(\s*tags:\s*)\[(.*)\]\s*$")


def stale_notes() -> list[str]:
    """ROOT-relative paths of notes made stale by a supersedes edge (deduped)."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "wiki_tool.py"), "relations", "--stale"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return sorted({m.group(1).replace("\\", "/") for m in _STALE_RE.finditer(r.stdout or "")})


def ensure_tag(note_rel: str, dry_run: bool = False) -> bool:
    """Add `superseded` to the note's frontmatter tags. Returns True if it changed (or would)."""
    p = ROOT / note_rel
    if not p.exists():
        return False
    lines = p.read_text(encoding="utf-8").split("\n")
    if not lines or lines[0].strip() != "---":
        return False  # no frontmatter; leave it
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return False
    for i in range(1, end):
        m = _TAGS_RE.match(lines[i])
        if m:
            items = [x.strip() for x in m.group(2).split(",") if x.strip()]
            if TAG in items:
                return False
            items.append(TAG)
            lines[i] = f"{m.group(1)}[{', '.join(items)}]"
            break
    else:
        lines.insert(end, f"tags: [{TAG}]")   # no tags line -> add one
    if not dry_run:
        p.write_text("\n".join(lines), encoding="utf-8")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tag_superseded", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args(argv)

    stale = stale_notes()
    changed = [n for n in stale if ensure_tag(n, dry_run=args.dry_run)]
    verb = "would tag" if args.dry_run else "tagged"
    print(f"{len(stale)} stale note(s); {verb} {len(changed)} (rest already tagged)")
    for n in changed:
        print(f"  {n}")
    if changed and not args.dry_run:
        print("note: run `wiki_tool.py build` (or finalize-ingest) to refresh the projection, then commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
