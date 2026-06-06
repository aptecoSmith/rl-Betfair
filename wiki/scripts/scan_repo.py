"""scan_repo.py - rl-betfair repo-scan ingester (a thin wrapper over register + batch).

Walks the rl-betfair markdown knowledge corpus, registers each file as a v3 source
(referenced in place - never copied), and queues it in a resumable `batch` ledger for
the `batch` skill to compile ONE source at a time (anti-shortcut: a note per
concept / finding / decision, not per file). This is rl-betfair's own front door for
the operator's "point it at the whole folder and ingest" requirement.

Scope v1: markdown KNOWLEDGE only (NOT .py code; NOT process scaffolding). What it sweeps:
  - plans/**/{purpose,lessons_learnt,findings,design,design_decisions}.md — the
    knowledge-bearing files. master_todo / progress / session_prompt(s) /
    hard_constraints are per-plan process scaffolding (task lists, changelogs,
    build prompts) and are deliberately NOT queued — they'd cost compile tokens for
    little knowledge. Widen PLANS_KNOWLEDGE_NAMES if that judgement changes.
  - docs/**/*.md — curated reference, all of it.
  - cross-cutting logs + root knowledge docs: plans/EXPERIMENTS.md,
    plans/EXPLORATIONS.md, CLAUDE.md, genes_census.md.
The wiki's own notes/machinery (wiki/**) are never ingested as sources.

Registration uses the engine's `intake.register_folder` / `wiki_tool.register_source`
(sources are referenced, not copied; locations are machine-keyed in
Schema/sources.jsonl). Queuing uses the engine's `batch` ledger
(.runtime/batch/<name>.json - gitignored, machine-local). A source only counts
'done' once a real note cites it (coverage-tied), so `batch status` reconciles the
queue against compiled knowledge, not files touched.

Usage:
  python scripts/scan_repo.py             # register + queue (ledger 'repo-md')
  python scripts/scan_repo.py --dry-run   # preview what would be registered; write nothing
  python scripts/scan_repo.py --name foo  # queue into a differently-named ledger

After running, work the queue via the `batch` skill:
  python scripts/wiki_tool.py ... (batch status / next), then ingest + finalize-ingest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiki_tool as wt   # noqa: E402
import batch             # noqa: E402

REPO = wt.ROOT.parent            # wiki/ is vendored inside the rl-betfair repo
MD_EXTS = {".md", ".markdown"}
# plans/ is swept for KNOWLEDGE-BEARING files only; the rest is process scaffolding.
PLANS_KNOWLEDGE_NAMES = {"purpose.md", "lessons_learnt.md", "findings.md",
                         "design.md", "design_decisions.md"}
SWEEP_ALL_DIRS = ["docs"]            # curated reference — every *.md
SWEEP_KNOWLEDGE_DIRS = ["plans"]     # only PLANS_KNOWLEDGE_NAMES basenames
# cross-cutting logs (under plans/) + root knowledge docs, referenced individually
NAMED_DOCS = ["plans/EXPERIMENTS.md", "plans/EXPLORATIONS.md", "CLAUDE.md", "genes_census.md"]
# never descend into these (the wiki's own notes/machinery, vcs/venv/caches)
SKIP_DIRS = {"wiki", ".git", ".venv", "venv", "node_modules", "__pycache__", ".obsidian"}


def _md_files():
    """Yield knowledge markdown: all of docs/, knowledge-named files under plans/, + named docs."""
    for d in SWEEP_ALL_DIRS:
        base = REPO / d
        if base.exists():
            for p in sorted(base.rglob("*")):
                if (p.is_file() and p.suffix.lower() in MD_EXTS
                        and not any(part in SKIP_DIRS for part in p.parts)):
                    yield p
    for d in SWEEP_KNOWLEDGE_DIRS:
        base = REPO / d
        if base.exists():
            for p in sorted(base.rglob("*")):
                if (p.is_file() and p.name in PLANS_KNOWLEDGE_NAMES
                        and not any(part in SKIP_DIRS for part in p.parts)):
                    yield p
    for f in NAMED_DOCS:
        p = REPO / f
        if p.is_file():
            yield p


def collect(dry_run: bool = False):
    """Return [(loc, sid)] for every markdown knowledge file, registered in place (no copy).

    Locations are canonical forward-slash absolute paths (`Path.resolve().as_posix()`) so the
    same file always hashes to the same src-id regardless of how it was first registered — this
    is what lets the bulk scan dedupe against the hand-ingested proof-first sources.
    """
    seen: set = set()
    out: list = []
    for p in _md_files():
        loc = p.resolve().as_posix()
        if loc in seen:
            continue
        seen.add(loc)
        sid = None if dry_run else wt.register_source(loc, is_file=True)
        out.append((loc, sid))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scan_repo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="repo-md", help="batch ledger name (default: repo-md)")
    ap.add_argument("--dry-run", action="store_true", help="preview; register nothing, queue nothing")
    args = ap.parse_args(argv)

    items = collect(dry_run=args.dry_run)
    verb = "would register" if args.dry_run else "registered"
    print(f"{verb} {len(items)} markdown KNOWLEDGE source(s) "
          f"(plans/ {'|'.join(sorted(n[:-3] for n in PLANS_KNOWLEDGE_NAMES))} + docs/ + named logs)")
    if args.dry_run:
        for path, _ in items[:20]:
            print(f"  {path}")
        if len(items) > 20:
            print(f"  ... (+{len(items) - 20} more)")
        return 0

    ledger = batch.load_ledger(args.name)
    added = batch.plan_entries(ledger, [(sid, path) for path, sid in items if sid])
    batch.save_ledger(ledger)
    s = batch.summary(ledger)
    print(f"queued {added} new source(s) into batch '{args.name}' "
          f"(.runtime/batch/{args.name}.json, gitignored)")
    print(f"batch '{args.name}': {s['total']} total, {s['pending']} pending, "
          f"{s['done']} done, {s['skipped']} skipped")
    print("next: compile the queue (the `batch` skill - it is scripts/batch.py, NOT a wiki_tool subcommand):")
    print(f"  python scripts/batch.py --name {args.name} status        # progress + the next source")
    print(f"  python scripts/batch.py --name {args.name} next          # claim one -> ingest -> finalize-ingest")
    print(f"  python scripts/batch.py --name {args.name} done <src-id> # after a note cites it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
