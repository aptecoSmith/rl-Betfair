"""Resumable batch-ingest ledger for large corpora.

Ports llm_wiki's "batch with per-file progress" so a big migration survives interruption - but routes
every source through v2's quality model instead of bulk-dumping notes. The ledger is just the work
queue; the agent still compiles each source via the extract/ingest skills + finalize-ingest. A source
is only counted 'done' once real notes cite it (the coverage floor via wiki_tool.coverage_map), so
progress reflects compiled knowledge, not files touched.

The ledger lives in .runtime/batch/<name>.json (machine-local, gitignored) so it never pollutes the
wiki. Pure ledger ops (plan_entries, claim_next, mark, reconcile, summary) are unit-tested; IO + CLI
wrap them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiki_tool as wt  # noqa: E402
import intake  # noqa: E402

STATUSES = ("pending", "in_progress", "done", "skipped")


# --------------------------------------------------------------------------- #
# pure ledger ops (no IO)
# --------------------------------------------------------------------------- #
def new_ledger(name: str) -> dict:
    return {"name": name, "entries": {}}


def plan_entries(ledger: dict, items) -> int:
    """Add (sid, path) items as pending. Existing entries keep their status (resumable). Returns # added."""
    entries = ledger.setdefault("entries", {})
    added = 0
    for sid, path in items:
        if sid not in entries:
            entries[sid] = {"path": str(path), "status": "pending", "notes": 0, "reason": ""}
            added += 1
    return added


def claim_next(ledger: dict, resume_first: bool = True):
    """Next unit of work as (sid, path, resumed), or None when drained.

    One-at-a-time lock (resume_first=True, the default): if a source is already `in_progress`, hand THAT
    one back so it gets finished before another is started — this is what stops the agent reading many
    sources and summarising them together. `resume_first=False` is the `--force` override: skip the lock
    and claim the next pending source (deliberate parallelism)."""
    entries = ledger.get("entries", {})
    if resume_first:                                          # finish an in-flight source before a new one
        for sid in sorted(entries):
            if entries[sid]["status"] == "in_progress":
                return sid, entries[sid]["path"], True
    for sid in sorted(entries):
        if entries[sid]["status"] == "pending":
            return sid, entries[sid]["path"], False
    for sid in sorted(entries):                               # fallback: resume an in_progress one
        if entries[sid]["status"] == "in_progress":
            return sid, entries[sid]["path"], True
    return None


def mark(ledger: dict, sid: str, status: str, **fields) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    e = ledger.setdefault("entries", {}).get(sid)
    if e is None:
        raise KeyError(sid)
    e["status"] = status
    e.update(fields)


def reconcile(ledger: dict, coverage: dict) -> dict:
    """Tie ledger status to real notes (the coverage floor). coverage: {sid: [note_paths]}.

    - a pending/in_progress source now cited by >=1 note -> done (auto-detected).
    - a done source that has lost all citing notes -> back to pending (flagged as regressed).
    Records the note count on each entry. Returns {"completed": [...], "regressed": [...]}.
    """
    completed, regressed = [], []
    for sid, e in ledger.get("entries", {}).items():
        n = len(coverage.get(sid, []))
        e["notes"] = n
        if e["status"] in ("pending", "in_progress") and n >= 1:
            e["status"] = "done"
            completed.append(sid)
        elif e["status"] == "done" and n == 0:
            e["status"] = "pending"
            regressed.append(sid)
    return {"completed": completed, "regressed": regressed}


def summary(ledger: dict) -> dict:
    counts = {s: 0 for s in STATUSES}
    for e in ledger.get("entries", {}).values():
        st = e.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
    total = sum(counts[s] for s in STATUSES)
    accounted = counts["done"] + counts["skipped"]
    counts["total"] = total
    counts["percent"] = round(accounted * 100 / total) if total else 0
    return counts


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def ledger_path(name: str) -> Path:
    return wt.ROOT / ".runtime" / "batch" / f"{name}.json"


def load_ledger(name: str) -> dict:
    p = ledger_path(name)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return new_ledger(name)


def save_ledger(ledger: dict) -> Path:
    p = ledger_path(ledger["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_summary(name: str, s: dict) -> None:
    print(f"batch '{name}': {s['percent']}% accounted "
          f"({s['done']} done, {s['skipped']} skipped, {s['in_progress']} in-progress, "
          f"{s['pending']} pending of {s['total']})")


def _require(ledger: dict, sid: str) -> bool:
    if sid not in ledger.get("entries", {}):
        print(f"error: {sid} not in batch '{ledger['name']}'", file=sys.stderr)
        return False
    return True


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="batch", description="Resumable batch-ingest ledger for large corpora.")
    ap.add_argument("--name", default="default", help="ledger name (run several migrations side by side)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan", help="register every doc under a folder and queue it (resumable)")
    sp.add_argument("folder")
    sp.add_argument("--no-recursive", action="store_true")
    sub.add_parser("status", help="reconcile against real notes and show progress")
    sp = sub.add_parser("next", help="print the next source to ingest and mark it in-progress")
    sp.add_argument("--force", action="store_true",
                    help="override the one-at-a-time lock: start a new source even if one is in progress")
    sp = sub.add_parser("done", help="confirm a source done (refuses if no note cites it yet)")
    sp.add_argument("sid")
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("skip", help="mark a source skipped, with a reason (accounted, not dropped)")
    sp.add_argument("sid")
    sp.add_argument("--reason", required=True)
    sp = sub.add_parser("requeue", help="put a source back to pending")
    sp.add_argument("sid")
    args = ap.parse_args(argv)

    ledger = load_ledger(args.name)

    if args.cmd == "plan":
        items = [(sid, path) for path, sid in
                 intake.register_folder(args.folder, recursive=not args.no_recursive)]
        added = plan_entries(ledger, items)
        save_ledger(ledger)
        print(f"registered + queued {added} new source(s) ({len(items)} seen)")
        _print_summary(args.name, summary(ledger))
        return 0

    if args.cmd == "status":
        r = reconcile(ledger, wt.coverage_map())
        save_ledger(ledger)
        if r["completed"]:
            print(f"auto-completed (now cited by notes): {', '.join(r['completed'])}")
        if r["regressed"]:
            print(f"REGRESSED (lost all citing notes): {', '.join(r['regressed'])}")
        _print_summary(args.name, summary(ledger))
        nxt = claim_next(ledger)
        if nxt:
            print(f"next: {nxt[0]}  {nxt[1]}")
        return 0

    if args.cmd == "next":
        reconcile(ledger, wt.coverage_map())                 # auto-close any in-flight source notes now cite
        nxt = claim_next(ledger, resume_first=not args.force)
        if not nxt:
            save_ledger(ledger)
            print("queue drained - nothing pending")
            return 0
        sid, path, resumed = nxt
        mark(ledger, sid, "in_progress", ts=wt._today())
        save_ledger(ledger)
        if resumed and not args.force:
            # one-at-a-time lock: a source is still in flight - finish it before starting another
            print(f"{sid}  {path}")
            print("^ STILL IN PROGRESS - finish this source first: discover -> notes/claims -> "
                  "finalize-ingest -> `batch done`.")
            print("  Process one source fully at a time. (`batch next --force` to deliberately parallelise.)")
        else:
            print(("resuming " if resumed else "") + f"{sid}  {path}")
            print("-> discover (entities) -> ingest (extract/ingest skills) -> finalize-ingest -> `batch done`")
        return 0

    if args.cmd == "done":
        n = len(wt.coverage_map().get(args.sid, []))
        if n == 0 and not args.force:
            print(f"refusing: no notes cite {args.sid} yet. Ingest it and run finalize-ingest first "
                  f"(or --force to override).", file=sys.stderr)
            return 1
        if not _require(ledger, args.sid):
            return 2
        mark(ledger, args.sid, "done", notes=n, ts=wt._today())
        save_ledger(ledger)
        print(f"done {args.sid} ({n} citing note(s))")
        return 0

    if args.cmd == "skip":
        if not _require(ledger, args.sid):
            return 2
        mark(ledger, args.sid, "skipped", reason=args.reason, ts=wt._today())
        save_ledger(ledger)
        print(f"skipped {args.sid}: {args.reason}")
        return 0

    if args.cmd == "requeue":
        if not _require(ledger, args.sid):
            return 2
        mark(ledger, args.sid, "pending")
        save_ledger(ledger)
        print(f"requeued {args.sid}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
