"""dialog.py — the CoDIAK dialog/decision record (v3 Phase E). Stdlib only.

Engelbart's CoDIAK runs three domains continuously: intelligence (ingest), **dialog** (how decisions
were reached), and the knowledge product (the notes). v2 captured the first and third but let the
*dialog* evaporate — reasoning lived in chat and disappeared. This module makes it a first-class,
**addressable, append-only** record at `Schema/dialog.jsonl`, so *how the wiki got to what it says* is
itself part of the wiki.

One entry per event:
    {id: <ULID>, ts, kind, text, links: [ids], author}
where `kind ∈ {question, decision, contradiction, rejection, schema-change}` and `links` are the
claim/note/source/dialog ids the event concerns — so a resolved contradiction is reconstructable from
the record alone (it links the conflicting claims *and* the contradiction it resolves).

This is **semantic** dialog, not a commit log (that's git) and not the replication manifest (that's the
append-only `ingest-log.jsonl`, which stays). Append-only — never rewritten.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import wiki_tool as wt        # call-time use only (no import cycle: wt imports us lazily)

DIALOG_KINDS = ("question", "decision", "contradiction", "rejection", "schema-change")


def dialog_path() -> Path:
    return Path(wt.SCHEMA) / "dialog.jsonl"


def append_event(kind, text, links=None, author="human", ts=None, eid=None) -> str:
    """Append one dialog event and return its id. Append-only; never rewrite the file."""
    if kind not in DIALOG_KINDS:
        raise ValueError(f"unknown dialog kind '{kind}' (want one of {', '.join(DIALOG_KINDS)})")
    ev = {"id": eid or wt.new_ulid(),
          "ts": ts or _dt.datetime.now().isoformat(timespec="seconds"),
          "kind": kind, "text": text, "links": list(links or []), "author": author}
    p = dialog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
    return ev["id"]


def load_dialog() -> list:
    p = dialog_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def dialog_trail(events, anchor) -> list:
    """Reconstruct the reasoning trail around `anchor` (a claim/note id OR a dialog event id).

    Gathers every event that references `anchor`, then transitively follows links that are themselves
    dialog events (so a resolution that links the contradiction it resolved pulls the contradiction in,
    and vice-versa). Returns the events ordered by timestamp.
    """
    by_id = {e["id"]: e for e in events if "id" in e}
    seen, frontier = {}, [anchor]
    while frontier:
        a = frontier.pop()
        for e in events:
            if e.get("id") == a or a in e.get("links", []):
                if e["id"] not in seen:
                    seen[e["id"]] = e
                    for l in e.get("links", []):
                        if l in by_id and l not in seen:
                            frontier.append(l)
    return sorted(seen.values(), key=lambda e: (e.get("ts", ""), e.get("id", "")))
