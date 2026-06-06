"""bootstrap.py — the tool-system co-evolution loop (v3 Phase G). Stdlib only.

Engelbart's bootstrapping (the ABC model): **A** = the work, **B** = improving how you do A, **C** =
improving how you improve. Almost no knowledge tool does B/C deliberately. This module closes the loop:
it watches for places where the *tool-system* underperforms — coverage-floor misses, schema/vocabulary
friction, recurring lint failures, low benchmark scores — and emits **versioned, evidence-backed
improvement proposals** for the schema, extraction prompts, or tag/link vocabularies. The wiki improves
*and the way the wiki is built improves* — measurably.

**Propose, a human disposes.** Nothing here auto-applies a change. Proposals are recorded as
`schema-change` events in the CoDIAK dialog record (each carrying the evidence that triggered it); a
human `accept`/`reject` is a `decision`; an accepted proposal bumps the tool-system version
(`Schema/tool-version.json`). Every change is reversible.
"""
from __future__ import annotations

import json
from pathlib import Path

import wiki_tool as wt        # call-time use only (no import cycle: wt imports us lazily)
import dialog as _dlg


def tool_version_path() -> Path:
    return Path(wt.SCHEMA) / "tool-version.json"


def load_tool_version() -> dict:
    p = tool_version_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": 1, "accepted": [], "rejected": []}


def save_tool_version(v: dict):
    tool_version_path().write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# signal detection — where is the tool-system underperforming?
# --------------------------------------------------------------------------- #
def _source_pages(path) -> int | None:
    """Page/segment count of a source via the optional extractors (feature-detected). None if absent."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extractors"))
        import extract as _ex
        return _ex.extract(str(path)).page_count()
    except Exception:
        return None


def gather_signals(floor: float = 1.0) -> list:
    """Read the live wiki and return quality/friction signals (the evidence proposals cite).

    Signals: coverage-floor misses (notes/page below `floor`), unknown-tag / unknown-link-type friction
    (from validate), and — when the benchmarking harness imports — structural weaknesses (thin notes,
    duplicate titles, orphans). Each signal is a dict carrying its own evidence.
    """
    signals = []
    notes = wt.find_notes()
    reg = wt.load_registry()
    cov = wt.coverage_map()                                   # source_id -> [note paths]
    mid = wt.machine_id()
    # 1. coverage-floor misses (the "weak extraction" signal)
    for sid, paths in cov.items():
        e = reg.get(sid)
        if not e:
            continue
        loc = e.get("locations", {}).get(mid) or {}
        p = loc.get("path")
        if not p:
            continue
        fp = Path(p) if Path(p).is_absolute() else (Path(wt.ROOT) / p)
        if not fp.exists():
            continue
        pages = _source_pages(fp)
        if pages and len(paths) / pages < floor:
            signals.append({"kind": "coverage", "source": sid, "title": e.get("title", sid),
                            "notes": len(paths), "pages": pages,
                            "ratio": round(len(paths) / pages, 3), "floor": floor})
    # 2. vocabulary / schema friction (from validate findings)
    findings = wt.validate(notes, wt.load_vocab(), reg)
    tag_hits, ltype_hits = {}, {}
    for _lvl, _path, msg in findings:
        m = msg
        if "not in context-tag vocabulary" in m:
            tag = m.split("'")[1] if "'" in m else "?"
            tag_hits[tag] = tag_hits.get(tag, 0) + 1
        elif "unknown link type" in m:
            lt = m.split("'")[1] if "'" in m else "?"
            ltype_hits[lt] = ltype_hits.get(lt, 0) + 1
    for tag, n in tag_hits.items():
        signals.append({"kind": "unknown-tag", "tag": tag, "count": n})
    for lt, n in ltype_hits.items():
        signals.append({"kind": "unknown-link-type", "type": lt, "count": n})
    # 3. structural weakness via the benchmarking harness (feature-detected)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarking"))
        import score as _score
        sc = _score.score(Path(wt.ROOT))
        if sc.get("thin_notes", 0) > 0:
            signals.append({"kind": "thin-notes", "count": sc["thin_notes"]})
        if sc.get("orphans", 0) > 0:
            signals.append({"kind": "orphans", "count": sc["orphans"]})
    except Exception:
        pass
    return signals


# --------------------------------------------------------------------------- #
# proposal generation — pure: signals in, evidence-backed proposals out
# --------------------------------------------------------------------------- #
def propose_from_signals(signals: list) -> list:
    """Map signals to concrete, reversible improvement proposals. Pure (no I/O)."""
    proposals = []
    for s in signals:
        k = s.get("kind")
        if k == "coverage":
            proposals.append({
                "target": "extraction/coverage-floor",
                "text": (f"Under-extracted source '{s.get('title')}' ({s['source']}): {s['notes']} "
                         f"note(s) for {s['pages']} page(s) = {s['ratio']} < floor {s['floor']}. "
                         f"Re-run the extract skill on it, or raise the notes/page floor for reference "
                         f"docs if this source is genuinely sparse."),
                "evidence": s})
        elif k == "unknown-tag":
            proposals.append({
                "target": "Schema/tag-vocabulary.md",
                "text": (f"Add context tag '{s['tag']}' to the controlled vocabulary, or retag the "
                         f"{s['count']} note(s) using it. Recurring use suggests a real gap."),
                "evidence": s})
        elif k == "unknown-link-type":
            proposals.append({
                "target": "Schema/link-types.md",
                "text": (f"Add link type '{s['type']}' (with an inverse) to the edge vocabulary, or "
                         f"remap the {s['count']} edge(s) using it to an existing type."),
                "evidence": s})
        elif k == "thin-notes":
            proposals.append({
                "target": "extraction/extract-skill",
                "text": (f"{s['count']} thin note(s) detected. Tighten the extract skill to enumerate "
                         f"per-segment substance, or lower the thin-note threshold if intentional."),
                "evidence": s})
        elif k == "orphans":
            proposals.append({
                "target": "connectivity/ingest-skill",
                "text": (f"{s['count']} orphan note(s) unreachable from a hub. Strengthen the "
                         f"'link into a hub' step, or add a hub for the cluster."),
                "evidence": s})
    return proposals


def emit_proposals(proposals: list, author="model:bootstrap") -> list:
    """Record each proposal as a `schema-change` dialog event carrying its evidence. Returns event ids."""
    ids = []
    for p in proposals:
        text = f"PROPOSAL [{p['target']}]: {p['text']}  EVIDENCE: {json.dumps(p['evidence'], sort_keys=True)}"
        eid = _dlg.append_event("schema-change", text, links=[], author=author)
        ids.append(eid)
    return ids


def dispose(proposal_id: str, accept: bool, note: str = "") -> dict:
    """A human accepts/rejects a proposal. Accept bumps the tool-system version. Logged as a decision."""
    v = load_tool_version()
    verb = "accepted" if accept else "rejected"
    if accept:
        v["version"] += 1
        v.setdefault("accepted", []).append({"proposal": proposal_id, "version": v["version"]})
    else:
        v.setdefault("rejected", []).append(proposal_id)
    save_tool_version(v)
    _dlg.append_event("decision",
                      f"{verb} tool-system proposal {proposal_id}"
                      + (f" -> tool-version {v['version']}" if accept else "")
                      + (f". {note}" if note else ""),
                      links=[proposal_id], author="human")
    return v
