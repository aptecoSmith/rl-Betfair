#!/usr/bin/env python3
"""dream.py - generate interesting questions from the wiki's own graph (the "llm-wiki dream").

Pure, deterministic generators over a list of note records (so they're testable without the live
wiki). Each question is tagged internal (answerable by synthesizing existing notes) or external
(needs new sources -> gated on operator approval). An append-only open-questions register tracks
status (open/queued/answered/dismissed) so overnight runs don't re-ask dismissed questions.

A note record is a dict: {name, cloud, type, title, links:[names], tags:[..], body_len:int}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

INTERNAL, EXTERNAL = "internal", "external"
STUB_INDEGREE = 3      # referenced by >= this many notes
STUB_BODYLEN = 400     # but shorter than this -> a stub hub
THIN_TOPIC_MEMBERS = 2  # topic with <= this many members is thin


def _qid(text: str) -> str:
    return "q-" + hashlib.sha1(text.lower().encode("utf-8")).hexdigest()[:8]


def _q(text, kind, generator, score, provenance):
    return {"id": _qid(text), "text": text, "kind": kind,
            "generator": generator, "score": round(float(score), 3),
            "provenance": sorted(provenance)}


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #
def index_notes(records):
    return {r["name"]: r for r in records}


def undirected_adjacency(records):
    by_name = index_notes(records)
    adj = {r["name"]: set() for r in records}
    for r in records:
        for tgt in r.get("links", []):
            if tgt in by_name:
                adj[r["name"]].add(tgt)
                adj[tgt].add(r["name"])
    return adj


def indegree(records):
    deg = {r["name"]: 0 for r in records}
    by_name = index_notes(records)
    for r in records:
        for tgt in r.get("links", []):
            if tgt in by_name:
                deg[tgt] += 1
    return deg


def topic_members(records):
    """A note 'belongs to' a topic if it is adjacent to that topic-type note."""
    adj = undirected_adjacency(records)
    topics = {r["name"] for r in records if r.get("type") == "topic"}
    members = {t: set() for t in topics}
    for t in topics:
        for n in adj.get(t, ()):
            if n != t:
                members[t].add(n)
    return members


def _linked_between(a_set, b_set, adj):
    for a in a_set:
        if adj.get(a, set()) & b_set:
            return True
    return False


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def gen_structural_holes(records):
    """Two topic clusters with no link between them -> a bridging question (internal)."""
    members = topic_members(records)
    adj = undirected_adjacency(records)
    topics = sorted(members)
    out = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            t1, t2 = topics[i], topics[j]
            m1, m2 = members[t1], members[t2]
            if len(m1) < 2 or len(m2) < 2:
                continue
            if t2 in adj.get(t1, set()):       # topics directly linked - not a hole
                continue
            if _linked_between(m1 | {t1}, m2 | {t2}, adj):
                continue
            score = len(m1) * len(m2)
            out.append(_q(f"How does {t1} relate to {t2}?", INTERNAL,
                          "structural-hole", score, [t1, t2]))
    return out


def gen_recurring_entities(records):
    """An entity linked from notes across >=2 distinct topics -> why? (internal)."""
    members = topic_members(records)
    out = []
    for r in records:
        if r.get("type") != "entity":
            continue
        e = r["name"]
        in_topics = sorted(t for t, ms in members.items() if e in ms)
        if len(in_topics) >= 2:
            score = len(in_topics)
            joined = " and ".join(in_topics)
            out.append(_q(f"Why does {e} recur across {joined}?", INTERNAL,
                          "recurring-entity", score, [e] + in_topics))
    return out


def gen_stub_hubs(records):
    """Heavily-referenced but thin notes -> deepen them (external: usually needs sources)."""
    deg = indegree(records)
    out = []
    for r in records:
        if r.get("type") in ("topic", "log", "query"):
            continue
        d = deg.get(r["name"], 0)
        if d >= STUB_INDEGREE and r.get("body_len", 0) < STUB_BODYLEN:
            out.append(_q(f"Deepen {r['name']}: referenced by {d} notes but still a stub.",
                          EXTERNAL, "stub-hub", d, [r["name"]]))
    return out


def gen_thin_topics(records):
    """Topic with few members -> grow it (external: needs sources)."""
    members = topic_members(records)
    out = []
    for t, ms in members.items():
        if len(ms) <= THIN_TOPIC_MEMBERS:
            out.append(_q(f"Grow the {t} topic: only {len(ms)} member note(s).",
                          EXTERNAL, "thin-topic", 1.0 + (THIN_TOPIC_MEMBERS - len(ms)), [t]))
    return out


GENERATORS = [gen_structural_holes, gen_recurring_entities, gen_stub_hubs, gen_thin_topics]


def generate_questions(records):
    seen, out = set(), []
    for gen in GENERATORS:
        for q in gen(records):
            if q["id"] in seen:
                continue
            seen.add(q["id"])
            out.append(q)
    return out


def rank_questions(questions):
    """Stable rank: internal first (cheaper/safer), then score desc, then id."""
    ordered = sorted(questions, key=lambda q: (q["kind"] != INTERNAL, -q["score"], q["id"]))
    for i, q in enumerate(ordered, 1):
        q["rank"] = i
    return ordered


# --------------------------------------------------------------------------- #
# open-questions register
# --------------------------------------------------------------------------- #
def load_register(path):
    p = Path(path)
    reg = {}
    if not p.exists():
        return reg
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            reg[obj["id"]] = obj
        except json.JSONDecodeError:
            pass
    return reg


def merge_register(register, questions):
    """Add genuinely new questions as status=open; never resurrect dismissed/answered ones.

    Returns (merged_register, new_open_list)."""
    new_open = []
    for q in questions:
        existing = register.get(q["id"])
        if existing is None:
            entry = dict(q, status="open")
            register[q["id"]] = entry
            new_open.append(entry)
        # existing dismissed/answered/queued -> leave as-is (no re-ask)
    return register, new_open


def save_register(path, register):
    lines = ["# Open-questions register - dream skill. status: open|queued|answered|dismissed"]
    for obj in register.values():
        lines.append(json.dumps(obj, ensure_ascii=False, sort_keys=True))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


RESOLVED = ("dismissed", "answered", "queued")


def pending_candidates(register, questions):
    """Candidates not already resolved in the register - i.e. still awaiting a prune verdict."""
    resolved = {qid for qid, e in register.items() if e.get("status") in RESOLVED}
    return [q for q in questions if q["id"] not in resolved]


def apply_verdicts(register, judged):
    """Record the LLM's prune verdicts. Each item is a candidate plus a 'status'
    (open|dismissed|queued) and optional 'reason'. Never overwrites an answered question."""
    for q in judged:
        qid = q.get("id")
        if not qid:
            continue
        if register.get(qid, {}).get("status") == "answered":
            continue
        entry = {k: q[k] for k in ("id", "text", "kind", "generator", "score", "provenance") if k in q}
        entry["status"] = q.get("status", "open")
        if q.get("reason"):
            entry["reason"] = q["reason"]
        register[qid] = entry
    return register


# --------------------------------------------------------------------------- #
# adapter to the live wiki (runtime only; not used by unit tests)
# --------------------------------------------------------------------------- #
def records_from_wiki():
    import wiki_tool as wt  # type: ignore
    recs = []
    for n in wt.find_notes():
        recs.append({
            "name": n.name, "cloud": n.cloud, "type": n.meta.get("type", ""),
            "title": n.title, "links": [wt.link_target_stem(t) for t in n.links],
            "tags": n.as_list("tags"), "body_len": len(n.body),
        })
    return recs


def cmd_propose(args):
    """List candidates awaiting a verdict. The LLM prunes the obvious ones, then `record`s."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    register = load_register(args.register)
    questions = rank_questions(generate_questions(records_from_wiki()))
    pending = pending_candidates(register, questions)
    if args.internal_only:
        pending = [q for q in pending if q["kind"] == INTERNAL]
    pending = pending[: args.limit]
    if args.json:
        print(json.dumps(pending, ensure_ascii=False, indent=2))
        return 0
    print(f"# {len(pending)} candidate(s) awaiting a prune verdict "
          f"(dismiss the obvious, then `dream.py record`)\n")
    for q in pending:
        flag = "AUTO " if q["kind"] == INTERNAL else "GATED"
        print(f"{q['rank']:2}. [{flag}] ({q['score']}) {q['text']}  [{q['id']}]")
        print(f"     from: {', '.join(q['provenance'])}")
    return 0


def cmd_record(args):
    """Apply prune verdicts. Reads a JSON array of judged candidates (id + status [+reason])."""
    data = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    register = apply_verdicts(load_register(args.register), json.loads(data))
    save_register(args.register, register)
    counts = {}
    for e in register.values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    print("register: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dream")
    ap.add_argument("--register", default="Schema/open-questions.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="list candidates for the LLM to prune")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--internal-only", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_propose)
    r = sub.add_parser("record", help="apply prune verdicts (open/dismissed/queued)")
    r.add_argument("--file", help="JSON file of judged candidates; omit to read stdin")
    r.set_defaults(func=cmd_record)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
