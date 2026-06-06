#!/usr/bin/env python3
"""wiki_tool.py - deterministic engine for llm-wiki-v3. Stdlib only.

Subcommands: init, doctor, register, build, validate, connectivity, search,
source-check, log, scan, finalize-ingest, stamp-ids, claim-add, claims-lint,
relations, project, query, dialog, verify, bootstrap, view, coverage, rollback,
classify. Run `wiki_tool.py <cmd> --help`.

v3 Phase A adds granular object IDs: every note carries a permanent ULID in its
frontmatter (the source of truth), links resolve by ID so renames never break
them, and a derived filename<->id alias index keeps legacy [[wikilink]] text
working. Stdlib-only throughout — the zero-dependency core still runs unchanged.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "Schema"
CLOUDS = ["shared", "personal"]
LOCAL_CONFIG = ROOT / ".wiki-local.json"
REGISTRY = SCHEMA / "sources.jsonl"
CATALOG = SCHEMA / "catalog.jsonl"
TAG_VOCAB = SCHEMA / "tag-vocabulary.md"

STATUSES = {"seed", "draft", "stable"}
LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]*)?\]\]")
# Same as LINK_RE but also captures the optional |display alias as group 2 (used by relink).
LINK_FULL_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]*))?\]\]")
# A note named/titled like a page — the tell-tale of page-padding a reference doc instead of extracting
# concepts/objectives/keywords. Matches a page marker ANYWHERE in the name (use .search), so it catches
# bare "p12"/"page 4-5" AND prefixed forms like "istqb-ctfl-p30-..." / "p1-cover" / "...-p8-9-acks".
# (Mirrored in doctypes.PAGE_NAME_RE — keep in sync.)
PAGE_NAME_RE = re.compile(r"(?:^|[-_ ])(?:pp?|pg|pages?)[-_ ]?\d+(?:[-_ ]?\d+)?(?=[-_ ]|$)", re.I)

# severity levels for findings
ERROR, WARN = "error", "warn"

# --------------------------------------------------------------------------- #
# Granular object IDs (Phase A) — a ULID in frontmatter is the source of truth.
# A ULID is a 26-char Crockford-base32 string: a 48-bit millisecond timestamp +
# 80 bits of randomness, lexicographically sortable. Generated with the stdlib
# only (time + os.urandom), so the zero-dependency core keeps working with no
# external ULID library. Links resolve by ID first, filename second.
# --------------------------------------------------------------------------- #
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"          # Crockford base32 (no I L O U)
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")        # canonical (uppercase) ULID
# A link BY ID: [[id:01J9Z3...]] — rename-safe. Case-insensitive on input.
ID_LINK_RE = re.compile(r"^id:([0-9A-HJKMNP-TV-Za-hjkmnp-tv-z]{26})$")
# Block-level anchor, RESERVED for Phase B claims (not implemented in Phase A):
# a trailing ` ^blockid` marks an addressable sub-note block. Reserved now so
# authors and the claim model don't collide later. See frontmatter-contract.md.
BLOCK_ID_RE = re.compile(r"[ \t]\^([0-9A-Za-z][0-9A-Za-z_-]{1,31})[ \t]*$")


def _b32(value: int, length: int) -> str:
    """Encode the low bits of `value` as `length` Crockford-base32 chars (big-endian)."""
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def new_ulid(ts_ms: int | None = None) -> str:
    """Generate a fresh ULID (stdlib only). Time prefix keeps IDs roughly sortable by creation."""
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")          # 80 random bits
    return _b32(ts_ms & ((1 << 48) - 1), 10) + _b32(rand, 16)


def is_ulid(value) -> bool:
    """True iff `value` is a canonical (uppercase) 26-char ULID string."""
    return isinstance(value, str) and bool(ULID_RE.match(value))


def parse_link(target: str):
    """Classify a [[link]] target. Returns ('id', <ULID>) for [[id:...]] else ('stem', <stem>)."""
    t = (target or "").strip()
    m = ID_LINK_RE.match(t)
    if m:
        return ("id", m.group(1).upper())
    return ("stem", link_target_stem(t))


def build_index_maps(notes):
    """Derived, in-memory alias index: (by_id, by_stem).

    by_id:  ULID -> [Note] (normally one; >1 means a duplicate-id corruption, caught by validate).
    by_stem: filename-stem -> [Note] (>1 means an ambiguous stem, flagged by the resolver).
    Recomputed from the files on every call — never persisted as a source of truth.
    """
    by_id, by_stem = {}, {}
    for n in notes:
        if n.id:
            by_id.setdefault(n.id, []).append(n)
        by_stem.setdefault(n.name, []).append(n)
    return by_id, by_stem


def resolve_link(target, by_id, by_stem):
    """Resolve a [[link]] to note(s). Returns (matches: list[Note], kind).

    kind ∈ {'id', 'stem', 'ambiguous', 'dangling'}. Prefers an exact ID match; falls back to the
    filename stem; reports 'ambiguous' when a stem maps to more than one note.
    """
    kind, key = parse_link(target)
    if kind == "id":
        matches = by_id.get(key, [])
        return (matches, "id" if matches else "dangling")
    matches = by_stem.get(key, [])
    if not matches:
        return ([], "dangling")
    if len(matches) > 1:
        return (matches, "ambiguous")
    return (matches, "stem")


def _node_key(note) -> str:
    """Stable graph-node key for a note: its ULID if stamped, else a filename-based fallback.

    The fallback keeps the zero-dependency / pre-migration path working (un-stamped notes still
    graph by name); once stamped, the key is rename-invariant.
    """
    return note.id or ("name:" + note.name)


def _resolve_ref(ref, by_id, by_stem):
    """Resolve a typed-link `to` (a bare ULID, an `id:...`, or a filename stem) to note(s)."""
    r = str(ref).strip()
    if is_ulid(r.upper()):
        m = by_id.get(r.upper(), [])
        return (m, "id" if m else "dangling")
    return resolve_link(r, by_id, by_stem)


def typed_edges(notes, by_id, by_stem):
    """Resolve every frontmatter typed link to a (from_note, to_note, type) triple.

    Skips edges whose `to` doesn't resolve to a note (connectivity reports those as dangling).
    """
    out = []
    for n in notes:
        for tl in n.typed_links:
            matches, kind = _resolve_ref(tl.get("to"), by_id, by_stem)
            for m in matches:
                out.append((n, m, str(tl.get("type", ""))))
    return out


def relations_of(note, edges, inverses):
    """All typed relations of `note`, from its perspective, including derived inverses.

    Returns [(relation_type, other_note)]. An edge A→B (type T) appears as (T, B) for A and as
    (inverse(T), A) for B — so "who is X's sibling" is answerable from either end, no LLM inference.
    """
    out = []
    for a, b, t in edges:
        if a is note and b is not note:
            out.append((t, b))
        elif b is note and a is not note:
            out.append((inverses.get(t, t), a))
    return out


def stale_notes(edges):
    """(stale_note, superseding_note) for each `supersedes` edge — the older note (and its claims) is stale."""
    return [(b, a) for (a, b, t) in edges if t == "supersedes"]


def contradiction_edges(edges):
    """(note_a, note_b) for each `contradicts` edge — surfaced to the audit for resolution."""
    return [(a, b) for (a, b, t) in edges if t == "contradicts"]

# junk guard: what finalize is allowed to auto-commit
ALLOWED_TOP_DIRS = {"shared", "personal", "Schema", "scripts", "extractors", "skills",
                    "automation", "templates", "docs", "tests", "inbox", "inbox_personal",
                    ".obsidian", ".githooks"}
ALLOWED_ROOT_FILES = {"AGENTS.md", "CLAUDE.md", "README.md", "requirements.txt",
                      ".gitignore", ".gitattributes", "start_claude.bat", "start_codex.bat"}


# --------------------------------------------------------------------------- #
# frontmatter parsing (constrained YAML subset, see Schema/frontmatter-contract.md)
# --------------------------------------------------------------------------- #
def _split_top_level(s: str, sep: str = ","):
    """Split `s` on `sep` only at brace/bracket depth 0 (so `{a: 1, b: 2}` stays intact)."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _flow_mapping(s: str):
    """Parse a YAML flow mapping `{to: X, type: Y}` into a dict, or None if `s` isn't one.

    Used for typed links (Phase C): `links: [{to: <id>, type: <type>}]`. Keeps the stdlib parser.
    """
    s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    inner = s[1:-1].strip()
    d = {}
    for part in _split_top_level(inner, ","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip()] = _scalar(v.strip())
    return d


def _scalar(raw: str):
    s = raw.strip()
    if s == "" or s == "[]":
        return [] if s == "[]" else ""
    flow = _flow_mapping(s)
    if flow is not None:
        return flow
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return s


def _inline_list(raw: str):
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    return [_scalar(x) for x in _split_top_level(inner, ",") if x.strip() != ""]


def parse_frontmatter(text: str):
    """Return (meta: dict, body: str)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    meta = {}
    i = 1
    while i < end:
        line = lines[i]
        if line.strip() == "" or line.lstrip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z0-9_\-]+):(.*)$", line)
        if not m:
            i += 1
            continue
        key, rest = m.group(1), m.group(2).strip()
        if rest.startswith("[") and rest.endswith("]"):
            meta[key] = _inline_list(rest)
        elif rest == "":
            # possible block list
            items = []
            j = i + 1
            while j < end and re.match(r"^\s*-\s+", lines[j]):
                items.append(_scalar(re.sub(r"^\s*-\s+", "", lines[j])))
                j += 1
            if items:
                meta[key] = items
                i = j
                continue
            meta[key] = ""
        else:
            meta[key] = _scalar(rest)
        i += 1
    body = "\n".join(lines[end + 1:])
    return meta, body


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_local_config() -> dict:
    if LOCAL_CONFIG.exists():
        return json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
    return {}


def machine_id() -> str:
    cfg = load_local_config()
    return cfg.get("machine_id") or socket.gethostname()


def load_vocab() -> dict:
    """Parse the fenced lists out of tag-vocabulary.md."""
    out = {"types": [], "entity-subtypes": [], "context-tags": []}
    if not TAG_VOCAB.exists():
        return out
    text = TAG_VOCAB.read_text(encoding="utf-8")
    for key in out:
        m = re.search(r"<!--\s*%s\s*-->(.*?)<!--\s*/%s\s*-->" % (key, key), text, re.S)
        if m:
            out[key] = re.findall(r"^\s*-\s+(\S+)", m.group(1), re.M)
    return out


def load_link_types() -> dict:
    """Parse the controlled edge vocabulary + inverse map from Schema/link-types.md (Phase C).

    Returns {'types': set(...), 'inverses': {type: inverse}}. Missing file => empty (typed links are
    an opt-in power-up; the zero-dependency core works without the doc, only typed edges then fail
    validation as 'unknown type').
    """
    out = {"types": set(), "inverses": {}}
    p = SCHEMA / "link-types.md"
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8")
    for fam in ("epistemic", "structural", "entity"):
        m = re.search(r"<!--\s*%s\s*-->(.*?)<!--\s*/%s\s*-->" % (fam, fam), text, re.S)
        if m:
            out["types"].update(re.findall(r"^\s*-\s+(\S+)", m.group(1), re.M))
    m = re.search(r"<!--\s*inverses\s*-->(.*?)<!--\s*/inverses\s*-->", text, re.S)
    if m:
        for a, b in re.findall(r"^\s*-\s+(\S+)\s*:\s*(\S+)", m.group(1), re.M):
            out["inverses"][a] = b
    return out


def load_registry() -> dict:
    reg = {}
    if not REGISTRY.exists():
        return reg
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        reg[obj["id"]] = obj
    return reg


def save_registry(reg: dict):
    lines = ["# Source registry - one JSON object per line. Managed by wiki_tool (register / source-check)."]
    for obj in reg.values():
        lines.append(json.dumps(obj, ensure_ascii=False, sort_keys=True))
    REGISTRY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ingest_log_path():
    return SCHEMA / "ingest-log.jsonl"


def append_ingest_log(event: dict):
    """Append-only, immutable record of ingest events - the replication manifest.

    Unlike the registry (current state, mutated by source-check), this is never rewritten, so it is
    an ordered, faithful history: location + hash of every source as it entered, plus finalize marks.
    """
    event = dict(event)
    event.setdefault("ts", _dt.datetime.now().isoformat(timespec="seconds"))
    p = _ingest_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def load_ingest_log():
    p = _ingest_log_path()
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


# --------------------------------------------------------------------------- #
# note model
# --------------------------------------------------------------------------- #
class Note:
    def __init__(self, path: Path):
        self.path = path
        self.cloud = path.relative_to(ROOT).parts[0]
        self.name = path.stem
        text = path.read_text(encoding="utf-8")
        self.meta, self.body = parse_frontmatter(text)
        self.links = [m.strip() for m in LINK_RE.findall(self.body)]

    @property
    def title(self):
        m = re.search(r"^#\s+(.+)$", self.body, re.M)
        return m.group(1).strip() if m else self.name

    @property
    def id(self) -> str:
        """The note's permanent ULID (uppercased), or '' if not yet stamped."""
        v = self.meta.get("id", "")
        return v.upper() if isinstance(v, str) else ""

    @property
    def typed_links(self) -> list:
        """Frontmatter typed edges (Phase C): list of {'to': <id/stem>, 'type': <type>}.

        Distinct from `self.links` (untyped body `[[wiki-links]]`, treated as see-also). [] if none.
        """
        v = self.meta.get("links", [])
        if not isinstance(v, list):
            return []
        return [x for x in v if isinstance(x, dict) and x.get("to")]

    def as_list(self, key):
        v = self.meta.get(key, [])
        if isinstance(v, list):
            return v
        return [v] if v else []


HUB_NAMES = {"index"}


def _rel(path) -> str:
    """Repo-root-relative POSIX path string for findings/catalog (ROOT is rebound in tests)."""
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def find_notes() -> list[Note]:
    notes = []
    for cloud in CLOUDS:
        base = ROOT / cloud
        if not base.exists():
            continue
        for p in base.rglob("*.md"):
            if p.name in ("log.md",):
                continue
            notes.append(Note(p))
    return notes


def link_target_stem(target: str) -> str:
    return target.split("/")[-1].replace(".md", "").strip()


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_init(args):
    cfg = load_local_config()
    cfg.setdefault("machine_id", args.machine or socket.gethostname())
    cfg.setdefault("active_cloud", "shared")
    cfg["vector_tier"] = _detect_vector_tier()
    LOCAL_CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"machine_id   = {cfg['machine_id']}")
    print(f"active_cloud = {cfg['active_cloud']}")
    print(f"vector_tier  = {'available' if cfg['vector_tier'] else 'BM25 only (optional bits absent)'}")
    return cmd_doctor(args)


def _detect_vector_tier() -> bool:
    try:
        import numpy  # noqa: F401
        return (ROOT / "scripts" / "models").exists()
    except Exception:
        return False


def cmd_doctor(args):
    ok = True
    print(f"python       = {sys.version.split()[0]}")
    for d in CLOUDS + ["Schema", "scripts", "templates", "inbox", "inbox_personal"]:
        exists = (ROOT / d).exists()
        ok = ok and exists
        print(f"dir {d:<14} {'ok' if exists else 'MISSING'}")
    print(f"config       = {'present' if LOCAL_CONFIG.exists() else 'run init'}")
    reg = load_registry()
    notes = find_notes()
    print(f"sources      = {len(reg)}")
    print(f"notes        = {len(notes)}")
    vocab = load_vocab()
    print(f"types        = {', '.join(vocab['types']) or 'NONE (check tag-vocabulary.md)'}")
    # search tier (detect by import availability; never instantiate -> no model download)
    import importlib.util as _u
    has_np = _u.find_spec("numpy") is not None
    has_m2v = _u.find_spec("model2vec") is not None
    if has_np and has_m2v:
        tier = "BM25 + semantic vectors (model2vec)"
    elif has_np:
        tier = "BM25 (hashing vector fallback; install model2vec for semantic)"
    else:
        tier = "BM25 only (install numpy + model2vec for semantic)"
    print(f"search       = {tier}")
    print("               see docs/search.md to enable semantic search")
    return 0 if ok else 1


def register_source(location, is_file=True, title=None, content_type=None):
    """Programmatic registration. Returns the source id. Reference-not-copy: records location only."""
    sha = ""
    present = True
    if is_file:
        p = Path(location)
        if p.exists():
            sha = _sha256_file(p)
        else:
            present = False
    sid = "src-" + hashlib.sha1(str(location).encode("utf-8")).hexdigest()[:6]
    reg = load_registry()
    mid = machine_id()
    today = _today()
    entry = reg.get(sid, {
        "id": sid,
        "title": title or Path(location).name,
        "content_type": content_type or (Path(location).suffix.lstrip(".") or "url"),
        "sha256": sha,
        "first_ingested": today,
        "locations": {},
    })
    entry["locations"][mid] = {("path" if is_file else "url"): str(location),
                               "present": present, "last_seen": today if present else ""}
    if sha and not entry.get("sha256"):
        entry["sha256"] = sha
    reg[sid] = entry
    save_registry(reg)
    append_ingest_log({"event": "register", "id": sid, "title": entry["title"],
                       "content_type": entry["content_type"], "sha256": entry.get("sha256", ""),
                       "location": str(location), "machine": mid, "present": present})
    return sid


def cmd_register(args):
    if not args.path and not args.url:
        print("error: --path or --url required", file=sys.stderr)
        return 2
    sid = register_source(args.path or args.url, is_file=bool(args.path),
                          title=args.title, content_type=args.content_type)
    print(sid)
    return 0


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _today() -> str:
    return _dt.date.today().isoformat()


# --------------------------------------------------------------------------- #
# stamp-ids — granular ID migration (Phase A)
# --------------------------------------------------------------------------- #
def stamp_note_id(path: Path, dry_run: bool = False):
    """Ensure a note's frontmatter carries a valid ULID `id:`. Idempotent.

    Returns the new ULID if one was added (or would be, under --dry-run), else None when the note
    already has a valid id or has no frontmatter to stamp. Surgical: inserts `id:` as the first
    frontmatter line (or replaces a malformed id line) without reformatting the rest of the file.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None                                   # no frontmatter — leave it (validate will warn)
    meta, _ = parse_frontmatter(text)
    existing = meta.get("id", "")
    if existing and is_ulid(str(existing).upper()):
        return None                                   # already stamped — idempotent no-op
    new_id = new_ulid()
    lines = text.splitlines(keepends=True)
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    id_line_idx = None
    if end is not None:
        for i in range(1, end):
            if re.match(r"^id\s*:", lines[i]):
                id_line_idx = i
                break
    if not dry_run:
        if id_line_idx is not None:                   # replace a malformed/blank id line in place
            lines[id_line_idx] = f"id: {new_id}\n"
        else:                                         # insert right after the opening '---'
            lines.insert(1, f"id: {new_id}\n")
        path.write_text("".join(lines), encoding="utf-8")
    return new_id


def relink_note(note: Note, by_id, by_stem, dry_run: bool = False) -> int:
    """Rewrite this note's `[[stem]]` links to rename-safe `[[id:<ULID>|stem]]` form.

    Only rewrites links whose stem resolves to exactly one *stamped* note (unambiguous + has an id);
    ambiguous or unstamped targets are left as-is. Preserves an existing |display alias, otherwise
    uses the target's filename stem as readable display text. Returns the number of links rewritten.
    """
    text = note.path.read_text(encoding="utf-8")
    count = 0

    def _repl(m):
        nonlocal count
        whole, inner, display = m.group(0), m.group(1), m.group(2)
        kind, key = parse_link(inner)
        if kind == "id":
            return whole                              # already id-form
        matches = by_stem.get(key, [])
        if len(matches) == 1 and matches[0].id and matches[0] is not note:
            tgt = matches[0]
            count += 1
            return f"[[id:{tgt.id}|{display or key}]]"
        return whole

    new_text = LINK_FULL_RE.sub(_repl, text)
    if count and not dry_run:
        note.path.write_text(new_text, encoding="utf-8")
    return count


def cmd_stamp_ids(args):
    notes = [n for n in find_notes() if n.name not in HUB_NAMES]
    stamped = []
    for n in notes:
        nid = stamp_note_id(n.path, dry_run=args.dry_run)
        if nid:
            stamped.append((_rel(n.path), nid))
    for rel, nid in stamped:
        print(f"{'would stamp' if args.dry_run else 'stamped'}  {rel}  id={nid}")
    relinked = 0
    if args.relink:
        notes = [n for n in find_notes() if n.name not in HUB_NAMES]   # reload to see new ids
        by_id, by_stem = build_index_maps(notes)
        for n in notes:
            c = relink_note(n, by_id, by_stem, dry_run=args.dry_run)
            if c:
                relinked += c
                print(f"{'would relink' if args.dry_run else 'relinked'}  {_rel(n.path)}  ({c} link(s))")
    verb = "would " if args.dry_run else ""
    summary = f"stamp-ids: {verb}stamp {len(stamped)} note(s)"
    if args.relink:
        summary += f", {verb}relink {relinked} link(s)"
    print(summary)
    return 0


def cmd_build(args):
    notes = find_notes()
    reg = load_registry()
    by_id, by_stem = build_index_maps(notes)
    # catalog
    cat_lines = []
    for n in sorted(notes, key=lambda x: str(x.path)):
        cat_lines.append(json.dumps({
            "id": n.id,
            "path": str(n.path.relative_to(ROOT)).replace("\\", "/"),
            "cloud": n.cloud,
            "type": n.meta.get("type", ""),
            "title": n.title,
            "tags": n.as_list("tags"),
            "sources": n.as_list("sources"),
            "aliases": n.as_list("aliases"),
            "updated": n.meta.get("updated", ""),
            "links": [link_target_stem(t) for t in n.links],
            # resolved outgoing edges by target id (rename-safe view of the graph)
            "links_resolved": _resolved_link_ids(n, by_id, by_stem),
            # typed edges (Phase C): [{to, type}] resolved to target ids where possible
            "typed_links": _resolved_typed_links(n, by_id, by_stem),
        }, ensure_ascii=False, sort_keys=True))
    CATALOG.write_text("\n".join(cat_lines) + ("\n" if cat_lines else ""), encoding="utf-8")
    # derived filename<->id alias index (regenerable; gitignored under .runtime/; never the truth)
    _write_alias_index(by_id, by_stem)
    # per-cloud index hubs
    for cloud in CLOUDS:
        cnotes = [n for n in notes if n.cloud == cloud and n.name not in HUB_NAMES]
        _write_index(cloud, cnotes)
    # render Sources blocks from registry
    rendered = 0
    for n in notes:
        if _render_sources_block(n, reg):
            rendered += 1
    print(f"built catalog ({len(notes)} notes), indexes for {', '.join(CLOUDS)}, sources blocks: {rendered}")
    return 0


def _resolved_link_ids(note, by_id, by_stem):
    """Outgoing links resolved to target ids where possible (else the raw stem). Rename-safe."""
    out = []
    for t in note.links:
        matches, kind = resolve_link(t, by_id, by_stem)
        if matches:
            out.extend(m.id or ("name:" + m.name) for m in matches)
        else:
            out.append(link_target_stem(t))
    return out


def _resolved_typed_links(note, by_id, by_stem):
    """Typed edges with `to` resolved to a target id where possible. For the catalog/projection."""
    out = []
    for tl in note.typed_links:
        matches, kind = _resolve_ref(tl.get("to"), by_id, by_stem)
        target = (matches[0].id or ("name:" + matches[0].name)) if matches else str(tl.get("to"))
        out.append({"to": target, "type": str(tl.get("type", ""))})
    return out


def _write_alias_index(by_id, by_stem):
    """Materialize the derived alias index to .runtime/alias-index.json for the other surfaces.

    A convenience cache only: every code path computes the maps in-memory from the files, so a stale
    or missing file is never load-bearing. Gitignored (.runtime/).
    """
    idx = {
        "by_id": {nid: {"name": ms[0].name, "path": _rel(ms[0].path),
                        "title": ms[0].title, "cloud": ms[0].cloud}
                  for nid, ms in sorted(by_id.items())},
        "by_stem": {stem: sorted(m.id for m in ms if m.id)
                    for stem, ms in sorted(by_stem.items())},
    }
    out = ROOT / ".runtime"
    out.mkdir(parents=True, exist_ok=True)
    (out / "alias-index.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_index(cloud: str, notes: list[Note]):
    base = ROOT / cloud
    if not base.exists():
        return
    by_type: dict[str, list[Note]] = {}
    for n in notes:
        by_type.setdefault(n.meta.get("type", "untyped"), []).append(n)
    out = [f"# {cloud} index", "", "_Generated by `wiki_tool build`. The hub for connectivity._", ""]
    for t in sorted(by_type):
        out.append(f"## {t}")
        for n in sorted(by_type[t], key=lambda x: x.title.lower()):
            out.append(f"- [[{n.name}]] - {n.title}")
        out.append("")
    (base / "index.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def _render_sources_block(note: Note, reg: dict) -> bool:
    if "## Sources" not in note.body:
        return False
    sids = note.as_list("sources")
    rows = []
    for sid in sids:
        e = reg.get(sid)
        if not e:
            rows.append(f"- `{sid}` (not in registry)")
            continue
        locs = e.get("locations", {})
        present_any = any(v.get("present") for v in locs.values())
        where = "; ".join(f"{m}:{'present' if v.get('present') else 'missing'}" for m, v in locs.items())
        status = "" if present_any else "  **[source missing - cited via lifted quote]**"
        rows.append(f"- `{sid}` {e.get('title','')} ({where}){status}")
    block = "## Sources\n" + ("\n".join(rows) if rows else "_none_") + "\n"
    new_body = re.sub(r"## Sources\n(?:.*?)(?=\n## |\Z)", block, note.body, count=1, flags=re.S)
    if new_body != note.body:
        full = "---\n" + _dump_frontmatter(note.meta) + "---\n" + new_body
        note.path.write_text(full, encoding="utf-8")
        note.body = new_body
        return True
    return False


def _fmt_item(x):
    """Serialize a list item back to the constrained-YAML subset (incl. typed-link flow mappings)."""
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {v}" for k, v in x.items()) + "}"
    return str(x)


def _dump_frontmatter(meta: dict) -> str:
    out = []
    for k, v in meta.items():
        if isinstance(v, list):
            out.append(f"{k}: [{', '.join(_fmt_item(x) for x in v)}]")
        elif isinstance(v, bool):
            out.append(f"{k}: {'true' if v else 'false'}")
        else:
            out.append(f"{k}: {v}")
    return "\n".join(out) + "\n"


def cmd_validate(args):
    findings = validate(find_notes(), load_vocab(), load_registry())
    return _report(findings, args)


def validate(notes, vocab, reg):
    findings = []
    types = set(vocab["types"])
    subtypes = set(vocab["entity-subtypes"])
    ctags = set(vocab["context-tags"])
    link_types = load_link_types()["types"]
    for n in notes:
        rel = str(n.path.relative_to(ROOT)).replace("\\", "/")
        if n.name in HUB_NAMES:
            continue
        # Phase A: granular object id. Missing is a (gentle) warning so the migration is incremental
        # and the zero-dependency path keeps working; a malformed id is a real corruption -> error.
        raw_id = n.meta.get("id", "")
        if not raw_id:
            findings.append((WARN, rel, "missing 'id' (run: wiki_tool stamp-ids)"))
        elif not is_ulid(str(raw_id).upper()):
            findings.append((ERROR, rel, f"id '{raw_id}' is not a valid ULID"))
        t = n.meta.get("type")
        if not t:
            findings.append((ERROR, rel, "missing 'type'"))
            continue
        if t not in types:
            findings.append((ERROR, rel, f"type '{t}' not in vocabulary"))
        if "subtype" in n.meta and n.meta["subtype"]:
            if t != "entity":
                findings.append((ERROR, rel, "subtype only allowed on entity notes"))
            elif n.meta["subtype"] not in subtypes:
                findings.append((ERROR, rel, f"subtype '{n.meta['subtype']}' not in vocabulary"))
        cloud = n.meta.get("cloud")
        if cloud not in CLOUDS:
            findings.append((ERROR, rel, f"cloud '{cloud}' invalid"))
        elif cloud != n.cloud:
            findings.append((ERROR, rel, f"cloud '{cloud}' does not match folder '{n.cloud}'"))
        status = n.meta.get("status")
        if status not in STATUSES:
            findings.append((ERROR, rel, f"status '{status}' invalid"))
        # Phase C: typed links — type must be in the controlled vocabulary; `to` required.
        for tl in n.typed_links:
            ltype = str(tl.get("type", ""))
            if not ltype:
                findings.append((ERROR, rel, "typed link missing 'type'"))
            elif ltype not in link_types:
                findings.append((ERROR, rel,
                                 f"unknown link type '{ltype}' (see Schema/link-types.md)"))
        sources = n.as_list("sources")
        if status in ("draft", "stable") and not sources:
            findings.append((ERROR, rel, f"status '{status}' requires non-empty sources"))
        for d in ("created", "updated"):
            val = str(n.meta.get(d, ""))
            try:
                _dt.date.fromisoformat(val)
            except ValueError:
                findings.append((ERROR, rel, f"{d} '{val}' not ISO-8601"))
        for tag in n.as_list("tags"):
            if tag not in ctags:
                findings.append((ERROR, rel, f"tag '{tag}' not in context-tag vocabulary"))
        for sid in sources:
            if sid not in reg:
                findings.append((ERROR, rel, f"source '{sid}' not in registry"))
    # Phase A: ids must be unique across the corpus (a duplicate breaks ID-based resolution).
    seen_ids: dict[str, list[str]] = {}
    for n in notes:
        if n.name in HUB_NAMES or not n.id:
            continue
        seen_ids.setdefault(n.id, []).append(str(n.path.relative_to(ROOT)).replace("\\", "/"))
    for nid, paths in seen_ids.items():
        if len(paths) > 1:
            for p in sorted(paths):
                findings.append((ERROR, p, f"duplicate id '{nid}' shared by {len(paths)} notes"))
    return findings


def cmd_connectivity(args):
    findings = connectivity(find_notes())
    return _report(findings, args)


def connectivity(notes):
    """Graph the wiki on stable object IDs (Phase A), not filenames, so renames never break links.

    Nodes are keyed by `_node_key` (a note's ULID, or a name-based fallback while un-stamped). Links
    resolve via the derived id/stem index: an [[id:...]] link survives a rename; a legacy [[stem]]
    link resolves through the current filename and is flagged if its stem is ambiguous.
    """
    findings = []
    by_id, by_stem = build_index_maps(notes)

    # resolve + cloud direction + collect undirected edges (keyed on node ids)
    edges = set()           # frozenset-like sorted tuples of (node_key, node_key)
    out_links = {}          # node_key -> set(node_key) of outgoing targets
    for n in notes:
        nk = _node_key(n)
        out_links.setdefault(nk, set())
        rel = _rel(n.path)
        for tgt in n.links:
            matches, kind = resolve_link(tgt, by_id, by_stem)
            if kind == "dangling":
                findings.append((ERROR, rel, f"dangling link [[{tgt}]]"))
                continue
            if kind == "ambiguous":
                names = ", ".join(sorted(m.name for m in matches))
                findings.append((WARN, rel,
                                 f"ambiguous link [[{tgt}]] resolves to {len(matches)} notes "
                                 f"({names}); disambiguate with [[id:...]]"))
            for r in matches:
                if r is n:
                    continue
                rk = _node_key(r)
                out_links[nk].add(rk)
                edges.add(tuple(sorted((nk, rk))))
                if n.cloud == "shared" and r.cloud == "personal":
                    findings.append((ERROR, rel,
                                     f"shared note links to personal [[{tgt}]] (cloud direction)"))
    # Phase C: typed frontmatter edges also contribute to reachability (so a typed-only link is not an
    # orphan). Inverses are derivable, so a single directed typed edge is "complete" — no reciprocity
    # warning. We do flag a dangling target and an explicit inverse *mismatch* when both sides carry edges.
    inverses = load_link_types()["inverses"]
    tfwd = {}               # (from_key, to_key) -> set of types authored that direction
    for n in notes:
        nk = _node_key(n)
        rel = _rel(n.path)
        for tl in n.typed_links:
            matches, kind = _resolve_ref(tl.get("to"), by_id, by_stem)
            typ = str(tl.get("type", ""))
            if kind == "dangling":
                findings.append((ERROR, rel,
                                 f"dangling typed link {{to: {tl.get('to')}, type: {typ}}}"))
                continue
            if kind == "ambiguous":
                findings.append((WARN, rel,
                                 f"ambiguous typed link to '{tl.get('to')}' ({len(matches)} notes); "
                                 f"use an id"))
            for r in matches:
                if r is n:
                    continue
                rk = _node_key(r)
                out_links[nk].add(rk)
                edges.add(tuple(sorted((nk, rk))))
                tfwd.setdefault((nk, rk), set()).add(typ)
                if n.cloud == "shared" and r.cloud == "personal":
                    findings.append((ERROR, rel,
                                     f"shared note typed-links to personal '{r.name}' (cloud direction)"))
    for (nk, rk), types_fwd in tfwd.items():
        back = tfwd.get((rk, nk))
        if back:
            for t in types_fwd:
                inv = inverses.get(t, t)
                if inv not in back:
                    src = next((x for x in notes if _node_key(x) == nk), None)
                    tgt = next((x for x in notes if _node_key(x) == rk), None)
                    if src and tgt:
                        findings.append((WARN, _rel(src.path),
                                         f"typed link '{t}' to '{tgt.name}' not matched by its inverse "
                                         f"'{inv}' (found {sorted(back)})"))
    # bidirectional check (hubs exempt as source: they list leaves by design;
    # requiring every leaf to back-link the hub is noise, not a graph problem)
    for n in notes:
        if n.name in HUB_NAMES:
            continue
        nk = _node_key(n)
        for tgt in n.links:
            matches, kind = resolve_link(tgt, by_id, by_stem)
            if kind == "dangling":
                continue
            for r in matches:
                if r is n:
                    continue
                if nk not in out_links.get(_node_key(r), set()):
                    findings.append((WARN, _rel(n.path),
                                     f"link to [[{r.name}]] is not reciprocated"))
    # reachability from hubs, per cloud
    for cloud in CLOUDS:
        cnotes = [n for n in notes if n.cloud == cloud]
        if not cnotes:
            continue
        hub_keys = [_node_key(n) for n in cnotes
                    if n.name in HUB_NAMES or n.meta.get("type") == "topic"]
        if not hub_keys:
            findings.append((WARN, cloud, "no hub (index/topic) - cannot check reachability"))
            continue
        keys = {_node_key(n) for n in cnotes}
        adj = {}
        for a, b in edges:
            if a in keys and b in keys:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        seen = set()
        stack = list(hub_keys)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, ()))
        for n in cnotes:
            if _node_key(n) not in seen and n.name not in HUB_NAMES:
                findings.append((ERROR, _rel(n.path), "orphan - not reachable from any hub"))
    return findings


def cmd_relations(args):
    """Query the typed-edge graph — relations are data, not LLM inference (Phase C)."""
    notes = find_notes()
    by_id, by_stem = build_index_maps(notes)
    lt = load_link_types()
    edges = typed_edges(notes, by_id, by_stem)
    if args.stale:
        rows = stale_notes(edges)
        for older, newer in rows:
            print(f"STALE  {_rel(older.path)}  superseded by  {_rel(newer.path)}")
        print(f"{len(rows)} stale (superseded) note(s)")
        return 0
    if args.contradictions:
        rows = contradiction_edges(edges)
        for a, b in rows:
            print(f"CONTRADICTS  {_rel(a.path)}  <->  {_rel(b.path)}")
        print(f"{len(rows)} contradiction edge(s)")
        return 0
    if args.from_ref:
        matches, kind = _resolve_ref(args.from_ref, by_id, by_stem)
        if not matches:
            print(f"no note matches '{args.from_ref}'")
            return 1
        src = matches[0]
        rels = relations_of(src, edges, lt["inverses"])
        if args.type:
            rels = [(t, o) for (t, o) in rels if t == args.type]
        for t, o in sorted(rels, key=lambda x: (x[0], x[1].name)):
            print(f"{src.name}  --{t}-->  {o.name}  ({_rel(o.path)})")
        print(f"{len(rels)} relation(s) from {src.name}"
              + (f" of type '{args.type}'" if args.type else ""))
        return 0
    if args.type:
        rows = [(a, b) for (a, b, t) in edges if t == args.type]
        for a, b in rows:
            print(f"{a.name}  --{args.type}-->  {b.name}")
        print(f"{len(rows)} '{args.type}' edge(s)")
        return 0
    # default: a summary (per-type counts + the audit surface)
    counts = {}
    for _a, _b, t in edges:
        counts[t] = counts.get(t, 0) + 1
    print(f"typed edges: {len(edges)} across {len(notes)} note(s)")
    for t in sorted(counts):
        print(f"  {t}: {counts[t]}")
    print(f"stale (superseded): {len(stale_notes(edges))}; "
          f"contradictions: {len(contradiction_edges(edges))}")
    return 0


# --------------------------------------------------------------------------- #
# embedded projection + structured query (Phase D) — the dependency-inversion test
# --------------------------------------------------------------------------- #
def cmd_project(args):
    """(Re)build the derived SQLite projection from the files. Regenerable, gitignored."""
    import projection
    path = projection.build()
    con = projection.connect_ro(path)
    try:
        meta = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta")}
    finally:
        con.close()
    print(f"projected -> {_rel(path)} (WAL)")
    print(f"  notes={meta.get('notes')} claims={meta.get('claims')} "
          f"links={meta.get('links')} sources={meta.get('sources')}")
    return 0


def cmd_query(args):
    """Run a structured query against the projection — SQL, with the LLM off."""
    import projection
    import time as _time
    if args.list:
        print("canned queries (wiki_tool query <name> [filters]):")
        for name in sorted(projection.CANNED):
            _fn, req, hlp = projection.CANNED[name]
            req_s = " ".join(f"--{r}" for r in req)
            print(f"  {name:18} {req_s:16} {hlp}")
        return 0
    t0 = _time.perf_counter()
    try:
        if args.sql:
            rows = projection.run_sql(args.sql)
        elif args.name:
            params = {"source": args.source, "from": args.from_ref, "type": args.type,
                      "of": args.of, "tag": args.tag, "note": args.note}
            rows = projection.run_canned(args.name, params, public=args.public)
        else:
            print("error: give a canned query name, --sql, or --list", file=sys.stderr)
            return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    dt = (_time.perf_counter() - t0) * 1000
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    elif not rows:
        print("(no rows)")
    else:
        cols = list(rows[0].keys())
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
    print(f"# {len(rows)} row(s) in {dt:.1f} ms")
    return 0


# --------------------------------------------------------------------------- #
# CoDIAK dialog record (Phase E) — capture the reasoning, don't let it evaporate
# --------------------------------------------------------------------------- #
def cmd_dialog(args):
    """Append to / read / reconstruct the append-only dialog record (Schema/dialog.jsonl)."""
    import dialog as dlg
    if args.action == "add":
        if not args.kind or not args.text:
            print("error: dialog add requires --kind and --text", file=sys.stderr)
            return 2
        links = [x.strip() for x in (args.links or "").split(",") if x.strip()]
        try:
            eid = dlg.append_event(args.kind, args.text, links=links, author=args.author)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(eid)
        return 0
    if args.action == "log":
        events = dlg.load_dialog()
        if args.kind:
            events = [e for e in events if e.get("kind") == args.kind]
        for e in events[-args.limit:]:
            print(f"{e.get('ts','')}  {e.get('kind',''):14}  {e.get('id','')}  {e.get('text','')[:80]}")
            if e.get("links"):
                print(f"{'':38}-> {', '.join(e['links'])}")
        print(f"{len(events)} dialog event(s)")
        return 0
    if args.action == "trail":
        if not args.ref:
            print("error: dialog trail requires an id", file=sys.stderr)
            return 2
        trail = dlg.dialog_trail(dlg.load_dialog(), args.ref)
        for e in trail:
            print(f"{e.get('ts','')}  {e.get('kind',''):14}  {e.get('id','')}  {e.get('text','')}")
            if e.get("links"):
                print(f"{'':38}-> {', '.join(e['links'])}")
        print(f"{len(trail)} event(s) in the trail for {args.ref}")
        return 0
    return 0


# --------------------------------------------------------------------------- #
# bootstrapping loop (Phase G) — co-evolve the tool-system; propose, a human disposes
# --------------------------------------------------------------------------- #
def cmd_bootstrap(args):
    import bootstrap as bs
    if args.action == "propose":
        signals = bs.gather_signals(floor=args.floor)
        proposals = bs.propose_from_signals(signals)
        if not proposals:
            print("no tool-system underperformance signals detected; no proposals.")
            return 0
        ids = bs.emit_proposals(proposals)
        for pid, p in zip(ids, proposals):
            print(f"PROPOSAL {pid}  [{p['target']}]")
            print(f"  {p['text']}")
        print(f"\n{len(proposals)} evidence-backed proposal(s) emitted as schema-change events in the "
              f"dialog record.\nReview: `dialog log --kind schema-change`  |  "
              f"dispose: `bootstrap accept|reject <id>`")
        return 0
    if args.action in ("accept", "reject"):
        if not args.ref:
            print(f"error: bootstrap {args.action} requires a proposal id", file=sys.stderr)
            return 2
        v = bs.dispose(args.ref, accept=(args.action == "accept"), note=args.note or "")
        print(f"{args.action}ed proposal {args.ref}; tool-system version is now {v['version']} "
              f"(logged to the dialog record)")
        return 0
    if args.action == "status":
        v = bs.load_tool_version()
        print(f"tool-system version: {v['version']}")
        print(f"  accepted: {len(v.get('accepted', []))}   rejected: {len(v.get('rejected', []))}")
        for a in v.get("accepted", []):
            print(f"    v{a['version']} <- {a['proposal']}")
        return 0
    return 0


# --------------------------------------------------------------------------- #
# view control (Phase H) — generated views over the projection; LLM off, no Obsidian
# --------------------------------------------------------------------------- #
def cmd_view(args):
    import views
    if args.list:
        print("views (wiki_tool view <name> [--of <id>] [--html] [--public] [--stdout]):")
        for name in sorted(views.VIEWS):
            _fn, req, hlp = views.VIEWS[name]
            print(f"  {name:22} {' '.join('--' + r for r in req):10} {hlp}")
        return 0
    if not args.name:
        print("error: give a view name or --list", file=sys.stderr)
        return 2
    fmt = "html" if args.html else "md"
    try:
        if args.stdout:
            print(views.render(args.name, params={"of": args.of}, public=args.public, fmt=fmt))
        else:
            p = views.render_to_file(args.name, params={"of": args.of}, public=args.public, fmt=fmt)
            print(f"rendered -> {_rel(p)}")
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


# --------------------------------------------------------------------------- #
# claim-level provenance (Phase B) — verify each claim's locator against its source
# --------------------------------------------------------------------------- #
_SOURCE_TEXT_CACHE: dict = {}


def get_source_text(source_id, reg, *, use_cache=True):
    """Return the full extracted text of a registered source on this machine, or None.

    Reads .md/.txt directly (stdlib); routes other formats through the optional extractors and returns
    None if the dependency is absent or the file is missing on this machine. None => the locator can't
    be verified here (degrade to a warning), it is NOT proof the claim is wrong.
    """
    if use_cache and source_id in _SOURCE_TEXT_CACHE:
        return _SOURCE_TEXT_CACHE[source_id]
    entry = reg.get(source_id)
    text = None
    if entry:
        loc = entry.get("locations", {}).get(machine_id()) or {}
        path = loc.get("path")
        if path:
            p = Path(path) if Path(path).is_absolute() else (ROOT / path)
            if p.exists():
                suffix = p.suffix.lower()
                if suffix in (".md", ".markdown", ".txt", ".csv", ".tsv", ".log",
                              ".json", ".yaml", ".yml", ".rst", ""):
                    try:
                        text = p.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        text = None
                else:
                    text = _extract_source_text(p)
    if use_cache:
        _SOURCE_TEXT_CACHE[source_id] = text
    return text


def _extract_source_text(path):
    """Feature-detected extraction of a non-text source to plain text. None if extractors unavailable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extractors"))
        import extract as _ex          # extractors/extract.py
        from base import ExtractorUnavailable  # noqa
    except Exception:
        return None
    try:
        result = _ex.extract(str(path))
    except Exception:
        return None
    return "\n\n".join(s.text for s in result.segments)


def lint_claims(notes, reg):
    """Lint every note's claim sidecar: structure + locator resolution against the source.

    Findings (ERROR blocks under --strict, warns otherwise; matches validate/connectivity):
      ERROR  structural problem, unregistered source, or a quote NOT found in the source (fabrication).
      WARN   weak/unverifiable locator, or source missing so the lifted quote can't be re-checked.
    Returns the standard [(level, path, msg)] findings list.
    """
    import claims as _claims
    findings = []
    _SOURCE_TEXT_CACHE.clear()
    seen_claim_ids: dict = {}
    for n in notes:
        cpath = _claims.claims_path(n.path)
        if not cpath.exists():
            continue
        rel = _rel(cpath)
        note_sources = set(n.as_list("sources"))
        for claim in _claims.load_claims(n.path):
            cid = claim.get("id", "?")
            seen_claim_ids.setdefault(str(cid), []).append(rel)
            for problem in _claims.validate_claim_structure(claim):
                findings.append((ERROR, rel, problem))
            sid = str(claim.get("source_id", ""))
            if not sid:
                continue
            if sid not in reg:
                findings.append((ERROR, rel, f"claim '{cid}' source '{sid}' not in registry"))
                continue
            if sid not in note_sources:
                findings.append((WARN, rel,
                                 f"claim '{cid}' cites '{sid}' not in the note's frontmatter sources"))
            src_text = get_source_text(sid, reg)
            if src_text is None:
                findings.append((WARN, rel,
                                 f"claim '{cid}' locator unverified (source '{sid}' missing/unreadable here)"))
                continue
            status, reason = _claims.locator_resolves(claim, src_text)
            if status == "fail":
                findings.append((ERROR, rel, f"claim '{cid}': {reason}"))
            elif status == "unverified":
                findings.append((WARN, rel, f"claim '{cid}': {reason}"))
    for cid, paths in seen_claim_ids.items():
        if len(paths) > 1:
            for p in sorted(set(paths)):
                findings.append((ERROR, p, f"duplicate claim id '{cid}' across sidecars"))
    return findings


def cmd_claims_lint(args):
    findings = lint_claims(find_notes(), load_registry())
    n_claims = sum(len(_load_claims_for(n)) for n in find_notes())
    print(f"# claims: {n_claims} across {sum(1 for n in find_notes() if _load_claims_for(n))} note(s)")
    return _report(findings, args)


def _load_claims_for(note):
    import claims as _claims
    return _claims.load_claims(note.path)


def cmd_claim_add(args):
    """Ground a selected quote in a source and record a claim to a note's sidecar (deterministic).

    Refuses if the quote is not present in the source — the LLM supplies text+quote+source; the code
    verifies the provenance is real. This is the division of labour at the CLI boundary.
    """
    import claims as _claims
    note_path = ROOT / args.note if not Path(args.note).is_absolute() else Path(args.note)
    if not note_path.exists():
        print(f"error: note not found: {args.note}", file=sys.stderr)
        return 2
    reg = load_registry()
    if args.source not in reg:
        print(f"error: source '{args.source}' not in registry", file=sys.stderr)
        return 2
    src_text = get_source_text(args.source, reg)
    if src_text is None:
        print(f"error: cannot read source '{args.source}' on this machine to ground the quote",
              file=sys.stderr)
        return 2
    frag = _claims.ground_quote(src_text, args.quote)
    if frag is None:
        print("error: quote not found in source — refusing to assert ungrounded provenance",
              file=sys.stderr)
        return 1
    claim = {"id": new_ulid(), "text": args.text, "source_id": args.source, "locator": frag,
             "asserted_by": args.asserted_by, "verified_by": "none", "created": _today()}
    if args.confidence is not None:
        claim["confidence"] = args.confidence
    _claims.append_claim(note_path, claim)
    print(f"{claim['id']}  -> {_rel(_claims.claims_path(note_path))}")
    return 0


def cmd_verify(args):
    """Promote claims from asserted -> verified (Phase F). Human judgement, or a cross-source rule.

    Never auto-promotes: a model-asserted claim only becomes verified through an explicit `verify`.
    A `cross-source` promotion is allowed only when an independent claim from another source corroborates
    it (deterministic). Every promotion is logged to the CoDIAK dialog record.
    """
    import claims as _claims
    import dialog as _dlg
    by = args.by
    if by not in ("human", "cross-source"):
        print("error: --by must be 'human' or 'cross-source'", file=sys.stderr)
        return 2
    notes = find_notes()
    all_claims = [c for n in notes for c in _claims.load_claims(n.path)]
    # resolve the target claim(s): a single --claim id, or every claim in --note
    targets = []   # (note, claim)
    if args.claim:
        for n in notes:
            for c in _claims.load_claims(n.path):
                if c.get("id") == args.claim:
                    targets.append((n, c))
        if not targets:
            print(f"error: claim {args.claim} not found", file=sys.stderr)
            return 2
    elif args.note:
        n = next((x for x in notes if _rel(x.path) == args.note.replace("\\", "/")), None)
        if not n:
            print(f"error: note {args.note} not found", file=sys.stderr)
            return 2
        targets = [(n, c) for c in _claims.load_claims(n.path)]
        if not targets:
            print(f"no claims in {args.note}")
            return 0
    else:
        print("error: give --claim <id> or --note <path>", file=sys.stderr)
        return 2
    done = 0
    for n, claim in targets:
        cid = claim.get("id")
        if by == "cross-source":
            corro = _claims.corroborating_claims(claim, all_claims)
            if not corro:
                print(f"refused {cid}: no independent source corroborates it (cannot cross-verify)")
                continue
            note_txt = f"verified claim {cid} by cross-source corroboration ({corro[0].get('source_id')})"
            links = [cid, corro[0].get("id")]
            author = "cross-source"
        else:
            note_txt = f"verified claim {cid} by human review"
            links = [cid]
            author = "human"
        if _claims.set_claim_verified(n.path, cid, by):
            _dlg.append_event("decision", note_txt, links=links, author=author)
            print(f"verified  {cid}  as {by}  ({_rel(n.path)})")
            done += 1
    if args.sign:
        print("  note: GPG signing is a business-tier feature and is OFF by default — not signed. "
              "(Enable explicitly per Schema/agency.md; not built in the core kit.)")
    print(f"{done} claim(s) verified as {by}; each promotion logged to the dialog record")
    return 0


# --------------------------------------------------------------------------- #
# BM25 search
# --------------------------------------------------------------------------- #
def _tokens(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rank(query, docs, k1=1.5, b=0.75):
    """docs: list of (id, tokens). Returns list of (id, score) desc."""
    N = len(docs)
    if N == 0:
        return []
    df = {}
    for _, toks in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(toks) for _, toks in docs) / N
    qtoks = _tokens(query)
    scores = []
    for did, toks in docs:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks)
        s = 0.0
        for qt in qtoks:
            if qt not in tf:
                continue
            idf = math.log(1 + (N - df[qt] + 0.5) / (df[qt] + 0.5))
            s += idf * (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / avgdl))
        if s > 0:
            scores.append((did, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def cmd_search(args):
    notes = find_notes()
    docs = []
    meta = {}
    text_map = {}
    for n in notes:
        rel = str(n.path.relative_to(ROOT)).replace("\\", "/")
        blob = (n.title + " " + " ".join(n.as_list("aliases")) + " " +
                " ".join(n.as_list("tags")) + " " + n.body)
        docs.append((rel, _tokens(blob)))
        meta[rel] = n.title
        text_map[rel] = blob
    ranked = bm25_rank(args.query, docs)
    tier = "bm25"
    if getattr(args, "semantic", False):
        ranked, tier = _semantic_rerank(args.query, ranked, text_map, args.limit)
    ranked = ranked[: args.limit]
    if not ranked:
        print("no matches")
        return 0
    print(f"# tier: {tier}")
    for did, score in ranked:
        print(f"{score:6.2f}  {did}  - {meta.get(did, '')}")
    return 0


def _semantic_rerank(query, bm25_ranked, text_map, limit):
    """Hybrid: BM25 candidates reranked by vector cosine. Falls back to BM25 if tier absent."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import vectorstore as vs
    except Exception:
        return bm25_ranked, "bm25 (vector tier unavailable)"
    cand = [d for d, _ in bm25_ranked[: max(limit * 3, limit)]] or list(text_map)
    store = vs.VectorStore(ROOT / ".runtime" / "search", embedder=vs.get_embedder())
    store.update([(c, text_map.get(c, "")) for c in cand])
    return store.rerank(query, cand), f"hybrid ({store.embedder.name})"


def coverage_map():
    """source_id -> list of note paths citing it (i.e. the source has been turned into knowledge)."""
    cov = {}
    for n in find_notes():
        for sid in n.as_list("sources"):
            cov.setdefault(sid, []).append(str(n.path.relative_to(ROOT)).replace("\\", "/"))
    return cov


def _snapshot_static(inbox, sub, src_abs):
    """Copy a dated snapshot of a static (append-only) source into inbox*/processed/static/, but only
    when its content changed since the last snapshot. Returns the snapshot Path, or None if unchanged."""
    import shutil
    stem, ext = Path(sub).stem, Path(sub).suffix
    snap_dir = ROOT / inbox / "processed" / "static"
    snap_dir.mkdir(parents=True, exist_ok=True)
    current = src_abs.read_bytes()
    existing = sorted(snap_dir.glob(f"{stem}-*{ext}"))
    if existing and existing[-1].read_bytes() == current:
        return None  # unchanged since the last snapshot - don't churn
    snap = snap_dir / f"{stem}-{_today()}{ext}"
    n = 2
    while snap.exists() and snap.read_bytes() != current:
        snap = snap_dir / f"{stem}-{_today()}-{n}{ext}"
        n += 1
    shutil.copy2(src_abs, snap)
    return snap


def process_inbox():
    """Resolve dropped inbox sources that >=1 note now cites.

    - `inbox*/pending/Files/<f>`: a one-shot drop - MOVED to `inbox*/processed/<f>`; registry updated.
    - `inbox*/static/<f>`: an append-only source (e.g. running meeting notes) - left in place; a dated
      snapshot is copied to `inbox*/processed/static/<stem>-<date><ext>` when its content changed, so
      you keep editing the original and still have a per-ingest record to diff against. The registry
      keeps pointing at the live static file.
    - `Clippings/<f>`: an Obsidian Web Clipper drop in the root `Clippings/` folder - MOVED to
      `inbox/processed/Clippings/<f>` (a normal one-shot drop).
    processed/ is gitignored. Returns [(source_id, new_rel_path)] for whatever was moved/snapshotted.
    """
    reg = load_registry()
    cov = coverage_map()
    mid = machine_id()
    done, changed = [], False
    for sid, entry in reg.items():
        if sid not in cov:
            continue
        loc = entry.get("locations", {}).get(mid)
        if not loc or "path" not in loc:
            continue
        norm = str(loc["path"]).replace("\\", "/")
        src_abs = Path(norm) if Path(norm).is_absolute() else (ROOT / norm)
        # web-clipper drops land in the root Clippings/ folder - treat like a pending drop (move it)
        try:
            rel_parts = src_abs.resolve().relative_to(ROOT.resolve()).parts
        except (ValueError, OSError):
            rel_parts = ()
        if rel_parts and rel_parts[0] == "Clippings" and src_abs.exists():
            sub = "/".join(rel_parts[1:])
            dest = ROOT / "inbox" / "processed" / "Clippings" / sub
            dest.parent.mkdir(parents=True, exist_ok=True)
            src_abs.replace(dest)
            loc["path"] = f"inbox/processed/Clippings/{sub}"
            loc["present"], loc["last_seen"] = True, _today()
            done.append((sid, loc["path"]))
            changed = True
            continue
        for inbox in ("inbox", "inbox_personal"):
            smarker = f"{inbox}/static/"
            sidx = norm.find(smarker)
            if sidx != -1:                                    # static: snapshot, never move
                if not src_abs.exists():
                    break
                snap = _snapshot_static(inbox, norm[sidx + len(smarker):], src_abs)
                loc["present"], loc["last_seen"] = True, _today()
                changed = True
                if snap is not None:
                    done.append((sid, str(snap.relative_to(ROOT)).replace("\\", "/")))
                break
            marker = f"{inbox}/pending/Files/"
            idx = norm.find(marker)
            if idx == -1:
                continue
            if not src_abs.exists():                          # pending: move
                break
            sub = norm[idx + len(marker):]
            dest = ROOT / inbox / "processed" / sub
            dest.parent.mkdir(parents=True, exist_ok=True)
            src_abs.replace(dest)
            loc["path"] = f"{inbox}/processed/{sub}"
            loc["present"], loc["last_seen"] = True, _today()
            done.append((sid, loc["path"]))
            changed = True
            break
    if changed:
        save_registry(reg)
    return done


def cmd_process_inbox(args):
    done = process_inbox()
    for sid, p in done:
        print(f"{sid} -> {p}")
    print(f"{len(done)} inbox file(s) processed (pending moved; static snapshotted)")
    return 0


def cmd_source_check(args):
    reg = load_registry()
    mid = machine_id()
    today = _today()
    changed = 0
    for e in reg.values():
        loc = e.get("locations", {}).get(mid)
        if not loc:
            continue
        path = loc.get("path")
        if path is None:
            continue  # url-only, skip filesystem check
        present = Path(path).exists()
        if present != loc.get("present"):
            changed += 1
        loc["present"] = present
        if present:
            loc["last_seen"] = today
    save_registry(reg)
    dangling = [e["id"] for e in reg.values()
                if not any(v.get("present") for v in e.get("locations", {}).values())]
    print(f"checked {len(reg)} sources on {mid}; updated {changed}; dangling: {len(dangling)}")
    for sid in dangling:
        print(f"  dangling: {sid}  {reg[sid].get('title','')}")
    return 0


def cmd_ingest_log(args):
    events = load_ingest_log()
    if args.summary:
        regs = [e for e in events if e.get("event") == "register"]
        fins = [e for e in events if e.get("event") == "finalize"]
        srcs = {e["id"] for e in regs if "id" in e}
        print(f"ingest-log: {len(events)} event(s) - {len(regs)} register "
              f"({len(srcs)} distinct source(s)), {len(fins)} finalize")
        return 0
    for e in events[-args.limit:]:
        if e.get("event") == "register":
            state = "present" if e.get("present") else "missing"
            print(f"{e.get('ts','')}  register  {e.get('id','')}  [{state}]  "
                  f"{e.get('title','')}  <- {e.get('location','')}")
        else:
            print(f"{e.get('ts','')}  {e.get('event','')}  {e.get('message','')}")
    return 0


def cmd_log(args):
    cloud = "personal" if args.personal else "shared"
    path = ROOT / cloud / "log.md"
    if not path.exists():
        path.write_text(f"# {cloud} log\n\n", encoding="utf-8")
    entry = f"\n## [{_today()}] {args.title}\n\n{args.details or ''}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(f"logged to {path.relative_to(ROOT)}")
    return 0


def cmd_scan(args):
    cloud = "inbox_personal" if args.personal else "inbox"
    urls_doc = ROOT / cloud / "pending" / "urls.md"
    if not urls_doc.exists():
        print("no urls doc")
        return 0
    pending = []
    for line in urls_doc.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s+(https?://\S+)\s*$", line)
        if m:
            pending.append(m.group(1))
    print(f"{len(pending)} pending url(s) in {urls_doc.relative_to(ROOT)}")
    for u in pending:
        print(f"  {u}")
    if pending:
        print("(fetch+ingest wired in Phase 4; URLs are listed, not yet retrieved)")
    return 0


def _is_malformed(path):
    """Control chars, a colon, or private-use-area glyphs (e.g. the mangled C:\\tmp\\fin.txt)."""
    return any(ord(c) < 32 or c == ":" or 0xE000 <= ord(c) <= 0xF8FF for c in path)


def classify_change(path):
    """ok | surprise | junk - should finalize auto-commit this changed path?"""
    p = path.replace("\\", "/").strip()
    if _is_malformed(p):
        return "junk"
    parts = [x for x in p.split("/") if x]
    if not parts:
        return "surprise"
    if len(parts) == 1:
        name = parts[0]
        return "ok" if (name in ALLOWED_ROOT_FILES or name.endswith(".md")) else "surprise"
    return "ok" if parts[0] in ALLOWED_TOP_DIRS else "surprise"


def parse_porcelain_z(text):
    """Parse `git status --porcelain -z` -> [(status, path)]. Handles rename/copy source tokens."""
    tokens = text.split("\0")
    entries, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if not t:
            i += 1
            continue
        status, path = t[:2], t[3:]
        if "R" in status or "C" in status:
            i += 1  # consume the rename/copy source token that follows
        entries.append((status, path))
        i += 1
    return entries


def _repo_prefix():
    """Path of ROOT relative to the git repo root (e.g. 'wiki/'); '' when ROOT *is* the repo root.
    rl-betfair vendors this wiki as a subdir of a larger repo. `git status --porcelain` reports paths
    relative to the repo root, so finalize's junk-guard (classify_change) and `git add` must work in
    ROOT-relative terms — strip this prefix. No-op (byte-identical to upstream) when wiki == repo root."""
    r = _git(["rev-parse", "--show-prefix"], check=False)
    if r is None or r.returncode != 0:
        return ""
    return (r.stdout or "").strip()


def git_changes():
    # scope to ROOT's subtree ('.' from cwd=ROOT) so a parent repo's unrelated changes are ignored.
    r = _git(["status", "--porcelain", "-z", "--untracked-files=all", "--", "."], check=False)
    if r is None or r.returncode != 0:
        return []
    entries = parse_porcelain_z(r.stdout or "")
    prefix = _repo_prefix()  # '' in the upstream (wiki == repo root) layout -> no change
    if prefix:
        entries = [(st, p[len(prefix):] if p.startswith(prefix) else p) for st, p in entries]
    return entries


def cmd_finalize(args):
    rc = 0
    # stamp any new notes first so the graph + catalog are built on stable ids (Phase A).
    # Additive only — never auto-relinks (that stays an explicit `stamp-ids --relink`).
    print("== stamp-ids ==")
    cmd_stamp_ids(argparse.Namespace(relink=False, dry_run=getattr(args, "dry_run", False)))
    print("== build =="); cmd_build(args)
    print("== validate ==")
    vf = validate(find_notes(), load_vocab(), load_registry())
    _print_findings(vf)
    print("== connectivity ==")
    cf = connectivity(find_notes())
    _print_findings(cf)
    print("== claims-lint ==")
    clf = lint_claims(find_notes(), load_registry())
    _print_findings(clf)
    print("== coverage ==")          # substance gate: surfaces page-padding / thin stubs (advisory)
    covf, _ = coverage(find_notes(), load_registry())
    _print_findings(covf)
    print("== audit_public ==")
    af = _run_audit()
    _print_findings(af)
    errors = [f for f in vf + cf + clf + af if f[0] == ERROR]
    if errors and args.strict:
        print(f"FAIL (strict): {len(errors)} error(s); not committing.")
        return 1
    if errors:
        print(f"warning: {len(errors)} error-level finding(s) (non-strict; continuing).")
    if args.dry_run:
        print("dry-run: skipping git.")
        return rc
    done = process_inbox()
    if done:
        print(f"inbox: processed {len(done)} file(s) (pending moved; static snapshotted)")
    msg = args.message or f"ingest {_today()}"
    # append the finalize event BEFORE committing so it lands in the commit (replication log)
    append_ingest_log({"event": "finalize", "message": msg, "notes": len(find_notes())})
    # junk guard: never blanket `git add -A`. Auto-stage expected paths; hold surprises/junk back.
    ok_paths, surprises = [], []
    for _status, path in git_changes():
        kind = classify_change(path)
        (ok_paths if kind == "ok" else surprises).append((kind, path))
    if surprises:
        print("guard: unexpected paths held back (NOT committed) - review them:")
        for kind, p in surprises:
            print(f"  [{kind}] {p}")
        if args.strict:
            print("FAIL (strict): unexpected paths present; aborting.")
            return 1
    if ok_paths:
        _git(["add", "--"] + [p for _, p in ok_paths])
        r = _git(["commit", "-m", msg], check=False)
        if r is not None and r.returncode != 0:
            print("git commit: nothing to commit or commit failed.")
    else:
        print("nothing expected to commit.")
    if args.push:
        _git(["push"], check=False)
    # refresh the derived projection so query/MCP/views are current (gitignored; never load-bearing)
    print("== project ==")
    try:
        import projection
        projection.build()
        print("  projection refreshed")
    except Exception as e:
        print(f"  projection refresh skipped ({e})")
    return rc


def _run_audit():
    try:
        from audit_public import audit_findings  # type: ignore
        return audit_findings(ROOT)
    except Exception:
        # run as subprocess fallback
        ap = ROOT / "scripts" / "audit_public.py"
        if not ap.exists():
            return []
        r = subprocess.run([sys.executable, str(ap), "--json"], capture_output=True, text=True)
        try:
            return [(d["level"], d["path"], d["msg"]) for d in json.loads(r.stdout or "[]")]
        except Exception:
            return []


def _git(args_list, check=True):
    try:
        return subprocess.run(["git"] + args_list, cwd=str(ROOT),
                              capture_output=True, text=True)
    except FileNotFoundError:
        if check:
            print("git not found", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def _print_findings(findings):
    for level, path, msg in findings:
        print(f"  [{level}] {path}: {msg}")
    if not findings:
        print("  clean")


def _report(findings, args):
    _print_findings(findings)
    errors = [f for f in findings if f[0] == ERROR]
    if errors and getattr(args, "strict", False):
        return 1
    return 0


# --------------------------------------------------------------------------- #
# coverage / substance gate (Phase G follow-up) — catch under-extraction AND metric-gaming
# --------------------------------------------------------------------------- #
def _note_substance(note) -> int:
    """Approx chars of real prose: the body minus the H1 title and the generated `## Sources` block."""
    body = re.sub(r"^#\s+.*$", "", note.body, count=1, flags=re.M)
    body = re.sub(r"##\s*Sources\b.*", "", body, flags=re.S)
    return len(re.sub(r"\s+", " ", body).strip())


def coverage(notes, reg, min_chars=200):
    """Substance-aware per-source extraction quality. Returns (findings, stats).

    `notes/page` is gameable: page-padding (a note per page, even cover/ToC/index) hits it without
    capturing knowledge, and it can't tell 69 concept notes from 78 page-dumps. This flags the two real
    failure modes deterministically (no LLM): **padding** (page-named notes, or a high share of thin
    stubs) — and, advisory, thin substance. The cure for under-extraction is concept/objective/keyword
    extraction (see skills/extract); this gate makes the gaming visible so it can't pass silently.
    """
    by_src = {}
    for n in notes:
        if n.name in HUB_NAMES:
            continue
        for sid in n.as_list("sources"):
            by_src.setdefault(str(sid), []).append(n)
    findings, stats = [], []
    for sid, ns in sorted(by_src.items()):
        total = len(ns)
        thin = sum(1 for n in ns if _note_substance(n) < min_chars)
        paged = sum(1 for n in ns
                    if PAGE_NAME_RE.search(n.name) or PAGE_NAME_RE.search(str(n.title)))
        title = (reg.get(sid) or {}).get("title", sid)
        stats.append({"source": sid, "title": title, "notes": total,
                      "substantive": total - thin, "thin": thin, "page_named": paged})
        where = f"source:{sid}"
        if paged:
            findings.append((WARN, where,
                             f"{paged}/{total} notes citing '{title}' are page-named (e.g. p12) — "
                             f"looks like page-padding, not concept/objective extraction"))
        if total >= 3 and thin / total > 0.3:
            findings.append((WARN, where,
                             f"{thin}/{total} notes citing '{title}' are thin (<{min_chars} chars) — "
                             f"possible stub-padding"))
    return findings, stats


def cmd_coverage(args):
    notes, reg = find_notes(), load_registry()
    findings, stats = coverage(notes, reg, min_chars=args.min_chars)
    if args.source:
        stats = [s for s in stats if s["source"] == args.source]
        findings = [f for f in findings if args.source in f[1]]
    if args.json:
        print(json.dumps({"stats": stats,
                          "findings": [{"level": l, "where": w, "msg": m} for l, w, m in findings]},
                         indent=2, ensure_ascii=False))
    else:
        print(f"{'source':16}{'notes':>6}{'subst':>6}{'thin':>5}{'page':>5}  title")
        for s in sorted(stats, key=lambda x: (-x["page_named"], -x["thin"])):
            print(f"{s['source']:16}{s['notes']:6}{s['substantive']:6}{s['thin']:5}{s['page_named']:5}"
                  f"  {str(s['title'])[:46]}")
        print()
        _print_findings(findings)
    extra_fail = False
    if getattr(args, "objectives", False):
        extra_fail = _objective_coverage_report(notes, reg, only=args.source)
    return 1 if (getattr(args, "strict", False) and (findings or extra_fail)) else 0


def _objective_coverage_report(notes, reg, only=None) -> bool:
    """Objective+keyword coverage for reference-type sources (the syllabus rule). Returns True if gaps."""
    import doctypes as dt
    by_src = {}
    for n in notes:
        if n.name in HUB_NAMES:
            continue
        for sid in n.as_list("sources"):
            by_src.setdefault(str(sid), []).append(n)
    gaps = False
    print("== objective/keyword coverage (reference docs) ==")
    any_ref = False
    for sid, ns in sorted(by_src.items()):
        if only and sid != only:
            continue
        e = reg.get(sid) or {}
        if dt.classify(e.get("title", "")) != "reference":
            continue
        any_ref = True
        txt = get_source_text(sid, reg)
        if not txt:
            print(f"  {sid}  {str(e.get('title',''))[:42]}: source unreadable here — skipped")
            continue
        cov = dt.reference_coverage(txt, ns)
        o, k = cov["objectives"], cov["keywords"]
        org = f"  [{cov['page_notes']}/{cov['notes']} are PAGE notes, not concepts]" if cov["page_notes"] else ""
        print(f"  {sid}  {str(e.get('title',''))[:42]}: "
              f"objectives {o['covered']}/{o['total']}, keywords {k['covered']}/{k['total']}{org}")
        if o["missing"]:
            print(f"      missing objectives: {', '.join(o['missing'][:15])}")
        if k["missing"]:
            print(f"      missing keywords:   {', '.join(k['missing'][:15])}")
        if o["missing"] or k["missing"]:
            gaps = True
    if not any_ref:
        print("  (no reference/syllabus sources detected)")
    return gaps


def cmd_classify(args):
    """Show the detected doc-type per source (drives type-specific extraction + coverage)."""
    import doctypes as dt
    reg = load_registry()
    counts = {}
    for sid, e in sorted(reg.items()):
        if args.source and sid != args.source:
            continue
        title = str(e.get("title", ""))
        typ = dt.classify(title, get_source_text(sid, reg) if args.deep else "")
        counts[typ] = counts.get(typ, 0) + 1
        print(f"{sid:16}{typ:15}{title[:48]}")
    print("\nby type:", ", ".join(f"{t}={c}" for t, c in sorted(counts.items())) or "(no sources)")
    print("extraction guidance per type: Schema/doc-types.md  (skill: skills/curate/SKILL.md)")
    return 0


# --------------------------------------------------------------------------- #
# rollback — undo an ingestion via the git history finalize already creates
# --------------------------------------------------------------------------- #
def _note_files_changed(commit) -> int:
    r = _git(["show", "--name-only", "--pretty=format:", commit], check=False)
    if not r:
        return 0
    return sum(1 for l in (r.stdout or "").splitlines()
               if (l.startswith("shared/") or l.startswith("personal/")) and l.endswith(".md"))


def _rollback_source(args):
    """Roll back ONE source's ingestion (not a whole commit): delete the notes that cite *only* this
    source, plus their claim sidecars. Notes that also cite other sources are kept (they still cite it).

    Why file-based, not commit-based: a `finalize-ingest` commit mixes several docs, so `--to <hash>`
    can't isolate one source. This is the right tool for "roll this doc back out and re-ingest it".
    `--dry-run` previews. The source file itself is left in place so you can re-stage a slice."""
    reg = load_registry()
    sid = args.source
    e = reg.get(sid)
    if not e:
        print(f"error: source '{sid}' is not in the registry (see `register --list` / catalog)",
              file=sys.stderr)
        return 2
    title = e.get("title", sid)
    import claims as _claims
    citing = [n for n in find_notes() if sid in n.as_list("sources")]
    only = [n for n in citing if set(n.as_list("sources")) == {sid}]
    multi = [n for n in citing if len(set(n.as_list("sources"))) > 1]
    print(f"rollback source {sid}  '{title}'")
    print(f"  {len(citing)} citing note(s): {len(only)} cite ONLY this source -> delete; "
          f"{len(multi)} multi-source -> kept")
    if args.dry_run:
        for n in sorted(only, key=lambda x: x.name)[:30]:
            extra = "  (+claims)" if _claims.claims_path(n.path).exists() else ""
            print(f"   would delete  {_rel(n.path)}{extra}")
        if len(only) > 30:
            print(f"   ... +{len(only) - 30} more")
        for n in sorted(multi, key=lambda x: x.name)[:10]:
            print(f"   would keep    {_rel(n.path)}  sources={n.as_list('sources')}")
        print("  (dry run — nothing changed)")
        return 0
    deleted = 0
    for n in only:
        cp = _claims.claims_path(n.path)
        if cp.exists():
            cp.unlink()
        try:
            n.path.unlink()
            deleted += 1
        except OSError as ex:
            print(f"  could not delete {_rel(n.path)}: {ex}", file=sys.stderr)
    if multi:
        regmsg = f"kept {sid} registered ({len(multi)} multi-source note(s) still cite it)"
    else:
        reg.pop(sid, None)
        save_registry(reg)
        regmsg = f"unregistered {sid}"
    try:
        import dialog as _dlg
        _dlg.append_event("rejection",
                          f"rolled back source {sid} ('{title}'): deleted {deleted} single-source "
                          f"note(s) for re-ingestion", links=[sid], author="human")
    except Exception:
        pass
    try:
        import projection
        projection.build()
        proj = "; projection rebuilt"
    except Exception as ex:
        proj = f"; projection rebuild skipped ({ex})"
    print(f"  deleted {deleted} note(s) (+claim sidecars); {regmsg}{proj}")
    print("  source file left in place. Re-stage a slice (or the whole doc) to re-ingest; run "
          "`connectivity` after to spot any links the removed notes left dangling.")
    return 0


def cmd_rollback(args):
    """Undo an ingestion. `--source <id>` removes one doc's notes (file-based; mixed commits can't isolate
    a source). Otherwise each `finalize-ingest` is one git commit, so `--last`/`--to` revert/reset it —
    atomic and regenerable (projection rebuilt after)."""
    if getattr(args, "source", None):
        return _rollback_source(args)
    r = _git(["rev-parse", "--is-inside-work-tree"], check=False)
    if not r or r.returncode != 0 or (r.stdout or "").strip() != "true":
        print("error: not a git repo — rollback needs the commit history", file=sys.stderr)
        return 2
    if args.list:
        log = _git(["log", "-20", "--pretty=format:%h\t%ad\t%s", "--date=short"], check=False)
        print("recent commits ('*' = added/changed notes; undo with `rollback --to <hash>`):")
        for line in (log.stdout or "").splitlines():
            h = line.split("\t", 1)[0]
            n = _note_files_changed(h)
            print(f"  {'*' if n else ' '} {line}   ({n} note file(s))")
        return 0
    target = "HEAD" if args.last else args.to
    if not target:
        print("error: give --last, or --to <hash> (or --list to choose)", file=sys.stderr)
        return 2
    dirty = bool((_git(["status", "--porcelain"], check=False).stdout or "").strip())
    if dirty and not args.force:
        print("refusing: the working tree has uncommitted changes (an ingest may be in progress). "
              "Commit/stash it first, or pass --force.", file=sys.stderr)
        return 1
    if args.hard:
        ref = (target + "~1") if args.last else target
        print(f"rollback (HARD): git reset --hard {ref}  — discards everything after it")
        res = _git(["reset", "--hard", ref], check=False)
    else:
        print(f"rollback: git revert --no-edit {target}  — inverse commit; history preserved")
        res = _git(["revert", "--no-edit", target], check=False)
    if res is None or res.returncode != 0:
        print(f"git failed: {(res.stderr.strip() if res else 'git not found')}", file=sys.stderr)
        print("(a revert conflict means later commits touched the same notes — resolve it, or use "
              "`rollback --to <hash> --hard` to reset to a clean point.)", file=sys.stderr)
        return 1
    try:
        import projection
        projection.build()
        print("projection rebuilt from the rolled-back files.")
    except Exception as e:
        print(f"projection rebuild skipped: {e}")
    print("rollback complete.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(prog="wiki_tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init"); s.add_argument("--machine"); s.set_defaults(func=cmd_init)
    s = sub.add_parser("doctor"); s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("register")
    s.add_argument("--path"); s.add_argument("--url")
    s.add_argument("--title"); s.add_argument("--content-type", dest="content_type")
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("build"); s.set_defaults(func=cmd_build)

    s = sub.add_parser("stamp-ids",
                       help="add a permanent ULID id: to any note lacking one (idempotent migration)")
    s.add_argument("--relink", action="store_true",
                   help="also rewrite [[stem]] links to rename-safe [[id:...]] form")
    s.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    s.set_defaults(func=cmd_stamp_ids)

    for name, fn in (("validate", cmd_validate), ("connectivity", cmd_connectivity)):
        s = sub.add_parser(name); s.add_argument("--strict", action="store_true"); s.set_defaults(func=fn)

    s = sub.add_parser("claims-lint",
                       help="verify every claim's locator resolves to a real span in its source")
    s.add_argument("--strict", action="store_true")
    s.set_defaults(func=cmd_claims_lint)

    s = sub.add_parser("claim-add",
                       help="ground a quote in a source and record a claim to a note's sidecar")
    s.add_argument("--note", required=True, help="note path, e.g. shared/concepts/foo.md")
    s.add_argument("--source", required=True, help="source id (src-...) the quote comes from")
    s.add_argument("--quote", required=True, help="exact supporting span lifted from the source")
    s.add_argument("--text", required=True, help="the claim statement")
    s.add_argument("--asserted-by", dest="asserted_by", default="human",
                   help="'human' (default) or 'model:<name>'")
    s.add_argument("--confidence", type=float, default=None)
    s.set_defaults(func=cmd_claim_add)

    s = sub.add_parser("verify",
                       help="promote claims asserted -> verified (human or cross-source rule)")
    s.add_argument("--claim", help="claim id to verify")
    s.add_argument("--note", help="verify every claim in this note (path)")
    s.add_argument("--by", default="human", help="'human' (default) or 'cross-source'")
    s.add_argument("--sign", action="store_true",
                   help="(business tier; OFF by default) request a GPG signature — not built in the core")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("coverage",
                       help="substance-aware extraction quality per source (catches page-padding/thin stubs)")
    s.add_argument("--source", help="limit to one source id")
    s.add_argument("--min-chars", dest="min_chars", type=int, default=200,
                   help="a note below this many prose chars counts as thin (default 200)")
    s.add_argument("--objectives", action="store_true",
                   help="also check objective+keyword coverage for reference/syllabus sources")
    s.add_argument("--strict", action="store_true", help="exit non-zero if anything is flagged")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_coverage)

    s = sub.add_parser("classify", help="show the detected doc-type per source (drives extraction rules)")
    s.add_argument("--source", help="limit to one source id")
    s.add_argument("--deep", action="store_true", help="also sniff source content (slower; extracts)")
    s.set_defaults(func=cmd_classify)

    s = sub.add_parser("rollback",
                       help="undo an ingestion: --source <id> (one doc) or --last/--to (a finalize commit)")
    s.add_argument("--source", help="roll back ONE source's ingestion: delete notes citing only it (+claims)")
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="with --source: preview what would be deleted; write nothing")
    s.add_argument("--list", action="store_true", help="show recent commits + which added notes")
    s.add_argument("--last", action="store_true", help="undo the most recent commit (HEAD)")
    s.add_argument("--to", help="undo/reset to a specific commit hash")
    s.add_argument("--hard", action="store_true",
                   help="git reset --hard (discard everything after) instead of a safe revert")
    s.add_argument("--force", action="store_true", help="proceed even with an uncommitted working tree")
    s.set_defaults(func=cmd_rollback)

    s = sub.add_parser("relations",
                       help="query typed edges (e.g. who is X's sibling) — data, not LLM inference")
    s.add_argument("--from", dest="from_ref", help="note id/stem: list its typed relations (+ inverses)")
    s.add_argument("--type", help="filter/list edges of this type")
    s.add_argument("--stale", action="store_true", help="notes made stale by a supersedes edge")
    s.add_argument("--contradictions", action="store_true", help="list contradicts edges")
    s.set_defaults(func=cmd_relations)

    s = sub.add_parser("bootstrap",
                       help="tool-system co-evolution: propose schema/prompt/vocab fixes from evidence")
    s.add_argument("action", choices=["propose", "accept", "reject", "status"])
    s.add_argument("ref", nargs="?", help="proposal (dialog event) id for accept/reject")
    s.add_argument("--floor", type=float, default=1.0, help="notes/page coverage floor (propose)")
    s.add_argument("--note", help="optional rationale for accept/reject")
    s.set_defaults(func=cmd_bootstrap)

    s = sub.add_parser("dialog", help="the CoDIAK dialog/decision record (append-only reasoning trail)")
    s.add_argument("action", choices=["add", "log", "trail"])
    s.add_argument("ref", nargs="?", help="for 'trail': the conclusion/claim/event id to reconstruct")
    s.add_argument("--kind", help="question | decision | contradiction | rejection | schema-change")
    s.add_argument("--text", help="the dialog text (for add)")
    s.add_argument("--links", help="comma-separated claim/note/source/dialog ids the event concerns")
    s.add_argument("--author", default="human", help="'human' or 'model:<name>'")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_dialog)

    s = sub.add_parser("project", help="(re)build the derived SQLite projection from the files")
    s.set_defaults(func=cmd_project)

    s = sub.add_parser("view", help="render a viewer-independent view from the projection (LLM off)")
    s.add_argument("name", nargs="?", help="view name (see --list)")
    s.add_argument("--list", action="store_true", help="list available views")
    s.add_argument("--of", help="conclusion/claim/event id (for dialog-trail)")
    s.add_argument("--html", action="store_true", help="render HTML instead of markdown")
    s.add_argument("--public", action="store_true", help="restrict to the shared/public view")
    s.add_argument("--stdout", action="store_true", help="print to stdout instead of writing out/")
    s.set_defaults(func=cmd_view)

    s = sub.add_parser("query", help="structured SQL query of the projection (with the LLM off)")
    s.add_argument("name", nargs="?", help="canned query name (see --list)")
    s.add_argument("--sql", help="raw read-only SQL (SELECT/WITH/PRAGMA; local CLI only)")
    s.add_argument("--list", action="store_true", help="list the canned queries")
    s.add_argument("--source"); s.add_argument("--from", dest="from_ref")
    s.add_argument("--type"); s.add_argument("--of"); s.add_argument("--tag"); s.add_argument("--note")
    s.add_argument("--public", action="store_true",
                   help="restrict to the MCP-public view (shared cloud, no home/personal)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("search")
    s.add_argument("query"); s.add_argument("--limit", type=int, default=10)
    s.add_argument("--semantic", action="store_true", help="hybrid BM25+vector rerank if tier available")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("source-check"); s.set_defaults(func=cmd_source_check)

    s = sub.add_parser("process-inbox"); s.set_defaults(func=cmd_process_inbox)

    s = sub.add_parser("ingest-log")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--summary", action="store_true")
    s.set_defaults(func=cmd_ingest_log)

    s = sub.add_parser("log")
    s.add_argument("--title", required=True); s.add_argument("--details")
    s.add_argument("--personal", action="store_true"); s.set_defaults(func=cmd_log)

    s = sub.add_parser("scan"); s.add_argument("--personal", action="store_true"); s.set_defaults(func=cmd_scan)

    s = sub.add_parser("finalize-ingest")
    s.add_argument("--message"); s.add_argument("--strict", action="store_true")
    s.add_argument("--push", action="store_true"); s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_finalize)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
