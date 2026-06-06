"""projection.py — the derived, regenerable SQLite projection (v3 Phase D). Stdlib `sqlite3` only.

This is the **lens, not the truth**: a single-file SQLite database **projected from the files** that
makes the wiki queryable with **structured SQL and the LLM switched off** — the precise rebuttal to
"markdown can't query," with no server. Delete it and rebuild from the files at any time; it is
gitignored (`.runtime/projection.db`) and nothing load-bearing depends on it (the BM25 core still runs
without it).

Tables: `objects` (notes), `tags`, `note_sources`, `claims`, `links` (typed + see-also), `sources`
(incl. mixed-object metadata — size/dims for non-text objects), `dialog` (Phase E), `embeddings`
(optional), `meta`. A `public` flag mirrors the MCP privacy model (shared cloud, no home/personal tag)
so the same engine can serve a redacted view.

Stdlib-only keeps the zero-dependency promise. Image dimensions use Pillow **if present**
(feature-detected) — absent, dims are simply NULL.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import wiki_tool as wt        # call-time use only (no import cycle: wt imports us lazily)
import claims as _claims

PRIVATE_TAGS = {"home", "personal"}
IMAGE_TYPES = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif"}

SCHEMA_SQL = """
CREATE TABLE objects (
  id TEXT PRIMARY KEY, path TEXT, name TEXT, cloud TEXT, type TEXT, subtype TEXT,
  title TEXT, status TEXT, created TEXT, updated TEXT, public INTEGER, verification TEXT
);
CREATE TABLE tags (object_id TEXT, tag TEXT);
CREATE TABLE note_sources (object_id TEXT, source_id TEXT);
CREATE TABLE sources (
  id TEXT PRIMARY KEY, title TEXT, content_type TEXT, sha256 TEXT, first_ingested TEXT,
  present INTEGER, size INTEGER, width INTEGER, height INTEGER, duration REAL, public INTEGER
);
CREATE TABLE claims (
  id TEXT PRIMARY KEY, note_id TEXT, note_path TEXT, source_id TEXT, text TEXT,
  locator TEXT, page TEXT, section TEXT, quote TEXT, char_range TEXT,
  asserted_by TEXT, verified_by TEXT, confidence REAL, created TEXT, public INTEGER
);
CREATE TABLE links (from_id TEXT, to_id TEXT, type TEXT, public INTEGER);
CREATE TABLE dialog (id TEXT PRIMARY KEY, ts TEXT, kind TEXT, text TEXT, links TEXT,
                     author TEXT, public INTEGER);
CREATE TABLE embeddings (object_id TEXT, dim INTEGER, vector BLOB);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX ix_claims_source ON claims(source_id);
CREATE INDEX ix_claims_note ON claims(note_id);
CREATE INDEX ix_links_from ON links(from_id);
CREATE INDEX ix_links_to ON links(to_id);
CREATE INDEX ix_links_type ON links(type);
CREATE INDEX ix_tags_tag ON tags(tag);
"""


def db_path() -> Path:
    return Path(wt.ROOT) / ".runtime" / "projection.db"


def _is_public(note) -> bool:
    if note.cloud != "shared":
        return False
    return not ({str(t).lower() for t in note.as_list("tags")} & PRIVATE_TAGS)


def _image_dims(path: Path):
    if path.suffix.lower().lstrip(".") not in IMAGE_TYPES:
        return (None, None)
    try:
        from PIL import Image          # feature-detected; absent -> NULL dims
        with Image.open(path) as im:
            return im.size              # (width, height)
    except Exception:
        return (None, None)


def build(notes=None, reg=None, db=None) -> Path:
    """Build (rebuild) the projection from the files. Fully regenerable; returns the db path."""
    if notes is None:
        notes = wt.find_notes()
    if reg is None:
        reg = wt.load_registry()
    path = Path(db) if db else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")     # readers (MCP) + writer (rebuild/watcher) coexist
        con.executescript(SCHEMA_SQL)
        by_id, by_stem = wt.build_index_maps(notes)
        pub_by_key = {}                              # node_key -> public flag, for link denormalization
        # objects + tags + note_sources + claims
        for n in sorted(notes, key=lambda x: (x.id or x.name)):
            if n.name in wt.HUB_NAMES:
                continue
            pub = 1 if _is_public(n) else 0
            pub_by_key[wt._node_key(n)] = pub
            note_claims = _claims.load_claims(n.path)
            con.execute(
                "INSERT OR REPLACE INTO objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (n.id, wt._rel(n.path), n.name, n.cloud, n.meta.get("type", ""),
                 n.meta.get("subtype", ""), n.title, n.meta.get("status", ""),
                 str(n.meta.get("created", "")), str(n.meta.get("updated", "")), pub,
                 _claims.note_verification(note_claims)))
            for tag in n.as_list("tags"):
                con.execute("INSERT INTO tags VALUES (?,?)", (n.id, str(tag)))
            for sid in n.as_list("sources"):
                con.execute("INSERT INTO note_sources VALUES (?,?)", (n.id, str(sid)))
            for c in note_claims:
                loc = c.get("locator") or {}
                con.execute(
                    "INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (c.get("id"), n.id, wt._rel(n.path), c.get("source_id", ""), c.get("text", ""),
                     json.dumps(loc, sort_keys=True), str(loc.get("page", loc.get("pages", ""))),
                     str(loc.get("section", "")), str(loc.get("quote", "")),
                     json.dumps(loc.get("char_range")) if loc.get("char_range") else "",
                     c.get("asserted_by", ""), c.get("verified_by", "none"),
                     c.get("confidence"), c.get("created", ""), pub))
        # links: typed edges + untyped body see-also, resolved to target ids
        for n in notes:
            if n.name in wt.HUB_NAMES:
                continue
            fk = wt._node_key(n)
            fpub = pub_by_key.get(fk, 0)
            for tl in n.typed_links:
                matches, _kind = wt._resolve_ref(tl.get("to"), by_id, by_stem)
                for m in matches:
                    tk = wt._node_key(m)
                    con.execute("INSERT INTO links VALUES (?,?,?,?)",
                                (fk, tk, str(tl.get("type", "")), 1 if (fpub and pub_by_key.get(tk, 0)) else 0))
            for tgt in n.links:
                matches, _kind = wt.resolve_link(tgt, by_id, by_stem)
                for m in matches:
                    if m is n:
                        continue
                    tk = wt._node_key(m)
                    con.execute("INSERT INTO links VALUES (?,?,?,?)",
                                (fk, tk, "see-also", 1 if (fpub and pub_by_key.get(tk, 0)) else 0))
        # sources (+ mixed-object metadata) — public iff cited by >=1 public note
        public_sids = set()
        for n in notes:
            if n.name not in wt.HUB_NAMES and _is_public(n):
                public_sids.update(str(s) for s in n.as_list("sources"))
        mid = wt.machine_id()
        for sid, e in sorted(reg.items()):
            loc = e.get("locations", {}).get(mid) or {}
            p = loc.get("path")
            size = w = h = None
            if p:
                fp = Path(p) if Path(p).is_absolute() else (Path(wt.ROOT) / p)
                if fp.exists():
                    try:
                        size = fp.stat().st_size
                    except OSError:
                        size = None
                    w, h = _image_dims(fp)
            present = 1 if any(v.get("present") for v in e.get("locations", {}).values()) else 0
            con.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (sid, e.get("title", ""), e.get("content_type", ""), e.get("sha256", ""),
                         e.get("first_ingested", ""), present, size, w, h, None,
                         1 if sid in public_sids else 0))
        # dialog (Phase E) — projected if Schema/dialog.jsonl exists. An event is public unless it links
        # any private (non-public) object/claim id, so the reasoning trail can be served read-only.
        private_ids = {r[0] for r in con.execute("SELECT id FROM objects WHERE public=0")}
        private_ids |= {r[0] for r in con.execute("SELECT id FROM claims WHERE public=0")}
        dlog = Path(wt.SCHEMA) / "dialog.jsonl"
        if dlog.exists():
            for line in dlog.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                links = d.get("links", [])
                pub = 0 if any(l in private_ids for l in links) else 1
                con.execute("INSERT OR REPLACE INTO dialog VALUES (?,?,?,?,?,?,?)",
                            (d.get("id"), d.get("ts", ""), d.get("kind", ""), d.get("text", ""),
                             json.dumps(links), d.get("author", ""), pub))
        n_obj = con.execute("SELECT count(*) FROM objects").fetchone()[0]
        n_claims = con.execute("SELECT count(*) FROM claims").fetchone()[0]
        for k, v in (("notes", n_obj), ("claims", n_claims),
                     ("links", con.execute("SELECT count(*) FROM links").fetchone()[0]),
                     ("sources", con.execute("SELECT count(*) FROM sources").fetchone()[0]),
                     ("schema_version", 1)):
            con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(v)))
        con.commit()
    finally:
        con.close()
    return path


def connect_ro(path=None) -> sqlite3.Connection:
    """Open the projection **read-only** (the query layer never mutates the record or the DB)."""
    p = Path(path) if path else db_path()
    if not p.exists():
        raise FileNotFoundError(f"projection not built ({p}); run `wiki_tool project`")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=1;")
    return con


def _rows(con, sql, params=()):
    return [dict(r) for r in con.execute(sql, params).fetchall()]


# --------------------------------------------------------------------------- #
# canned queries — each answers a fixed question with the LLM off. `public` filters to the MCP view.
# --------------------------------------------------------------------------- #
def q_claims_by_source(con, source, public=False):
    pub = " AND public=1" if public else ""
    return _rows(con, f"SELECT id, note_path, text, quote, page, asserted_by, verified_by "
                      f"FROM claims WHERE source_id=?{pub} ORDER BY id", (source,))


def q_unverified_claims(con, source=None, public=False):
    where = "(verified_by IS NULL OR verified_by NOT IN ('human','cross-source'))"
    params = []
    if source:
        where += " AND source_id=?"
        params.append(source)
    if public:
        where += " AND public=1"
    return _rows(con, f"SELECT id, note_path, source_id, text, asserted_by, verified_by "
                      f"FROM claims WHERE {where} ORDER BY id", tuple(params))


def q_verified_claims(con, source=None, public=False):
    where = "verified_by IN ('human','cross-source')"
    params = []
    if source:
        where += " AND source_id=?"
        params.append(source)
    if public:
        where += " AND public=1"
    return _rows(con, f"SELECT id, note_path, source_id, text, asserted_by, verified_by "
                      f"FROM claims WHERE {where} ORDER BY id", tuple(params))


def q_flagged_claims(con, public=False):
    """Model-asserted claims that aren't verified — the hallucination-risk flag (Phase F)."""
    pub = " AND public=1" if public else ""
    return _rows(con, f"SELECT id, note_path, source_id, text, asserted_by, verified_by "
                      f"FROM claims WHERE asserted_by LIKE 'model:%' "
                      f"AND (verified_by IS NULL OR verified_by NOT IN ('human','cross-source')){pub} "
                      f"ORDER BY id")


def q_relations(con, frm, type=None, public=False):
    """Typed relations of a note by id (or its objects.name), both directions; inverse applied for 'in'."""
    inverses = wt.load_link_types()["inverses"]
    node = con.execute("SELECT id FROM objects WHERE id=? OR name=?", (frm, frm)).fetchone()
    if not node:
        return []
    nid = node["id"]
    pub = " AND l.public=1" if public else ""
    out = _rows(con, f"SELECT o.id AS other_id, o.title AS other, l.type AS type, 'out' AS dir "
                     f"FROM links l JOIN objects o ON o.id=l.to_id WHERE l.from_id=?{pub}", (nid,))
    inn = _rows(con, f"SELECT o.id AS other_id, o.title AS other, l.type AS type, 'in' AS dir "
                     f"FROM links l JOIN objects o ON o.id=l.from_id WHERE l.to_id=?{pub}", (nid,))
    for r in inn:
        r["type"] = inverses.get(r["type"], r["type"])   # present from the queried note's perspective
    rows = out + inn
    if type:
        rows = [r for r in rows if r["type"] == type]
    return sorted(rows, key=lambda r: (r["type"], r["other"] or ""))


def q_supersedes(con, of=None, public=False):
    pub = " AND l.public=1" if public else ""
    where = "l.type='supersedes'" + pub
    params = []
    if of:
        where += " AND (older.id=? OR older.name=?)"
        params += [of, of]
    return _rows(con, f"SELECT newer.id AS superseding_id, newer.title AS superseding, "
                      f"older.id AS stale_id, older.title AS stale "
                      f"FROM links l JOIN objects newer ON newer.id=l.from_id "
                      f"JOIN objects older ON older.id=l.to_id WHERE {where} "
                      f"ORDER BY older.title", tuple(params))


def q_contradictions(con, public=False):
    pub = " AND l.public=1" if public else ""
    return _rows(con, f"SELECT a.title AS a, b.title AS b, a.id AS a_id, b.id AS b_id "
                      f"FROM links l JOIN objects a ON a.id=l.from_id JOIN objects b ON b.id=l.to_id "
                      f"WHERE l.type='contradicts'{pub} ORDER BY a.title")


def q_sources(con, type=None, public=False):
    where, params = [], []
    if type:
        where.append("content_type=?")
        params.append(type)
    if public:
        where.append("public=1")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return _rows(con, f"SELECT id, title, content_type, size, width, height, duration, present "
                      f"FROM sources{clause} ORDER BY id", tuple(params))


def q_tag(con, tag, public=False):
    pub = " AND o.public=1" if public else ""
    return _rows(con, f"SELECT o.id, o.title, o.type, o.path FROM tags t "
                      f"JOIN objects o ON o.id=t.object_id WHERE t.tag=?{pub} ORDER BY o.title", (tag,))


def q_note_claims(con, note, public=False):
    pub = " AND public=1" if public else ""
    return _rows(con, f"SELECT id, source_id, text, quote, page, asserted_by, verified_by "
                      f"FROM claims WHERE (note_id=? OR note_path=?){pub} ORDER BY id", (note, note))


def q_dialog(con, type=None, public=False):
    """The CoDIAK dialog record (Phase E). `type` filters by kind; `public` restricts to the shared view."""
    where, params = [], []
    if type:
        where.append("kind=?")
        params.append(type)
    if public:
        where.append("public=1")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return _rows(con, f"SELECT id, ts, kind, text, links, author FROM dialog{clause} ORDER BY ts, id",
                 tuple(params))


def q_dialog_trail(con, of, public=False):
    """Reconstruct the reasoning trail for a conclusion/claim/event id, from the projected dialog."""
    pub = " WHERE public=1" if public else ""
    rows = _rows(con, f"SELECT id, ts, kind, text, links, author FROM dialog{pub} ORDER BY ts, id")
    out = []
    for r in rows:
        links = json.loads(r["links"]) if r["links"] else []
        if r["id"] == of or of in links:
            r["links"] = links
            out.append(r)
    return out


# name -> (function, [required param names], help)
CANNED = {
    "claims-by-source": (q_claims_by_source, ["source"], "all claims citing a source"),
    "unverified-claims": (q_unverified_claims, [], "claims not human/cross-source verified (--source optional)"),
    "verified-claims": (q_verified_claims, [], "human/cross-source verified claims only (--source optional)"),
    "flagged-claims": (q_flagged_claims, [], "model-asserted, unverified claims (hallucination-risk flag)"),
    "relations": (q_relations, ["from"], "typed relations of a note (--type optional) — incl. derived inverse"),
    "supersedes": (q_supersedes, [], "what supersedes what (--of <id> optional)"),
    "contradictions": (q_contradictions, [], "all contradicts edges"),
    "sources": (q_sources, [], "registered source objects + mixed-object metadata (--type optional)"),
    "tag": (q_tag, ["tag"], "objects carrying a tag"),
    "note-claims": (q_note_claims, ["note"], "claims belonging to a note"),
    "dialog": (q_dialog, [], "the CoDIAK dialog record (--type filters by kind)"),
    "dialog-trail": (q_dialog_trail, ["of"], "reconstruct the reasoning trail for an id (--of)"),
}

# Queries safe to expose over the MCP server (public-filtered). Excludes raw SQL (CLI only).
MCP_QUERIES = set(CANNED)


def run_canned(name, params, public=False, path=None):
    """Run a canned query by name. `params` is a dict of filter args. Returns list[dict] rows."""
    import inspect
    if name not in CANNED:
        raise KeyError(f"unknown query '{name}' (known: {', '.join(sorted(CANNED))})")
    fn, required, _help = CANNED[name]
    for r in required:
        if not params.get(r):
            raise ValueError(f"query '{name}' requires --{r}")
    accepted = set(inspect.signature(fn).parameters) - {"con", "public"}
    con = connect_ro(path)
    try:
        kwargs = {kk: v for k, v in params.items() if v is not None
                  for kk in [k.replace("from", "frm")] if kk in accepted}
        return fn(con, public=public, **kwargs)
    finally:
        con.close()


def run_sql(sql, path=None):
    """Run a raw read-only SQL query (local/CLI only — never exposed over MCP). Rejects writes."""
    low = sql.strip().lower()
    if not (low.startswith("select") or low.startswith("with") or low.startswith("pragma")):
        raise ValueError("only read-only SELECT/WITH/PRAGMA queries are allowed")
    con = connect_ro(path)
    try:
        return _rows(con, sql)
    finally:
        con.close()
