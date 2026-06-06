"""doctypes.py — doc-type-aware curation (v3). Stdlib only.

A human curator treats a meeting transcript differently from a 78-page syllabus, an API reference, or a
software README. This module codifies that: each **doc-type profile** carries (1) detection hints,
(2) extraction *guidance* for the agent (the "finer instructions"), and (3) a deterministic **coverage
rule** — what "fully extracted" means for that type, so the gate measures the *right* thing instead of
the gameable notes/page proxy.

The flagship rule is **reference/syllabus coverage**: a syllabus enumerates its own targets — numbered
learning objectives (`FL-1.2.1`, `TA-2.3.1`) and per-chapter keyword lists — both regex-extractable
from the source. So we can check, deterministically, that the wiki has a note/claim per objective and a
note per keyword. That's real knowledge coverage, not paper.
"""
from __future__ import annotations

import re

OBJECTIVE_RE = re.compile(r"\b[A-Z]{1,6}-\d+\.\d+\.\d+\b")     # FL-1.2.1, TA-2.3.1, CTAL-…
KLEVEL_RE = re.compile(r"\(K[1-4]\)")                          # cognitive levels mark objectives
# page-name detector (mirrors wiki_tool.PAGE_NAME_RE) — use .search; catches "istqb-ctfl-p30-…", "p1-cover"
PAGE_NAME_RE = re.compile(r"(?:^|[-_ ])(?:pp?|pg|pages?)[-_ ]?\d+(?:[-_ ]?\d+)?(?=[-_ ]|$)", re.I)


def _substance(note) -> int:
    """Approx prose chars: body minus the H1 title and the generated '## Sources' block (stub detector)."""
    body = re.sub(r"^#\s+.*$", "", note.body, count=1, flags=re.M)
    body = re.sub(r"##\s*Sources\b.*", "", body, flags=re.S)
    return len(re.sub(r"\s+", " ", body).strip())


# Profiles, most-specific first; `general` is the fallback. `filename`/`content` are lowercase regex/
# substrings used by classify(); `coverage` names the rule; `guidance` is the agent's playbook (also in
# Schema/doc-types.md).
DOC_TYPES = [
    {"name": "reference",
     "filename": [r"syllabus", r"istqb", r"\bstandard\b", r"specification", r"\bspec\b", r"handbook",
                  r"overview", r"framework"],
     "content": ["learning objective", "keywords:", "k1)", "k2)", "k3)"],
     "min_hits": 2,
     "coverage": "objective-keyword",
     "guidance": "Extract one note per LEARNING OBJECTIVE and per KEYWORD; name notes by the "
                 "concept/keyword (never by page); record the objective code (FL-1.2.1) as an alias. "
                 "Expect many notes on dense pages, zero on cover/ToC/index."},
    {"name": "api",
     "filename": [r"\bapi\b", r"swagger", r"openapi", r"endpoint", r"\brest\b", r"reference"],
     "content": ["get /", "post /", "put /", "delete /", "endpoint", "request body", "response schema",
                 "http/1", "application/json"],
     "min_hits": 2,
     "coverage": "endpoint",
     "guidance": "Extract one note per ENDPOINT (method + path) and per shared schema/parameter; "
                 "name notes by the endpoint. Capture auth, status codes, request/response shapes."},
    {"name": "software-docs",
     "filename": [r"readme", r"\bdocs?\b", r"design", r"architecture", r"adr", r"runbook", r"install"],
     "content": ["## installation", "## usage", "## configuration", "## architecture", "```",
                 "getting started", "prerequisites"],
     "min_hits": 2,
     "coverage": "substance",
     "guidance": "Extract per module/component, per config option, per concept/decision; capture "
                 "commands and gotchas. Link components to the system topic."},
    {"name": "meeting-notes",
     "filename": [r"transcript", r"meeting", r"stand[ -]?up", r"\bcall\b", r"catchup", r"weekly",
                  r"\bsync\b", r"\bkt\b", r"upskill", r"notes"],
     "content": ["transcript", "action item", "attendees", "agenda", "minutes", "discussed", "follow up"],
     "min_hits": 1,
     "coverage": "substance",
     "guidance": "Extract DECISIONS, ACTION ITEMS, and ATTENDEES (as entity notes); capture the topics "
                 "and open questions. Few notes is fine — DO NOT pad. Link to the project + people."},
    {"name": "general",
     "filename": [], "content": [], "min_hits": 1, "coverage": "substance",
     "guidance": "Enumerate every entity/concept/method; name notes by concept; don't summarise or pad."},
]
_BY_NAME = {p["name"]: p for p in DOC_TYPES}


def profile(name) -> dict:
    return _BY_NAME.get(name, _BY_NAME["general"])


def classify(title: str, text: str = "") -> str:
    """Best-guess doc-type from filename (cheap) then a content sniff. Returns a profile name."""
    # An unambiguous syllabus signal (≥3 numbered objective codes) overrides the filename — so a
    # syllabus mis-named "…notes" still classifies as `reference`.
    if text and len(set(OBJECTIVE_RE.findall(text))) >= 3:
        return "reference"
    t = (title or "").lower()
    for p in DOC_TYPES:
        if p["filename"] and any(re.search(pat, t) for pat in p["filename"]):
            return p["name"]
    body = (text or "")[:6000].lower()
    if body:
        for p in DOC_TYPES:
            if p["content"] and sum(1 for k in p["content"] if k in body) >= p.get("min_hits", 1):
                return p["name"]
    return "general"


# --------------------------------------------------------------------------- #
# target extraction (the reference/syllabus coverage rule)
# --------------------------------------------------------------------------- #
def extract_objectives(text: str) -> list:
    return sorted(set(OBJECTIVE_RE.findall(text or "")))


# words that, leading a "term", mark a sentence fragment rather than a keyword (PDF keyword sections
# sometimes bleed into prose). Keyword extraction is best-effort + format-dependent; objective-code
# coverage is the exact signal.
_KW_STOP = {"even", "see", "for", "this", "the", "and", "via", "other", "ensuring", "recording",
            "retrieval", "scheduling", "consider", "such", "these", "those", "with", "where", "when",
            "which", "ensure", "verifying", "i.e", "e.g", "including", "based"}


def extract_keywords(text: str) -> list:
    """Pull a syllabus's declared keyword terms from its 'Keywords:' blocks. Best-effort + de-noised."""
    out = []
    for sec in re.findall(r"(?im)^[ \t]*keywords?\b\s*[:\-]?\s*((?:.+\n?)+?)(?:\n\s*\n|\Z)", text or ""):
        sec = re.sub(r"\s+", " ", sec)
        for term in re.split(r"[,;]", sec):
            t = re.sub(r"\s+", " ", term.strip().strip(".")).lower()
            if not (2 < len(t) <= 40) or not (1 <= len(t.split()) <= 5):
                continue
            if any(c in t for c in "():/%[]") or "i.e" in t or "e.g" in t or any(c.isdigit() for c in t):
                continue
            if t.split()[0] in _KW_STOP:
                continue
            out.append(t)
    return sorted(set(out))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def reference_coverage(source_text: str, notes, min_chars: int = 200) -> dict:
    """Objective + keyword coverage of `notes` against a reference source's own enumerated targets.

    Coverage counts only **substantive** notes (≥`min_chars` prose) — a stub doesn't 'cover' anything.
    An objective is covered if its code appears in a substantive note (body/alias); a keyword is covered
    if a substantive note's slug/alias/title matches it (incl. a despaced form, tolerating PDF artifacts)
    or the phrase appears in its body. Also reports how many citing notes are page-dumps vs concepts, so
    "objectives 64/64" can't hide the fact that the targets live in page notes rather than concept notes.
    """
    objectives = extract_objectives(source_text)
    keywords = extract_keywords(source_text)
    total_notes = len(notes)
    page_notes = sum(1 for n in notes if PAGE_NAME_RE.search(n.name))
    notes = [n for n in notes if _substance(n) >= min_chars]   # stubs don't count as coverage
    blob = " ".join((n.body + " " + " ".join(str(a) for a in n.as_list("aliases")) + " " + n.name).lower()
                    for n in notes)
    def despace(s):
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())
    slugs, despaced = set(), set()
    for n in notes:
        for s in (n.name, n.title, *[str(a) for a in n.as_list("aliases")]):
            slugs.add(_slug(s)); despaced.add(despace(s))
    obj_missing = [c for c in objectives if c.lower() not in blob]
    # match keyword -> note by slug, by despaced form (tolerates PDF mid-word spaces), or a body mention
    kw_missing = [k for k in keywords
                  if _slug(k) not in slugs and despace(k) not in despaced and k not in blob]
    return {
        "objectives": {"total": len(objectives), "covered": len(objectives) - len(obj_missing),
                       "missing": obj_missing},
        "keywords": {"total": len(keywords), "covered": len(keywords) - len(kw_missing),
                     "missing": kw_missing},
        "notes": total_notes, "page_notes": page_notes,
        "concept_notes": total_notes - page_notes,
    }
