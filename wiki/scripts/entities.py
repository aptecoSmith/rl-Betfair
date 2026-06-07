"""entities.py — deterministic per-document entity/concept *discovery* (v3). Stdlib only.

The failure this fixes: the ingest never *searched* a document for the things in it. Segmentation
(`extractors/`) chops a doc into chapters; claim grounding verifies a quote; but nothing ever asked
"what people, orgs, tools, places and concepts does this document name?" That was left entirely to the
agent noticing while it read — and noticing isn't searching. So a meeting note naming Samrat, Barkha,
Vix, CBE and the Manchester test lab could be folded into one bullet list and no guardrail could tell.

This module supplies the missing pass. Per the v3 invariant (*deterministic code extracts facts; the
LLM synthesises*), the split is:

  * DETERMINISTIC (here): surface every **candidate** entity/concept from raw source text — proper-noun
    phrases, acronyms, and known terms (seeded from a small lexicon *and* from the wiki's own existing
    node titles/aliases, so the corpus bootstraps its own recall). Then check which candidates are
    **represented** (a node title/alias/stem, or a `[[link]]` target) and which are **missing**.
  * LLM (the agent): for each candidate, decide node vs link vs not-worth-it. Missing high-confidence
    entities block the commit (the coverage gate consumes `classify()`); the agent clears them by
    creating a node, linking an existing one, or recording an explicit skip-with-reason.

Pure: operates on text and on plain strings (titles/aliases/links) passed in by the caller. Reading the
source file lives in wiki_tool (`get_source_text`), which keeps this zero-dependency.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# normalization — one canonical form so "Transport for Wales", "transport-for-wales"
# and "Transport  for  Wales" all compare equal. Hyphen/underscore/slash == space.
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[^a-z0-9'& ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str):
    return [t for t in normalize(s).split() if t]


_FRONTMATTER_RE = re.compile(r"\A﻿?---\r?\n.*?\r?\n---\r?\n", re.S)


def content_text(text: str) -> str:
    """Strip a leading YAML frontmatter block so we scan the *content*, not metadata keys
    (Title/Author/ContentType/...). Idempotent — a no-op when there's no fence."""
    return _FRONTMATTER_RE.sub("", text or "", count=1)


# --------------------------------------------------------------------------- #
# stoplist — generic Capitalized words that start sentences or are calendar/common words and are NOT
# entities. Keeps the proper-noun pass precise without a part-of-speech tagger. Lowercased forms.
# --------------------------------------------------------------------------- #
STOPWORDS = {
    # calendar
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "today", "tomorrow", "yesterday",
    # sentence-initial / discourse words commonly Capitalized
    "the", "this", "that", "these", "those", "there", "their", "they", "then", "thus",
    "we", "our", "us", "i", "it", "its", "he", "she", "his", "her", "you", "your",
    "a", "an", "and", "or", "but", "so", "if", "as", "at", "by", "in", "on", "of", "for",
    "to", "from", "with", "without", "into", "over", "under", "per", "via", "no", "not",
    "yes", "ok", "okay", "also", "however", "meanwhile", "currently", "open", "note",
    "next", "now", "see", "todo", "tbd", "n", "etc", "e", "g", "ie", "eg", "vs",
    "who", "what", "when", "where", "why", "how", "which", "all", "any", "some", "each",
    "more", "most", "less", "new", "old", "good", "bad", "high", "low", "first", "last",
    "discussion", "direction", "captured", "unknown", "flagged", "asking", "questions",
    # generic geographic/temporal acronyms that aren't wiki-worthy entities on their own
    "uk", "us", "usa", "eu", "am", "pm", "utc", "gmt",
    # source-metadata keys that can survive frontmatter stripping in odd formats
    "title", "author", "reference", "contenttype", "created", "ingested", "processed", "source",
    # common sentence-initial verbs / modals / adverbs that get Capitalized but are never entities.
    # (Only single-token or all-stopword candidates are dropped, so multiword names like "West Point"
    # survive — a real entity never consists *entirely* of these.)
    "can", "could", "would", "should", "shall", "will", "may", "might", "must", "do", "does",
    "did", "done", "doing", "make", "makes", "made", "making", "need", "needs", "look", "looks",
    "set", "sets", "go", "goes", "going", "gone", "get", "gets", "getting", "got", "give", "gives",
    "given", "take", "takes", "taken", "talk", "talks", "talking", "start", "starts", "started",
    "tell", "tells", "told", "use", "uses", "used", "using", "add", "adds", "added", "aim", "aims",
    "post", "posts", "slim", "chase", "click", "clicks", "save", "saves", "select", "selects",
    "describe", "describes", "encourage", "encourages", "felt", "feel", "feels", "hopefully",
    "perhaps", "instead", "exactly", "maybe", "probably", "actually", "basically", "essentially",
    "really", "just", "even", "still", "again", "once", "want", "wants", "wanted", "let", "lets",
    "put", "puts", "run", "runs", "running", "ran", "find", "finds", "found", "keep", "keeps",
    "kept", "say", "says", "said", "seen", "come", "comes", "came", "work", "works", "worked",
    "working", "think", "thinks", "thought", "know", "knows", "knew", "mean", "means", "meant",
    "show", "shows", "showed", "call", "calls", "called", "try", "tries", "tried", "move", "moves",
    "change", "changes", "changed", "help", "helps", "helped", "able", "going", "lots", "lot",
    "blink", "felt", "favs", "quick", "robot", "wary", "den", "talking",
    "only", "two", "three", "second", "third", "thing", "things", "people", "meeting", "meetings",
    "lunch", "video", "videos", "round", "rounds", "real", "reduce", "progress", "record", "records",
    "struggling", "struggle", "spreadsheet", "spreadsheets", "escalate", "escalated", "escalates",
    "stuff", "everything", "anything", "something", "nothing", "someone", "anyone", "everyone",
    "met", "meet", "meets", "meeting", "spoke", "speak", "spoken", "speaks", "saw", "join", "joins",
    "joined", "joining", "asked", "asks", "raised", "raise", "discussed", "discuss", "agreed", "agree",
    # pronoun/auxiliary contractions (never entities)
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "won't", "can't",
    "couldn't", "shouldn't", "wouldn't", "i'm", "i'll", "i've", "i'd", "we're", "we'll", "we've",
    "we'd", "you're", "you'll", "you've", "they're", "they'll", "they've", "it's", "that's",
    "there's", "here's", "he's", "she's", "what's", "who's", "let's",
}

# A small seed lexicon of lowercase-able tool / infra / domain TERMS that the proper-noun regex can't
# catch because they appear in prose as ordinary words. Deliberately short — the *main* recall comes
# from the proper-noun pass and from the wiki's own node titles/aliases (terms_from_notes). Extend as
# the domain grows; this is a recall aid, NOT an imposed taxonomy.
SEED_TERMS = {
    "jira", "confluence", "teamcity", "readyapi", "reqnroll", "playwright", "eggplant",
    "cucumber studio", "appium", "selenium", "git", "github", "gitlab", "bitbucket",
    "cli", "api", "vpn", "vdi", "rdp", "ec2", "aws", "azure", "docker", "kubernetes",
    "smoke test", "smoketest", "regression test", "release", "version", "environment",
    "repo", "repository", "escrow", "performance test", "load test", "uat", "e2e", "sbom",
}


# --------------------------------------------------------------------------- #
# candidate surfacing (deterministic)
# --------------------------------------------------------------------------- #
# A proper-noun phrase: a Capitalized word, optionally continued by more Capitalized words and the
# lowercase joiners (of/for/and/the/&) that hold multiword names together ("Transport for Wales",
# "Bank of England", "Barkha Kothapalli"). Single names ("Vix", "Samrat", "Manchester") match too.
_CAPWORD = r"[A-Z][A-Za-z0-9]*(?:'[A-Za-z]+)?"
_JOINER = r"(?:of|for|and|the|de|van|von|der|del|la|&)"
_PROPER_RE = re.compile(
    rf"\b{_CAPWORD}(?:[ \t]+(?:{_JOINER}[ \t]+)?{_CAPWORD})*\b")
# Acronyms / code-like names: ALLCAPS (CBE, AFC, UAF, E2E) and internal-caps (TfW, ReadyAPI, CamelCase).
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b|\b[A-Z]{2,}\b")
# Trailing joiner words a phrase may have greedily swallowed ("Vix and" -> "Vix").
_TRAIL_RE = re.compile(rf"\s+{_JOINER}$", re.I)


def _clean_phrase(p: str) -> str:
    p = _TRAIL_RE.sub("", p.strip())
    # also strip a leading joiner the regex kept ("The Vix" stays as-is only if 'The' is meaningful;
    # we drop a leading stopword token so "The Manchester Lab" -> "Manchester Lab").
    toks = p.split()
    while toks and toks[0].lower() in STOPWORDS:
        toks.pop(0)
    while toks and toks[-1].lower() in STOPWORDS:
        toks.pop()
    return " ".join(toks)


def _is_noise(surface: str) -> bool:
    n = normalize(surface)
    if not n or len(n) < 2:
        return True
    if n in STOPWORDS:
        return True
    # all tokens are stopwords, or it's a single very-common word
    toks = n.split()
    if all(t in STOPWORDS for t in toks):
        return True
    if n.isdigit():
        return True
    return False


def candidate_entities(text: str, extra_terms=None) -> list:
    """Surface candidate named entities from raw source text.

    Returns a list of {surface, norm, kind, count} dicts (deduped by norm, count = occurrences), where
    kind ∈ {'proper','acronym','term'}. `extra_terms` (e.g. the wiki's existing node titles/aliases)
    are matched case-insensitively as whole phrases so known entities are reliably re-found in new docs.
    """
    text = content_text(text)
    found: dict = {}

    def add(surface, kind):
        surface = surface.strip()
        if _is_noise(surface):
            return
        norm = normalize(surface)
        if not norm:
            return
        cur = found.get(norm)
        if cur is None:
            found[norm] = {"surface": surface, "norm": norm, "kind": kind, "count": 1}
        else:
            cur["count"] += 1
            # prefer a longer/more specific surface form for display
            if len(surface) > len(cur["surface"]):
                cur["surface"] = surface

    # 1) proper-noun phrases
    for m in _PROPER_RE.finditer(text):
        phrase = _clean_phrase(m.group(0))
        if not phrase:
            continue
        # a multi-word phrase is 'proper'; a single ALLCAPS/mixed-caps token is 'acronym'
        kind = "acronym" if (" " not in phrase and _ACRONYM_RE.fullmatch(phrase)) else "proper"
        add(phrase, kind)
    # 2) standalone acronyms the proper pass may have missed (e.g. inside lowercase context)
    for m in _ACRONYM_RE.finditer(text):
        add(m.group(0), "acronym")
    # 3) lexicon terms (seed ∪ caller-supplied), matched as whole words/phrases, case-insensitive
    terms = set(SEED_TERMS)
    if extra_terms:
        terms |= {normalize(t) for t in extra_terms if t}
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    low = re.sub(r"[-_/]+", " ", low)
    low = re.sub(r"[^a-z0-9'& ]+", " ", low)
    low = re.sub(r"\s+", " ", low)
    low = f" {low} "
    for term in terms:
        if not term or len(term) < 2 or term in STOPWORDS:   # don't let a stopword node-term ('an') leak
            continue
        n = low.count(f" {term} ")
        if n:
            found.setdefault(term, {"surface": term, "norm": term, "kind": "term", "count": 0})
            found[term]["count"] = max(found[term]["count"], n)
            if found[term]["kind"] == "proper":
                found[term]["kind"] = "term"
    return sorted(found.values(), key=lambda d: (-d["count"], d["norm"]))


def candidate_concepts(text: str, extra_terms=None) -> list:
    """Advisory concept candidates: the term-lexicon hits (domain/method words). Lighter than entities —
    concepts are harder to surface deterministically, so these inform the agent (WARN) rather than block.
    Returned shape matches candidate_entities for uniform handling."""
    cands = candidate_entities(text, extra_terms=extra_terms)
    return [c for c in cands if c["kind"] == "term"]


# --------------------------------------------------------------------------- #
# representation check — is a candidate already a node / alias / link target?
# --------------------------------------------------------------------------- #
def build_repr_index(strings) -> set:
    """Normalized set of everything the wiki currently *represents*: pass in note titles, aliases,
    filename stems, and `[[link]]` targets. A candidate is 'represented' iff its normal form is in here.
    Exact/alias matching (not loose substring) is deliberate: it means 'Vix' counts as represented only
    when a node is actually named/aliased 'Vix' — so a missing org node is caught even though 'vix-qa'
    exists, and aliases become the explicit, auditable signal that a surface form is covered."""
    idx = set()
    for s in strings or []:
        n = normalize(s)
        if n:
            idx.add(n)
    return idx


def terms_from_notes(titles_aliases) -> set:
    """The wiki's own entity surface forms, to seed recall in new docs (so once 'Jira' is a node, every
    future doc that says 'jira' re-finds it). Pass the same strings used for build_repr_index."""
    return {normalize(s) for s in (titles_aliases or []) if s and len(normalize(s)) >= 2}


def severity(cand) -> str:
    """How hard a *missing* candidate fails the gate. Every candidate that survives noise-filtering is a
    named thing the document refers to — a person/org/tool/place/term — and under the wiki's model
    ("anything a doc names gets a node") it must be accounted for: a node, a link, or an explicit skip.
    So missing candidates ERROR by default. The *stoplist* is the precision control (it drops generic
    capitalization), and the *skip ledger* is the relief valve for the rare genuinely-not-worth-a-node
    mention. Kept as a function so severity can be re-tuned in one place without touching the gate."""
    return "error"


def classify(candidates, repr_index, skips=None) -> dict:
    """Split candidates into represented / missing (with severity), honouring an explicit skip set
    (normalized surfaces the agent has consciously decided aren't node-worthy). Returns:
        {'represented': [...], 'missing_error': [...], 'missing_warn': [...], 'skipped': [...]}.
    """
    skips = {normalize(s) for s in (skips or [])}
    out = {"represented": [], "missing_error": [], "missing_warn": [], "skipped": []}
    for c in candidates:
        n = c.get("norm", "")
        if n in repr_index:
            out["represented"].append(c)
        elif n in skips:
            out["skipped"].append(c)
        elif severity(c) == "error":
            out["missing_error"].append(c)
        else:
            out["missing_warn"].append(c)
    return out
