"""claims.py — the claim model + per-note sidecar provenance (v3 Phase B). Stdlib only.

A **claim** is the atom of knowledge: a statement plus *where it came from*. This answers the
critique's strongest hit — "which sentence from which source produced this assertion." Claims live in a
per-note sidecar `<note>.claims.jsonl` (one JSON object per line) — **files are the source of truth**,
never a database. Phase D projects them for query; it never owns them.

Claim schema (one JSON object per line):
    id          ULID — stable, addressable identity
    text        the asserted statement (LLM-authored prose)
    source_id   src-... into Schema/sources.jsonl
    locator     where in the source: {page|pages|section|paragraph|slide|sheet|char_range|quote}
    asserted_by "model:<name>" (LLM) or "human"
    confidence  0..1 (optional)
    created     ISO-8601 date

**Division of labour** (plan principle 10): deterministic code GROUNDS a quote in the source text and
derives the locator (char range, page); the LLM writes `text` and SELECTS the supporting `quote` — it
never invents provenance. `ground_quote` refuses a quote that isn't actually in the source segment.

This module is pure: it operates on claim dicts and on source/segment *text* passed in by the caller.
Reading the source file (and extracting non-text formats) lives in wiki_tool, which keeps the
zero-dependency core working and feature-detects the optional extractors.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# locator keys that mean "where in the source"; at least one (truthy) is required
LOCATOR_KEYS = ("page", "pages", "section", "paragraph", "slide", "sheet", "char_range", "quote")
# provenance keys an extractor segment carries that are meaningful as a locator
_PROV_LOCATOR_KEYS = ("page", "pages", "section", "paragraph", "slide", "sheet")
VALID_ASSERTED = re.compile(r"^(human|model:[\w.\-]+)$")


# --------------------------------------------------------------------------- #
# text grounding (the deterministic half of the division of labour)
# --------------------------------------------------------------------------- #
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# Fold common typographic punctuation (smart quotes, dashes, ellipsis, nbsp) to ASCII, so a faithfully
# lifted quote isn't refused over cosmetic differences in PDFs / web clippings. Grounding stays honest:
# the content must still be present - only punctuation/whitespace noise is ignored, never missing text.
_PUNCT = {
    "“": '"', "”": '"', "„": '"', "″": '"',   # " " „ ″ -> "
    "‘": "'", "’": "'", "‚": "'", "′": "'",   # ' ' ‚ ′ -> '
    "–": "-", "—": "-", "−": "-",                   # – — − -> -
    "…": "...", " ": " ",                                # … -> ... ; nbsp -> space
}
_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT))


def _fold(s: str) -> str:
    return _norm_ws(_PUNCT_RE.sub(lambda m: _PUNCT[m.group()], s or ""))


def quote_in_text(quote: str, text: str) -> bool:
    """True iff `quote` occurs in `text`, ignoring whitespace wrapping AND typographic punctuation
    (smart quotes / dashes / ellipsis) - so a faithfully lifted quote from a PDF or clipping grounds,
    while a genuinely absent (paraphrased) quote is still refused."""
    q = _fold(quote)
    return bool(q) and q in _fold(text)


def raw_span(quote: str, text: str):
    """Exact byte-offset [start, end] of `quote` in `text`, or None if not an exact substring."""
    i = text.find(quote)
    return [i, i + len(quote)] if i != -1 else None


def ground_quote(source_text: str, quote: str):
    """Ground a selected quote in the source text. Returns a locator fragment, or None if ungrounded.

    The fragment always carries the normalized `quote`; it adds an exact `char_range` when the quote is
    a verbatim substring. None means the quote is NOT in the source — the caller must refuse it (the
    LLM tried to assert a span it did not ground).
    """
    if not quote_in_text(quote, source_text):
        return None
    frag = {"quote": quote.strip()}
    span = raw_span(quote.strip(), source_text)
    if span:
        frag["char_range"] = span
    return frag


def make_claim(text, source_id, quote, *, segment=None, asserted_by="human",
               confidence=None, created=None, claim_id=None):
    """Build a claim dict, grounding `quote` in `segment` (if given) or returning a quote-only locator.

    Raises ValueError if `quote` is not grounded in the segment text (ungrounded provenance). When
    `segment` is provided, the segment's structural provenance (page/section/...) is folded into the
    locator. The ULID is generated lazily from wiki_tool to avoid duplicating the generator.
    """
    locator: dict = {}
    if segment is not None:
        frag = ground_quote(segment.text, quote)
        if frag is None:
            raise ValueError("quote is not present in the source segment — cannot assert provenance")
        for k in _PROV_LOCATOR_KEYS:
            if k in segment.provenance:
                locator[k] = segment.provenance[k]
        locator.update(frag)
    else:
        locator["quote"] = quote.strip()
    if claim_id is None:
        import wiki_tool                                    # lazy: avoids an import cycle
        claim_id = wiki_tool.new_ulid()
    if created is None:
        import datetime as _dt
        created = _dt.date.today().isoformat()
    claim = {"id": claim_id, "text": text, "source_id": source_id, "locator": locator,
             "asserted_by": asserted_by, "created": created}
    if confidence is not None:
        claim["confidence"] = confidence
    return claim


# --------------------------------------------------------------------------- #
# sidecar I/O — <note>.claims.jsonl next to the note (committed; the source of truth)
# --------------------------------------------------------------------------- #
def claims_path(note_path) -> Path:
    p = Path(note_path)
    return p.with_name(p.stem + ".claims.jsonl")


def load_claims(note_path) -> list:
    p = claims_path(note_path)
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


def save_claims(note_path, claims) -> Path:
    p = claims_path(note_path)
    if not claims:
        if p.exists():
            p.unlink()
        return p
    lines = [json.dumps(c, ensure_ascii=False, sort_keys=True) for c in claims]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def append_claim(note_path, claim) -> Path:
    claims = load_claims(note_path)
    claims.append(claim)
    return save_claims(note_path, claims)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def has_locator(claim) -> bool:
    loc = claim.get("locator") or {}
    return any(loc.get(k) for k in LOCATOR_KEYS)


def validate_claim_structure(claim) -> list:
    """Structural problems with a claim, independent of the source text. Returns a list of messages."""
    problems = []
    import wiki_tool
    cid = claim.get("id", "")
    if not wiki_tool.is_ulid(str(cid).upper()):
        problems.append(f"claim id '{cid}' is not a valid ULID")
    if not str(claim.get("text", "")).strip():
        problems.append(f"claim '{cid}' has empty text")
    if not str(claim.get("source_id", "")).strip():
        problems.append(f"claim '{cid}' has no source_id")
    if not has_locator(claim):
        problems.append(f"claim '{cid}' has no locator (page/section/char-range/quote)")
    ab = str(claim.get("asserted_by", ""))
    if not VALID_ASSERTED.match(ab):
        problems.append(f"claim '{cid}' asserted_by '{ab}' invalid (want 'human' or 'model:<name>')")
    conf = claim.get("confidence")
    if conf is not None and not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
        problems.append(f"claim '{cid}' confidence '{conf}' out of range 0..1")
    return problems


# --------------------------------------------------------------------------- #
# agency marking (Phase F) — asserted_by (who claimed) vs verified_by (whether checked)
# --------------------------------------------------------------------------- #
VERIFIED_STATES = ("none", "human", "cross-source")


def is_unverified_llm(claim) -> bool:
    """True for a model-asserted claim that hasn't been verified — the hallucination-risk flag."""
    return (str(claim.get("asserted_by", "")).startswith("model:")
            and claim.get("verified_by", "none") not in ("human", "cross-source"))


def note_verification(claims_list) -> str:
    """Aggregate a note's verification from its claims: none | unverified | partial | verified."""
    if not claims_list:
        return "none"
    states = [c.get("verified_by", "none") for c in claims_list]
    verified = [s for s in states if s in ("human", "cross-source")]
    if len(verified) == len(states):
        return "verified"
    return "partial" if verified else "unverified"


def corroborating_claims(claim, all_claims) -> list:
    """Claims from a *different* source asserting the same statement (normalized `text` match).

    The deterministic basis for a `cross-source` promotion — verification by independent corroboration,
    not a rubber stamp.
    """
    txt = _norm_ws(claim.get("text", "")).lower()
    out = []
    for c in all_claims:
        if c.get("id") == claim.get("id") or c.get("source_id") == claim.get("source_id"):
            continue
        if txt and _norm_ws(c.get("text", "")).lower() == txt:
            out.append(c)
    return out


def set_claim_verified(note_path, claim_id, by) -> bool:
    """Set a claim's `verified_by` in its sidecar. Returns True if the claim was found and updated."""
    cs = load_claims(note_path)
    found = False
    for c in cs:
        if c.get("id") == claim_id:
            c["verified_by"] = by
            found = True
    if found:
        save_claims(note_path, cs)
    return found


def locator_resolves(claim, source_text: str):
    """Verify a claim's locator against the actual source text. Returns (status, reason).

    status ∈ {'ok', 'fail', 'unverified'}:
      ok         — the quote occurs in the source (or char_range is in-bounds).
      fail       — the quote is NOT in the source (a fabricated/ungrounded claim).
      unverified — only a weak locator (page/section) with nothing to check against the text.
    """
    loc = claim.get("locator") or {}
    quote = loc.get("quote")
    if quote:
        if quote_in_text(quote, source_text):
            return ("ok", "quote found in source")
        return ("fail", f"quote not found in source: {_norm_ws(quote)[:60]!r}")
    cr = loc.get("char_range")
    if cr and isinstance(cr, list) and len(cr) == 2:
        s, e = cr
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(source_text):
            return ("ok", "char_range in bounds")
        return ("fail", f"char_range {cr} out of bounds (len={len(source_text)})")
    return ("unverified", "no quote/char-range to verify against the source text")
