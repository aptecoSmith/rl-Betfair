---
name: extract
description: Anti-shortcut knowledge extraction from a large or dense source (the 78-page-PDF problem).
---

# Skill: extract

Use when ingesting anything large or dense (syllabi, specs, manuals, long PDFs/DOCX). The failure
mode this prevents: reading 78 pages and emitting 5 notes. Producing knowledge proportional to the
content is the whole job.

## When to invoke
- A source spans many pages/sections, or is reference material (syllabus, standard, API docs).
- `finalize-ingest` reported an `under-extracted` or `unrepresented entities` coverage finding.

## Search the document first (`discover`) — don't rely on noticing
Before writing anything, run the deterministic entity/concept search:

    python scripts/wiki_tool.py discover --source <src-id>

It lists every person / org / tool / place / term the document **names**, split into "already in the
wiki (link these)" and "missing (node / link / skip)". This is the step the old flow lacked — reading a
doc and *noticing* what's in it is not the same as *searching* it, and that is exactly how Samrat,
Barkha and Vix were dropped into prose with no nodes.

The list is **candidates, not commands.** For each, decide:
- **node** — a person/org/tool/place/concept the wiki should hold → create the note (an entity gets a
  `subtype`; concept/entity notes need a grounded claim).
- **link** — it's already a node → `[[link]]` it from a note that cites this source.
- **skip** — not actually an entity (a verb/adverb like *make*, *hopefully*, or a throwaway mention) →
  `python scripts/wiki_tool.py entity-skip --source <id> --term "make, hopefully, Monday" --reason "not an entity"`
  (several terms at once, comma-separated).

You can't silently drop one: the **entity-coverage gate** blocks `finalize-ingest` until every candidate
is a node, a link, or a recorded skip. Division of labour — **code finds** the candidates (so nothing is
missed), **you judge** noun-vs-noise (what a model is reliably good at), the **gate checks** you judged
them all.

## Procedure
1. **Segment.** Run `python extractors/extract.py <file> --json` to get structure-aware segments
   (chapter/section/objective; `--granularity paragraph` for syllabus-grade density). For a very large
   or interruptible doc, add `--chunks-dir .runtime/chunks/<source-slug>` to persist each segment as a
   durable chunk file (plus an `index.md`) - then turn one chunk into notes at a time and see what's
   left across sessions.
2. **Process each segment independently.** For one segment at a time, enumerate *every*:
   - entity - create it with `type: entity` **and** the matching `subtype`
     (person/group/org/product/tool/place); the subtype is what keeps people, tools and products
     distinguishable in the graph and queries, so never leave an entity un-subtyped,
   - concept/idea/method,
   - learning objective / skill / metric / requirement (for a syllabus: every numbered objective
     `FL-1.2.1 (K2)` and every keyword in the chapter keyword list).
   **Name notes by the concept/objective/keyword, never by page** (`boundary-value-analysis`, not
   `p12`). Create or update a note for each. Do not read the whole doc then summarize.
3. **Record claims with provenance (v3 Phase B) — MANDATORY.** A note is a *composition of claims*; a
   substantive concept/entity note with **zero claims is incomplete** and the `coverage` gate now blocks
   it (`claimless` ERROR). For each
   assertion a note makes, record a **claim** to the note's sidecar `<note>.claims.jsonl`, carrying the
   `source_id`, a **locator** (the segment's page/section + the exact supporting **quote** you lifted),
   `asserted_by: model:<name>`, and a confidence. The fastest, safe way:
   `python scripts/wiki_tool.py claim-add --note <note> --source <src-id> --quote "<exact span from
   the segment>" --text "<the claim>" --asserted-by model:<name>`. **Division of labour:** you select
   the quote and write the claim; the tool *grounds* it — it refuses any quote not actually in the
   source, so you can never assert provenance you didn't ground.
4. **Justify empties.** If a segment yields zero notes, state why (genuinely contentless, e.g. a
   page number) - otherwise you skipped something.
5. **Dedup/merge.** Concepts recurring across segments get one note, linked - never duplicated.
6. **Link into a hub.** Every new note links to a topic/index hub and to related notes.
7. **Finalize.** `python scripts/wiki_tool.py finalize-ingest`. It runs `claims-lint` (every claim's
   locator must resolve to a real span in its source), the **substance-aware `coverage` gate**, and the
   **entity-coverage gate** (every entity the source names must be a node/link/alias in a citing note,
   or a recorded skip). An ERROR means "what did I skip?" — go back; don't reach for `--no-strict`.

## The notes/page floor is a FLOOR, not a target — never page-pad
≥1 note/page is a *smoke alarm* for under-extraction, **not the goal**. The goal is one note per
*concept/objective/keyword* — which is normally **more** than one per page on dense pages and **zero**
on the cover, table of contents, acknowledgements, references, and index. **Never** create a note for
a boilerplate page just to hit the floor, and never split one page into `p4`+`p5` "to make two notes":
that games the number without capturing knowledge. The `coverage` gate flags exactly this —
**page-named notes** (`p12`) and **thin stubs** — so padding is now visible and a strict gate can block
it (`coverage --strict`). If you genuinely under-extracted, re-extract at concept granularity; if a
page is genuinely contentless, leave it noteless and say so (step 4).

## Anti-patterns
- Skimming the whole document before writing. Summarizing instead of extracting. Lumping a chapter
  into one note. Leaving recurring concepts as duplicates. Ignoring a coverage warning.
