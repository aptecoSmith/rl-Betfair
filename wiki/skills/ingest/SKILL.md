---
name: ingest
description: Turn one registered source into compiled, linked, sourced wiki notes.
---

# Skill: ingest

Use when the user says "ingest X" or confirms pending intake. One source at a time.

## Procedure
1. **Register** the source if not already: `python scripts/wiki_tool.py register --path <p>` (or
   `--url`). For folders/repos use `python scripts/intake.py folder <dir>`. Reference-not-copy:
   we record the location, we never copy bytes.
2. **Read it fully — one source at a time.** Large/dense -> use the `extract` skill. Never sample or
   skim, and finish this source through `finalize-ingest` before starting another (the `batch` lock
   enforces one-at-a-time, so you don't read many together and summarise them).
3. **Discover (search the doc):** `python scripts/wiki_tool.py discover --source <id>` lists every
   entity/concept the source NAMES — your worklist, so nothing is missed by not noticing. For each
   candidate: a **node**, a `[[link]]` to an existing one, or
   `entity-skip --source <id> --term "make, hopefully" --reason "not an entity"` (several at once).
4. **Search before creating:** `python scripts/wiki_tool.py search "<keywords>" --semantic` across
   active cloud + shared, so you extend/link existing notes instead of duplicating them.
5. **Write breadth.** A node for every named person/group/org/product/tool/place (entity + `subtype`)
   and every distinct concept; Topics/Projects where warranted (`templates/`). Do **not** fold several
   sources into one "synthesis" grab-bag — that aggregation is what the entity/coverage gates reject.
6. **Lift key quotes + ground claims** into the notes so the citation survives if the source later goes
   missing (a substantive concept/entity note with no grounded claim is incomplete — `claim-add`).
7. **Cross-link** every new note to a hub and to related notes via `[[wiki-links]]`. Personal notes
   may link up into shared; shared must not depend on personal.
8. **Set frontmatter:** correct `type`/`cloud`/`status`, ISO dates, controlled `tags`, `sources`
   listing the `src-id`(s).
9. **Record what you rejected (v3 Phase E).** If you decline a source (duplicate, unreliable, out of
   scope) or drop a claim, log *why* to the dialog record so the decision is recoverable:
   `python scripts/wiki_tool.py dialog add --kind rejection --text "<source/claim and why rejected>"`.
10. **Finalize:** `python scripts/wiki_tool.py finalize-ingest` (strict by default — validate,
    connectivity, claims-lint, coverage, **entity-coverage**). Not done until it passes. Finalize also
    auto-moves any dropped inbox file a note now cites into `inbox*/processed/` and updates the registry.

## Anti-patterns
Copying source bytes in. Inventing citations. Under-extracting. **Folding several sources into one
grab-bag, or reading many together and summarising.** Isolated notes. Free-form tags. Calling it done
before `finalize-ingest` is green.
