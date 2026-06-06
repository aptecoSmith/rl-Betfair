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
2. **Read it fully.** Large/dense -> use the `extract` skill. Never sample or skim.
3. **Search first:** `python scripts/wiki_tool.py search "<keywords>" --semantic` across active
   cloud + shared, so you extend existing notes instead of duplicating them.
4. **Write breadth.** A new Entity for every named person/group/org/product/tool/place; a new
   Concept for every distinct idea; Topics/Projects where warranted. Use the `templates/`.
5. **Lift key quotes into the notes** so the citation survives even if the source later goes missing.
6. **Cross-link** every new note to a hub and to related notes via `[[wiki-links]]`. Personal notes
   may link up into shared; shared must not depend on personal.
7. **Set frontmatter:** correct `type`/`cloud`/`status`, ISO dates, controlled `tags`, `sources`
   listing the `src-id`(s). Record claims with provenance (see the `extract` skill: `claim-add`).
8. **Record what you rejected (v3 Phase E).** If you decline a source (duplicate, unreliable, out of
   scope) or drop a claim, log *why* to the dialog record so the decision is recoverable:
   `python scripts/wiki_tool.py dialog add --kind rejection --text "<source/claim and why rejected>"`.
9. **Finalize:** `python scripts/wiki_tool.py finalize-ingest`. Not done until it passes. Finalize
   also auto-moves any dropped inbox file that a note now cites into `inbox*/processed/` (gitignored)
   and updates the registry - no manual move step.

## Anti-patterns
Copying source bytes in. Inventing citations. Under-extracting. Isolated notes. Free-form tags.
Calling it done before `finalize-ingest` is green.
