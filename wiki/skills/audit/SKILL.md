---
name: audit
description: Semantic editorial pass - keep the wiki self-consistent and growing (distinct from schema validation).
---

# Skill: audit

The editorial heartbeat the original design called "lint" - NOT the schema validator. Run on a
cadence (e.g. weekly) or after a batch of ingests. This is judgement work, done by you, not a script.

## Procedure
Read across the wiki (use `search` and the indexes) and look for:
1. **Contradictions** - two notes that disagree. Reconcile or flag. **Start from the typed edges:**
   `python scripts/wiki_tool.py relations --contradictions` lists every `contradicts` edge
   deterministically — these are data, not something to re-infer from prose.
2. **Stale claims** - statements superseded by a newer source. `python scripts/wiki_tool.py relations
   --stale` lists every note (and its claims) made stale by a `supersedes` edge. Update, keeping the old
   as history if useful.
3. **Orphans / weak links** - run `python scripts/wiki_tool.py connectivity`; fix anything not
   reachable from a hub, and any non-reciprocated links.
4. **Missing concepts** - important ideas mentioned but lacking their own note. Create them.
5. **Coverage gaps** - sources that produced suspiciously little; re-run the `extract` skill.
6. **Source health** - `python scripts/wiki_tool.py source-check`; note anything now `missing`.
7. **New questions / sources** - propose what to investigate or ingest next.

Then **act** - create the missing notes, fix the links, log meaningful changes with
`python scripts/wiki_tool.py log`. Don't just enumerate problems; resolve them.

**Capture the reasoning (v3 Phase E).** When you find and resolve a contradiction, record it in the
CoDIAK dialog record so the reasoning doesn't evaporate:
`python scripts/wiki_tool.py dialog add --kind contradiction --text "<what disagreed>" --links <claim/note ids>`
then, once resolved, `dialog add --kind decision --text "<how it was resolved>" --links <contradiction-id>,<claim ids>`.
The trail is then reconstructable with `dialog trail <id>` (and `query dialog-trail --of <id>`).

## Anti-patterns
Treating `validate` (schema) as if it were this. Listing issues without fixing them. Asking the user
which obvious missing concept to create instead of creating it.
