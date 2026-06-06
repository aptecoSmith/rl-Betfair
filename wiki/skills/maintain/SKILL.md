---
name: maintain
description: Routine health pass - rebuild indexes, refresh source presence, run the gates.
---

# Skill: maintain

Use after batch ingests, on a periodic cadence, or when `doctor` reports drift.

## Procedure
1. `python scripts/wiki_tool.py doctor` - structure + counts.
2. `python scripts/wiki_tool.py build` - regenerate catalog + indexes + Sources blocks.
3. `python scripts/wiki_tool.py source-check` - refresh per-machine source presence; note new
   `missing` sources (expected when on a different machine than where a source lives).
4. `python scripts/wiki_tool.py validate` and `connectivity` - fix schema + graph findings.
5. `python scripts/wiki_tool.py project` - rebuild the query projection (also done by finalize).
6. `python scripts/audit_public.py` - safety gate.
7. Optionally run the `audit` skill (semantic) if it's due.
8. **Co-evolve the tool-system (v3 Phase G).** `python scripts/wiki_tool.py bootstrap propose` looks
   for places the *tooling* underperforms (coverage-floor misses, vocabulary friction, thin notes,
   orphans) and emits evidence-backed proposals to the schema / extract skill / vocabularies as
   `schema-change` events. Review (`dialog log --kind schema-change`) and `bootstrap accept|reject
   <id>` — accepting bumps the tool-system version. **Propose, you dispose; nothing auto-applies.**
9. Log anything meaningful with `python scripts/wiki_tool.py log`.

## Anti-patterns
Papering over a failing gate by deleting the check. Skipping `source-check` so dangling sources go
unnoticed. Running the gates but not acting on findings.
