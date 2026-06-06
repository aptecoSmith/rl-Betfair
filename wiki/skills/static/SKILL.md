---
name: static
description: Ingest an append-only note (running meeting notes) that lives in inbox/static/ - snapshot, never move, and only ingest what's new.
---

# Skill: static

For a note you keep reopening and appending to - classically **running meeting notes** in one `.txt`.
A normal inbox drop gets moved to `processed/` after ingest, which is wrong here: you'd lose the file
you keep editing. A **static** source is referenced in place and snapshotted instead.

## When to invoke
- The user saved/updated a file in `inbox/static/` (or `inbox_personal/static/`) and wants it ingested.
- The opener reported a `static:` file.

## Procedure
1. **Register if new:** `python scripts/wiki_tool.py register --path "inbox/static/<file>"`. The src-id
   is derived from the path, so it's stable every time you re-ingest the same growing file.
2. **Find what's new.** Look in `inbox*/processed/static/` for the most recent snapshot of this file
   (`<stem>-<date>...`). Diff the current file against it and work **only on the newly-added text**
   (the latest meeting). If there is no prior snapshot, treat the whole file as new.
3. **Ingest just the new content.** Extend existing notes where the new entry continues a topic;
   create new entity/concept/log notes for genuinely new people, decisions, actions. Don't re-mint
   notes for older entries already captured.
4. **Cross-link + lift quotes** as usual; a dated `log` note per meeting is often the right home.
5. **Finalize:** `python scripts/wiki_tool.py finalize-ingest`. This writes a fresh dated snapshot to
   `inbox*/processed/static/` (copy, not move) **only if the file changed**, and leaves your original
   in `inbox/static/` so you can keep appending.

## Anti-patterns
- Moving the file out of `inbox/static/` (it must stay for the next entry).
- Re-ingesting the entire file every time and re-creating notes for old meetings - diff first.
- Snapshotting on every finalize even when nothing changed (the tool already skips unchanged files).
