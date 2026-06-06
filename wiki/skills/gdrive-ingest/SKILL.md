---
name: gdrive-ingest
description: Ingest a Google Drive folder recursively via the plugged-in agent's Drive connector.
---

# Skill: gdrive-ingest

Use when the user points at a Google Drive folder. There are no OAuth creds in this repo - Drive
access comes from *your* connector (the plugged-in agent's), keeping the zip clean.

## Two paths

**A. Connector path (no local sync).** If you have a Google Drive tool/connector:
1. List the folder recursively via the connector.
2. For each doc, fetch its content (export Google Docs as text/markdown).
3. Drop each into `inbox/pending/Files/` (or `inbox_personal/` if personal), then run the `ingest`
   skill per file. Register the Drive URL as the source location so it stays traceable.

**B. Synced-folder path (no connector).** If the user has Drive synced to a local folder:
1. `python scripts/intake.py folder "<local-drive-folder>"` - registers every doc in place
   (reference-not-copy), recursively.
2. Run the `ingest` / `extract` skills per registered source.

## Notes
- Tag everything `personal` (and use the personal cloud) if the folder is personal.
- Large Google Docs/PDFs go through the `extract` skill - don't shortcut them.
