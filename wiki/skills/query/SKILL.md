---
name: query
description: Answer a question from the compiled wiki, and file worthwhile answers back as pages.
---

# Skill: query

Use when the user asks something the wiki may cover.

## Procedure
1. **Index first:** read the active cloud's `index.md`.
2. **Search:** `python scripts/wiki_tool.py search "<question>" --semantic` (hybrid if the vector
   tier is on; BM25 otherwise).
3. **Structured questions go to the projection (v3 Phase D).** For precise, factual questions —
   "unverified claims from source X", "who is X's sibling", "what supersedes Z", "which sources are
   PDFs" — use `python scripts/wiki_tool.py query <name> [--from/--source/--type/--of …]`
   (`query --list` shows them; `--sql "SELECT …"` for ad-hoc). These run over the SQLite projection
   **with no LLM** — the answer is data, not inference. Run `project` first if the DB is stale.
4. **Open the top 1-3 notes** and answer primarily from compiled notes, citing the `src-id`s they
   carry. Open a raw source only if a note is a seed/incomplete or the user wants verification.
4. **Compound it.** If the answer is worth keeping (a comparison, a synthesis, a non-obvious
   conclusion), file it back as a `query` note (use `templates/query.md`), link it to the notes it
   drew on, and `finalize-ingest`. This is what makes the wiki compound instead of evaporating.
5. **File the question into the dialog record (v3 Phase E).** Capture the Q&A so the reasoning
   persists: `python scripts/wiki_tool.py dialog add --kind question --text "<the question + the
   answer in a line>" --links <the note/claim ids you drew on>`. The query→page loop is part of
   CoDIAK's dialog domain — recording it is how the *how* survives, not just the *what*.

## Anti-patterns
Jumping to raw sources before searching the wiki. Citing sources you didn't open. Letting a good
synthesis disappear into chat instead of filing it.
