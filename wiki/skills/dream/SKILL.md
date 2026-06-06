---
name: dream
description: Time-bound/overnight run - generate interesting questions from the wiki's own graph, auto-answer the internal ones, and queue external research for approval.
---

# Skill: dream

The "llm-wiki dream": the wiki reflects on what it already knows and works out what it should want to
know next. Run it bounded (e.g. "dream for 30 minutes" or "dream overnight"). It is **proposal-safe**:
it auto-answers only questions it can settle from existing notes, and never touches the web without
your say-so.

## When to invoke
- "run the dream", "dream for <time>", "dream overnight".

## Procedure
1. **Propose.** `python scripts/dream.py propose --json` lists ranked candidates from the graph
   (it does NOT register them yet). Each is tagged:
   - **AUTO (internal)** - answerable by *synthesizing existing notes* (structural holes between
     clusters, an entity recurring across topics). Cheap, low hallucination risk.
   - **GATED (external)** - needs *new sources* (deepen a referenced stub, grow a thin topic).
2. **Prune the obvious ones (do this first, every run).** Read each candidate with its `provenance`
   and judge: *is the answer trivially obvious?* The classic case: an entity that "recurs across" notes
   only because it is the **parent source document** of those notes - that's not an insight, it's
   bookkeeping. Annotate each candidate with a `status`:
   - obvious/trivial -> `"status": "dismissed", "reason": "..."` (it won't be re-asked),
   - genuinely non-obvious -> `"status": "open"`.
   Then `python scripts/dream.py record --file judged.json`. Only `open` ones proceed.
3. **Budget.** Work down the surviving ranked list until the time/question/source budget is spent.
   Log what you skipped (resumable).
4. **Auto-answer the internal ones.** For each `open` AUTO question within budget:
   - Synthesize an answer purely from the cited notes (its `provenance`).
   - Write a new `synthesis` note as **`status: seed`** with a `dream` provenance marker, linked into
     a hub and to the notes it drew on.
   - Mark the register entry `answered`. Run `finalize-ingest`.
   - Seed status matters: these are machine-made and must earn promotion to `stable` at your review.
5. **Queue the external ones - DO NOT touch the web.** For each `open` GATED question, record what
   answering would need (search terms / candidate sources) and set the register entry `queued`. Stop there.
6. **Morning digest.** Write a `log` note: proposed N, pruned-as-obvious P, answered M (link them),
   queued K external for approval, budget spent. This + `git diff` is your review surface.

## Operator interaction
- **"go research 1 2 3"** - for those GATED items: scrape/search -> register sources -> run the
  `extract`/`ingest` skills -> synthesize the answer -> mark `answered`.
- **"dismiss 4 7"** - set those register entries `dismissed`; the dream won't re-ask them.

## Anti-patterns
- Touching the web for a GATED question without approval. Promoting `seed` dream-notes to `stable`
  without review. Re-asking `dismissed`/`answered` questions. Blowing the budget. Inventing a gap that
  isn't real - every question carries provenance; if the provenance doesn't support it, drop it.
