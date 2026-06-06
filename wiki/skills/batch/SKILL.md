---
name: batch
description: Drive a large corpus through ingest one source at a time, resumably, without bypassing the quality gates.
---

# Skill: batch

Use for big migrations - a folder of dozens or hundreds of docs that one session can't finish. It does
**not** bulk-auto-ingest (that is how you get 5 notes from a 78-page PDF). It keeps a resumable
work-queue and makes you take each source through the normal `extract`/`ingest` skills +
`finalize-ingest`, so the coverage floor and connectivity gates still apply.

The ledger lives in `.runtime/batch/<name>.json` (machine-local). A source is only counted `done`
once real notes cite it (`wiki_tool.coverage_map`), so progress reflects compiled knowledge, not files
touched.

> **rl-betfair:** the repo's markdown corpus is already planned into the ledger named **`repo-md`**
> (by `scripts/scan_repo.py`). Always pass `--name repo-md` to the commands below — without it they use
> the empty `default` ledger and report "nothing pending". You do **not** re-run `plan` for it.

## Loop
1. **Plan once:** `python scripts/batch.py plan "<folder>"` - registers every doc under the folder
   (reference-not-copy) and queues it. Re-running is safe; finished sources are never re-queued.
2. **Take the next source:** `python scripts/batch.py next` - prints one source and marks it
   in-progress. Add `--name <n>` to run several migrations side by side.
3. **Ingest it properly:** run the `ingest` skill on that source; for large/dense docs use `extract`
   (segment -> per-segment enumeration -> clear the >=1-note/page floor). Cross-link into a hub.
4. **Finalize:** `python scripts/wiki_tool.py finalize-ingest`. This is the gate.
5. **Confirm + repeat:** `python scripts/batch.py done <src-id>` (it refuses if no note cites the
   source yet), then back to step 2. `python scripts/batch.py status` shows % accounted and the next
   item.

## Resuming
Run `python scripts/batch.py status` (or `next`) in a fresh session - the ledger remembers what's
done, what's in flight, and what's left. `status`/`next` reconcile against the actual notes first, so
anything you ingested last session is auto-marked done, and a `done` source that somehow lost all its
notes is flagged and re-queued.

## When you must skip
`python scripts/batch.py skip <src-id> --reason "duplicate of ..."` - skipped sources stay accounted
for, never silently dropped. `requeue <src-id>` puts one back to pending.

## Anti-patterns
Mass-registering and then emitting a handful of notes for the whole batch. Marking `done` without
running `finalize-ingest`. Treating `plan` as ingestion - it only queues; compiling each source is
still your job.
