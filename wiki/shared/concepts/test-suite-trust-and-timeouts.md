---
id: 01KTG4S3AMNTD9X6494ZCHC5GM
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0dbc77]
aliases: [test suite trust, known failures, worker cold-start timeout]
---

# Test-suite trust + the timeout gotchas

Routinely skipping tests as "it's pre-existing" erodes trust in the suite — **real regressions can hide
behind known failures** — so the failing ones get fixed or explicitly conditioned, not ignored.

## What it is

The concrete rl-betfair causes were all environment/timeout, not logic: the **Windows worker cold-start
is slow** (the worker subprocess loads torch + initialises CUDA, exceeding the e2e test's 30s WebSocket
wait — fix with a longer timeout / retry-with-backoff / pre-warm); **integration tests run real model
inference** on full-sized observations and need a per-test timeout (e.g. `@pytest.mark.timeout(120)`),
not the unit-test 30s default; and **data-dependent tests** should skip on empty *columns*
(`timeform_comment`/`past_races_json` present-but-null), not merely on a missing table.

## Why it matters

A test left red "because it's known" is camouflage for the next real regression. Either fix it or make
the skip conditional and explicit (with a reason). For this stack specifically: integration/e2e tests
are slow by nature (CUDA init, real forward passes) — budget their timeouts separately from unit tests.

## Sources
- `src-0dbc77` purpose.md (js_desktop:present)
