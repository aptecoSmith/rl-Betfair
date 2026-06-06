# Doc-type profiles — curate different sources differently

A human curator doesn't extract a meeting transcript the way they extract a 78-page syllabus. v3
codifies that: every source is **classified** into a doc-type (`wiki_tool.py classify`), and each type
carries its own **extraction guidance** (how the agent should turn it into notes) and its own
**coverage rule** (the deterministic gate that decides whether it was extracted *well* — instead of the
gameable `notes/page` proxy). Detection is by filename first, then a content sniff; `general` is the
fallback. The machine-readable profiles live in `scripts/doctypes.py`; this file is the human playbook.

## Profiles

### `reference` — syllabi, standards, specs, large knowledge PDFs (e.g. ISTQB)
- **Detect:** filename has *syllabus / ISTQB / standard / specification / handbook / overview /
  framework*, or the text has *Learning Objectives*, `Keywords:`, and `(K1)…(K3)` cognitive levels.
- **Extract:** one note per **learning objective** *and* per **keyword**. Name notes by the
  concept/keyword (`boundary-value-analysis`), **never by page**. Record the objective code
  (`FL-1.2.1`) as an **alias** so it's queryable. Dense pages → many notes; cover/ToC/index → none.
- **Coverage rule — `objective-keyword`** (`coverage --objectives`): the doc enumerates its own
  targets, so we check them exactly. Each objective code must appear in a citing note (body/alias);
  each keyword must have a matching note (slug/alias/title) or a body mention. Reports
  `objectives M/N`, `keywords P/Q`, and the **missing** list. This is the gate that replaces "≥1
  note/page" for reference docs.

### `api` — API / endpoint references, OpenAPI/Swagger
- **Detect:** filename has *api / swagger / openapi / endpoint / rest / reference*, or the text has
  `GET /`, `POST /`, *request body*, *response schema*, `application/json`.
- **Extract:** one note per **endpoint** (method + path) and per shared **schema/parameter**; name
  notes by the endpoint. Capture auth, status codes, request/response shapes.
- **Coverage rule — `endpoint`** (planned): note per `VERB /path` found in the source. (Currently uses
  the substance gate; the deterministic endpoint count is a clean next addition.)

### `software-docs` — READMEs, design docs, ADRs, runbooks, architecture
- **Detect:** filename has *readme / docs / design / architecture / adr / runbook / install*, or the
  text has `## Installation/Usage/Configuration/Architecture`, fenced code, *prerequisites*.
- **Extract:** per module/component, per config option, per concept/decision; capture commands and
  gotchas; link components to the system topic.
- **Coverage rule — `substance`**: no fixed enumerable target — rely on the substance/anti-padding gate.

### `meeting-notes` — transcripts, stand-ups, calls, KT sessions, weekly syncs
- **Detect:** filename has *transcript / meeting / stand-up / call / catchup / weekly / sync / KT /
  upskill / notes*, or the text has *Transcript / action item / attendees / minutes / discussed*.
- **Extract:** the **decisions**, **action items**, and **attendees** (as entity notes); capture topics
  and open questions. **Few notes is correct — do NOT pad.** Link to the project and the people.
- **Coverage rule — `substance`**: low note-count is expected; the gate only guards against thin
  stub-padding, not a page/objective floor.

### `general` — anything else (fallback)
- **Extract:** enumerate every entity/concept/method; name notes by concept; don't summarise or pad.
- **Coverage rule — `substance`**.

## Using it

```
wiki_tool.py classify                 # detected type per source (+ a by-type tally)
wiki_tool.py classify --deep          # also sniff content (slower; extracts)
wiki_tool.py coverage                  # substance gate (page-padding / thin stubs) — all types
wiki_tool.py coverage --objectives     # + objective/keyword coverage for reference docs (with missing lists)
wiki_tool.py coverage --objectives --strict   # exit non-zero if any reference target is uncovered
```

Adding a type or sharpening a rule goes through this file + `scripts/doctypes.py` (and, once you want
it tracked, a Phase-G `dialog` proposal). The `curate` skill (`skills/curate/SKILL.md`) tells the agent
to classify first, then follow the matching profile.
