# Tag vocabulary (controlled)

Tags are a **controlled vocabulary**, not free-form (D16/D25). `validate` rejects unknown tags.
Extend by adding to the lists here — the tool reads the fenced lists below as the source of truth.

## Note types (structural — exactly one per note)

<!-- types -->
- topic
- concept
- entity
- project
- synthesis
- log
- query
<!-- /types -->

## Entity subtypes (only on `type: entity`)

<!-- entity-subtypes -->
- person
- group
- org
- product
- tool
- place
<!-- /entity-subtypes -->

## Context tags (cross-cutting — zero or more per note)

<!-- context-tags -->
- work
- home
- research
- meetings
- lessons
- dream
- superseded
<!-- /context-tags -->

`superseded` is a **lifecycle marker**, not a context: it tags a note that a `supersedes` edge has
made stale, so the graph can colour stale knowledge distinctly. It is applied deterministically by
`scripts/tag_superseded.py` (run after ingest) — don't hand-add it; the script derives it from the
supersedes edges. The note stays readable as the pre-change reference.

## Privacy

- The `home` context tag and any note in the `personal` cloud are treated as **personal**:
  excluded from share/export and hideable from display.
- Add new context tags here as needs arise; keep them lowercase, singular, kebab-case.

## Mapping from earlier ideas

| Earlier idea | Lands as |
|---|---|
| People | `type: entity, subtype: person` |
| Groups | `type: entity, subtype: group` (or `org`) |
| Projects | `type: project` |
| Ideas | `type: concept` |
| Work / Home / Research / Meetings / Lessons | context tags |
