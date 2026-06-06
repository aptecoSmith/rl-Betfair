---
name: curate
description: Classify a source by doc-type and extract it the way a human curator would for THAT type.
---

# Skill: curate

A meeting transcript, a 78-page syllabus, an API reference, and a README each deserve a different
extraction approach. Classify first, then follow the matching profile — don't extract everything the
same way (and never page-pad to hit a number). Profiles + coverage rules: `Schema/doc-types.md`.

## When to invoke
Whenever ingesting a source — *before* the `extract`/`ingest` skills. It chooses *how* to extract.

## Procedure
1. **Classify.** `python scripts/wiki_tool.py classify --source <src-id>` (add `--deep` to sniff
   content). Note the doc-type: `reference` · `api` · `software-docs` · `meeting-notes` · `general`.
2. **Apply the profile's extraction guidance** (`Schema/doc-types.md`). In short:
   - **reference (syllabus/standard/spec):** one note per **learning objective** and per **keyword**;
     name by concept; record the objective code (`FL-1.2.1`) as an **alias**. Many notes on dense
     pages, **zero on cover/ToC/index**. Then run the `extract` skill at paragraph granularity.
   - **api:** one note per **endpoint** (method+path) and per shared schema/parameter.
   - **software-docs:** per module/component/config-option/decision; capture commands + gotchas.
   - **meeting-notes:** the **decisions, action items, attendees** (as entities), topics, open
     questions. **Few notes is correct — do NOT pad.** Link to the project + people.
   - **general:** enumerate every entity/concept; name by concept.
   - **Every entity note gets a `subtype`** (person/group/org/product/tool/place) — that distinction is
     how people, tools and products stay separable in the graph and in `subtype:` queries; never leave an
     entity un-subtyped (a flat `type: entity` produces an undifferentiated graph).
3. **Record claims** with provenance as you go (see the `extract` skill: `claim-add`).
4. **Check coverage the right way for the type:**
   - reference → `python scripts/wiki_tool.py coverage --objectives --source <src-id>` — it lists any
     **missing learning objectives / keywords** (the doc's own enumerated targets). Fill the gaps;
     don't page-pad.
   - everything else → `coverage` (substance gate: flags page-padding and thin stubs).
5. **Finalize.** `python scripts/wiki_tool.py finalize-ingest` (runs the substance coverage gate).

## Anti-patterns
Extracting every doc the same way. Page-padding a syllabus (a note per page incl. cover/ToC) to hit
`notes/page` — that's gaming the proxy; the goal is a note per objective/keyword. Padding meeting notes
with stubs (few notes is fine). Skipping `classify`, so the wrong coverage rule is applied.
