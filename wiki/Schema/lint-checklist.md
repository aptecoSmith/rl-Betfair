# Lint / validation checklist

What the deterministic gate checks. `finalize-ingest` runs these in order; the agent should also
run them after edits. `validate`/`connectivity` warn by default; `--strict` makes violations fail (D29).

## `validate` (schema)
- [ ] Frontmatter parses (stdlib subset, see frontmatter-contract.md).
- [ ] `id` present (warn → run `stamp-ids`); present ids are valid ULIDs and unique across the corpus.
- [ ] `type` present and in the note-type set; exactly one.
- [ ] `subtype` only on entities, from the entity-subtype set.
- [ ] `cloud` ∈ {shared, personal} and matches the note's folder.
- [ ] `status` ∈ {seed, draft, stable}.
- [ ] draft/stable ⇒ `sources` non-empty.
- [ ] `created`/`updated` are ISO-8601 dates.
- [ ] all `tags` are in the controlled context-tag vocabulary.
- [ ] every id in `sources` exists in `Schema/sources.jsonl`.

## `connectivity` (graph / anti-orphan)
- [ ] The link graph is built on **stable ids**, not filenames — renames never break `[[id:...]]` links.
- [ ] All `[[wiki-links]]` resolve to existing notes; an ambiguous `[[stem]]` (shared by ≥2 notes) is
      flagged (warn) — disambiguate with `[[id:...]]`.
- [ ] **Typed edges** (`links: [{to, type}]`, Phase C) count for reachability; a typed link to a
      missing id ⇒ dangling (error); an inverse mismatch (A `parent-of` B, B not `child-of` A) ⇒ warn.
      An **unknown edge type** fails `validate` (error) — additions go through `Schema/link-types.md`.
- [ ] Links are bidirectional (A↔B). Hubs (`index`) are exempt as source: they list every
      leaf by design, so the hub→leaf direction does not require leaves to back-link the hub.
- [ ] Every note is reachable from its cloud hub (root index + topic hubs). Unreachable note or
      cluster ⇒ orphan finding.
- [ ] No `shared` note depends on a `personal` note (cloud direction).

## `claims-lint` (claim-level provenance, v3 Phase B)
- [ ] Each claim in a `<note>.claims.jsonl` sidecar is structurally valid (ULID id, text, source_id,
      a locator, valid `asserted_by`).
- [ ] The claim's `source_id` is registered in `Schema/sources.jsonl`.
- [ ] The locator **resolves to a real span** in the source: the lifted `quote` is found in the source
      text (or `char_range` is in-bounds). A quote not found ⇒ **error** (fabrication guard). Source
      missing on this machine ⇒ warn (the lifted quote can't be re-checked here). `--strict` blocks.
- [ ] Claim ids are unique across sidecars.
- [ ] **Agency (Phase F):** each claim carries `asserted_by` (model/human) and `verified_by`
      (none/human/cross-source). Nothing auto-verifies; `cross-source` requires an independent
      corroborating claim. Model-asserted + unverified claims are flagged (`query flagged-claims`).

## `source-check` (registry)
- [ ] Re-resolve each source location on this machine; update `present`/`last_seen`.
- [ ] Notes citing a source absent on all machines are flagged `source_status: missing` (warning).

## `audit_public` (safety gate)
- [ ] No secrets (AWS/GitHub/private keys, inline passwords) in tracked text.
- [ ] No machine-local absolute paths committed outside the registry.

## `coverage` (anti-shortcut + anti-gaming, large docs) — `wiki_tool.py coverage`
- [ ] **Substance, not page count.** notes/page is a floor, not a target, and is gameable. The gate
      flags the two real failure modes per source: **page-named notes** (`p12` — page-padding instead
      of concept/objective/keyword extraction) and a high share of **thin stubs** (<200 prose chars).
- [ ] Run in `finalize-ingest` (advisory) and standalone (`coverage [--source] [--strict]`).
      `--strict` exits non-zero on any flag. The cure for under-extraction is concept-granular
      re-extraction (see `skills/extract`), **not** adding boilerplate/page notes to hit a number.
