# Link types (controlled edge vocabulary) — v3 Phase C

Typed edges give the graph **meaning**. A note's frontmatter carries directed typed links:

```yaml
links: [{to: 01J9Z3K8Q4R2N7B5C6D8E0F1G2, type: sibling-of}, {to: 01J9…, type: supports}]
```

`{to}` is the **id** of the target note (a bare ULID, or `id:<ULID>`; a filename stem also resolves via
the alias index). `{type}` must be one of the controlled types below — an **unknown type fails lint**.
Additions go *here* (and, once Phase G exists, through a logged proposal) — never ad hoc. Bare body
`[[wiki-links]]` remain **untyped `see-also`** for back-compat.

Edges are **data, not LLM inference**: "who is John's sibling" is a graph query over `sibling-of`
edges, not something the model re-derives from prose. The tool also **derives the inverse** (OHS
back-links), so you record an edge once and query it from either end.

## Epistemic edges (claims/notes relating as knowledge)

<!-- epistemic -->
- supports
- contradicts
- refines
- supersedes
- derived-from
<!-- /epistemic -->

## Structural edges (composition / reference)

<!-- structural -->
- part-of
- defines
- see-also
<!-- /structural -->

## Entity edges (people / orgs / places — the "John's sister" case)

Extensible — add new entity relations here as needs arise (keep them lowercase, kebab-case, and give
each an inverse below).

<!-- entity -->
- parent-of
- child-of
- sibling-of
- member-of
- works-for
- located-in
<!-- /entity -->

## Inverses

`A {to: B, type: T}` implies the back-link `B → A` of type `inverse(T)`. Connectivity honors these:
a `parent-of` is reciprocated by a `child-of`; symmetric types are their own inverse. The right-hand
side may be a *virtual* inverse (e.g. `supported-by`) that you don't author directly — it exists so the
back-link is derivable and a manually-mirrored edge can be checked for consistency.

<!-- inverses -->
- supports: supported-by
- contradicts: contradicts
- refines: refined-by
- supersedes: superseded-by
- derived-from: derives
- part-of: has-part
- defines: defined-by
- see-also: see-also
- parent-of: child-of
- child-of: parent-of
- sibling-of: sibling-of
- member-of: has-member
- works-for: employs
- located-in: contains
<!-- /inverses -->

## Audit surface

`supersedes` and `contradicts` feed the audit: a `supersedes` edge marks the older (superseded) note —
and its claims — **stale**; `contradicts` edges are listed for editorial resolution. See
`wiki_tool.py relations --stale` / `--contradictions`.
