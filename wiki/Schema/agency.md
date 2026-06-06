# Agency marking — who said it, and was it checked? (v3 Phase F)

Both Engelbart and the critique insist the machine **augments**; it is not the engine of truth ("the
LLM is a refreshener, not the curator"). So every claim records **agency**, and the wiki can tell a
machine assertion from a human-verified fact.

## Two independent axes (on each claim)

- **`asserted_by`** — *who made the claim*: `model:<name>` (an LLM) or `human`.
- **`verified_by`** — *whether it's been checked*: `none` (default) → `human` (a person reviewed it) or
  `cross-source` (independently corroborated by a claim from another source).

These are orthogonal: a human-asserted claim still starts `verified_by: none` (asserting ≠ verifying);
a model-asserted claim is only ever promoted by an explicit step — **nothing auto-verifies**.

The **flag** that matters: a claim that is `asserted_by: model:*` **and** still unverified is the
hallucination-risk case — surfaced by `wiki_tool.py query flagged-claims` and the Phase-H
`unverified-claims` view.

## Note-level aggregate

A note's verification (in the projection's `objects.verification`) is derived from its claims:
`none` (no claims) · `unverified` (no claim verified) · `partial` (some) · `verified` (all).

## Promotion

```
wiki_tool.py verify --claim <id> --by human          # a person vouches for it
wiki_tool.py verify --claim <id> --by cross-source   # allowed ONLY if another source corroborates
wiki_tool.py verify --note <path> --by human         # verify every claim in a note
```
- `cross-source` is a **deterministic rule**: it is refused unless an independent claim from a
  *different* source asserts the same statement (`claims.corroborating_claims`). Not a rubber stamp.
- Every promotion is logged to the CoDIAK dialog record (`Schema/dialog.jsonl`) as a `decision` event
  linked to the claim(s) — so the verification trail is itself recoverable.

## Querying / viewing

- `query verified-claims` / `query unverified-claims` / `query flagged-claims` (LLM off).
- Phase-H views render `verified-only` and `unverified-claims` from the projection.

## Signing (business tier — OFF by default)

GPG-signing verified claims is a **business-tier** feature (plan §8) and is **not built into the core
kit**. `verify --sign` is a documented placeholder that does **not** sign; enabling real signatures is
an explicit, separate, opt-in step (multi-user/ACL/crypto are out of scope for the personal kit).
