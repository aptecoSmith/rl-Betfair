# Hard Constraints

- `max_mutations_per_child: null` (default) must produce identical
  behaviour to current code.
- Architecture cooldown must still be respected — a cooled-down
  architecture gene is not eligible for selection.
- Backfill of missing HP keys (lines 692-693) must still happen for
  all genes regardless of cap.
- BreedingRecord.deltas must still be correct — mutated genes have
  their delta, unmutated genes have None.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
