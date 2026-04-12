# Hard Constraints

- Modal must auto-open when run_complete event arrives.
- Modal must be dismissable without navigating away from the page.
- The enriched summary must be backward compatible — if the backend
  sends the old minimal event, the frontend should still work (show
  what it can, skip what's missing).
- Raw JSON dump must be removed — replaced with either the modal or
  the compact idle-state card.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
