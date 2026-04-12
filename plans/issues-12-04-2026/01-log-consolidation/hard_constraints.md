# Hard Constraints

- All existing bet log parquet files must be preserved — move, don't delete.
- Test fixtures use `tmp_path` and pass `bet_logs_dir` explicitly — they
  must not be affected by the default change.
- `ModelStore` must still accept an explicit `bet_logs_dir` parameter for
  test and script use.
- The API endpoint must not expose absolute paths beyond the project root
  (resolve relative to project root, return the absolute path).
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
