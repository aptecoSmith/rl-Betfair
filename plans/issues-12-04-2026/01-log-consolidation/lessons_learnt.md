# Lessons Learnt — Log Consolidation

## From discussion

- `bet_logs_dir` was originally co-located with the DB because `ModelStore`
  owns the parquet writes. But bet logs are evaluation output, not model
  artefacts — they belong with the other logs.
- The three production call sites all independently derive the same path
  (`db_path.parent / "bet_logs"`) rather than reading from config. Adding
  `paths.bet_logs` to config eliminates this duplication.
- Process logs are already viewable via the in-memory buffer in the admin
  UI, but there was no way to find the on-disk log files or know which
  subdirectories exist.
