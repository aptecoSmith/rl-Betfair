# Lessons Learnt — Training Plans Integration

## From discussion

- The training plans system was built across Sessions 6-8 (exploration
  strategies, coverage dashboard, training plan UI) but the last-mile
  wiring to actually run a plan was never completed. The backend
  plumbing (orchestrator accepts plan, population manager consumes it)
  was done speculatively but never connected to the API layer.
- The save bug needs debugging — the wiring looks correct on paper
  (URLs match, plan_registry on app.state, error handling exists).
  Most likely cause is a validation error that's displayed but not
  prominent enough in the UI, or the API not running.
- A plan stores *what* to train (population shape, hp ranges, strategy)
  but not *how long* (generations, epochs). Adding these to the plan
  model makes plans self-contained and launchable with one click.
  Train/test dates are left out because they depend on data availability
  at launch time.
- Session splitting is the key quality-of-life feature for long runs.
  Without it, a crash at generation 4 of 5 means restarting from
  scratch. With it, each session is a checkpoint.
