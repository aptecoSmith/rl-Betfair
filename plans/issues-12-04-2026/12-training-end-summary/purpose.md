# 12 — Training End Summary Modal

## What

When a training run finishes (or stops/errors), show a modal with a
formatted summary instead of silently reverting to the idle wizard with
raw JSON in a `<pre>` tag.

## Why

- Currently the UI transitions from "training in progress" to the
  wizard idle state without any fanfare. If you stepped away, you'd
  have no idea training finished unless you noticed the wizard was
  back.
- The summary is dumped as raw JSON (`{{ lastRunSummary() | json }}`)
  which is unreadable. The data includes useful info (generations
  completed, final rankings, run_id) but it's not formatted for
  humans.
- A modal gives a clear "training is done" moment with actionable
  next steps: view scoreboard, start another run, view best model.
