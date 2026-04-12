# Lessons Learnt — Training End Summary Modal

## From discussion

- The current end-of-training UX is effectively invisible. The UI
  silently reverts to idle and dumps raw JSON. If the user stepped
  away during a multi-hour run, they'd have no idea it finished.

- The data for a good summary already exists in the `run_complete`
  event and `TrainingRunResult` — it just needs to be formatted and
  surfaced properly.

- Action buttons in the modal are important: the user's next step
  after training is almost always "look at the scoreboard" or "look
  at the best model". Making those one-click from the modal saves
  navigation.
