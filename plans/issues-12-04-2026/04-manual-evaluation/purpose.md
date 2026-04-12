# 04 — Manual Evaluation

## What

1. Add a standalone evaluation path — worker command + API endpoint —
   so models can be evaluated outside of a training run.
2. Add an Evaluation page to the frontend where the user can pick
   models, pick test days, and trigger evaluation with progress
   feedback.
3. Add re-evaluate actions to the model detail page and bulk select
   to the scoreboard.

## Why

- The Evaluator class is fully functional but locked inside the training
  loop. There's no way to re-evaluate a model on new data, re-run eval
  after a bug fix, or compare models on a specific set of days.
- The only ad-hoc evaluation mechanism is garaged model re-eval, which
  is automatic and uses the same test days as the training run.
- Common use cases that are impossible today:
  - "I have new race data — how do my best models perform on it?"
  - "I fixed the reward function — re-evaluate these 5 models"
  - "Compare these 3 models head-to-head on last week's races"
