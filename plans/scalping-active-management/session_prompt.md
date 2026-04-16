# Scalping Active Management — current session pointer

This file is a pointer to whichever session is **next** in
the plan. The full set of per-session prompts lives in
[`session_prompts/`](./session_prompts/) — one file per
session.

## Next session

**Session 03 — Risk / predicted-variance head.**

See [`session_prompts/03_risk_head.md`](./session_prompts/03_risk_head.md)
for the full brief.

Quick context: Session 02 landed the fill-probability aux
head (plumbing-off at `fill_prob_loss_weight=0.0`). Session
03 lands the second aux head — per-runner Gaussian NLL on
locked-P&L — structurally parallel to Session 02. Same
capture→attach flow, same migration-helper pattern, same
plumbing-off default. After Session 03 both aux heads will
exist as plumbing-off, and the
[`activation_playbook.md`](./activation_playbook.md)
becomes the next thing to run to turn them on.

## All sessions in this plan

| # | Title | Status | Prompt |
|---|---|---|---|
| 01 | Re-quote action + env plumbing | ✅ done | _no prompt archived — see `progress.md` for what landed_ |
| 02 | Fill-probability head | ✅ done | [`session_prompts/02_fill_prob_head.md`](./session_prompts/02_fill_prob_head.md) |
| 03 | Risk / predicted-variance head | ⏳ next | [`session_prompts/03_risk_head.md`](./session_prompts/03_risk_head.md) |
| 04 | Bet Explorer confidence + risk badges | pending | [`session_prompts/04_bet_explorer_badges.md`](./session_prompts/04_bet_explorer_badges.md) |
| 05 | Model-detail calibration card | pending | [`session_prompts/05_calibration_card.md`](./session_prompts/05_calibration_card.md) |
| 06 | Scoreboard MACE column | pending | [`session_prompts/06_scoreboard_mace_column.md`](./session_prompts/06_scoreboard_mace_column.md) |
| 07 | Training run + analysis | pending | [`session_prompts/07_validation_run.md`](./session_prompts/07_validation_run.md) |

Between Session 03 and Session 04, run
[`activation_playbook.md`](./activation_playbook.md) to turn
the aux-head weights up and promote them into the master
config.

## How to update this file

When a session lands:

1. Tick its box in `master_todo.md`.
2. Append its progress entry to `progress.md`.
3. Update the **Next session** section of THIS file to
   point at the next pending session's prompt.
4. Flip its row in the table above (`⏳ next` → `✅ done`,
   and the next row's status to `⏳ next`).

If a session adds a procedure that isn't a code change
(like the activation playbook), link it from the table
and from the appropriate prompt's "Before you start" list.
