# Discussion Session — Planning New Work

You are starting a discussion session for the rl-betfair project.
The operator wants to talk through ideas, issues, and features —
then turn them into actionable plan folders.

## Before anything else

Read these files to understand the project and what's already planned:

- `CLAUDE.md` — project overview and conventions
- `plans/issues-11-04-2026/order.md` — existing issue backlog and sprint order
- Browse `plans/issues-11-04-2026/*/progress.md` to see what's been completed

Check `C:\Users\jsmit\source\repos\ai-betfair\incoming/` for pending
cross-repo work.

## How discussion sessions work

The operator will raise issues, ideas, bugs, or feature requests
conversationally. For each one:

1. **Investigate first.** Read the relevant code before responding.
   Don't speculate — check what actually exists.

2. **Discuss the design.** Think through trade-offs, alternatives,
   edge cases. Push back if something won't work. Suggest better
   approaches if you see one.

3. **Create a numbered plan folder** when the discussion reaches a
   conclusion. The next available number follows on from the highest
   existing folder in `plans/issues-{date}/`. Each plan folder gets
   the full file set:
   - `purpose.md` — what and why
   - `master_todo.md` — ordered sessions with tick boxes and tests
   - `session_prompt.md` — standalone prompt for a new coding session
   - `hard_constraints.md` — non-negotiables
   - `progress.md` — empty, ready for session entries
   - `lessons_learnt.md` — seed with any analysis from the discussion
   - `knockon_ai_betfair.md` — if there's cross-repo impact, AND
     drop a note in `C:\Users\jsmit\source\repos\ai-betfair\incoming/`

4. **If it's a bug**, investigate it properly — read code, run
   queries, check data. Fix it if it's small. Plan it if it's big.

5. **If it's quick** (1 session, <30 min), consider just doing it
   rather than planning it.

## Conventions

- Commit message style: `fix:`, `feat:`, `plans:` prefix
- Push: `git push origin master`
- Tests: `python -m pytest tests/ --timeout=120 -q`
- Never kill GPU processes from bash
- Cross-repo knock-ons go in `ai-betfair/incoming/` as markdown files

## At the end of the session

- Update `order.md` if new issues were added
- Create sprint prompt files if needed
- Commit and push all plan files
