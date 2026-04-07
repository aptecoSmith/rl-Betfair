# Lessons Learnt — Next Steps

Anything surprising, counter-intuitive, or that would have saved
time if we'd known it earlier. Append at the end of every session.

Project-wide conventions (`CLAUDE.md`) take precedence over anything
recorded here — this file is for learnings that are too narrow or
too provisional to belong there yet.

`arch-exploration/lessons_learnt.md` is the frozen log from the
previous phase. Worth re-reading before starting anything in this
folder, especially:

- The "sampled ≠ used" rule (Session 0/1 entries)
- The gene-name → config-key mapping hazard (Session 1)
- Repairing genomes at two layers, not one (Session 3)
- `float_log` genes need log-space coverage buckets (Session 4)
- Option D (reflection-symmetric shaping) as a template for new
  closed-form reward terms (Session 7)

---

## 2026-04-07 — Backlog creation

- **The "out of scope" block of each session prompt is the best
  source of deferred-work candidates.** Not the PLAN.md, not TODO.md,
  not commit messages. The prompts were written at the moment a
  decision was made to defer something, with the reason attached.
  Keep this habit for future phases: every "Out of scope" or "Do
  not" line in a session prompt is a candidate for the next phase's
  backlog.

## 2026-04-07 — Session 10 (housekeeping sweep)

- **Assert against computed expected, never against a fresh
  hardcoded integer.** The `obs_dim` assertion went stale once
  already (1587 → 1630 → 1636) because someone kept "fixing" it by
  swapping in a new literal. The surviving assertion now only
  checks `pm.obs_dim == expected`, where `expected` is computed
  from the `DIM` constants exported by `env/betfair_env.py`. This
  is strictly weaker as a self-contained integer invariant, but
  strictly stronger as a drift detector — because it cannot lie
  about what the current layout is.

- **Pytest `addopts` filter does not stack with `-m` flags — it is
  replaced.** `pyproject.toml` has
  `addopts = "-m 'not integration and not gpu and not slow'"`, so
  running `pytest tests/ -m "not gpu and not slow"` **overrides the
  addopts `-m` entirely** and lets `integration`-marked tests in.
  One of them (`test_integration_session_1_5`) does a real training
  loop and hits the 60 s timeout. The correct way to run the fast
  suite is plain `pytest tests/` (let `addopts` do its job) or to
  explicitly include all three exclusions in `-m`. Noted in case a
  future session tries the same shortcut.

- **Schema inspector needs no maintenance when a new gene lands.**
  The `get_hyperparameter_schema` endpoint reads from
  `config.yaml → hyperparameters.search_ranges` directly and the
  frontend table is a pure projection over that list. A new gene
  in `config.yaml` auto-appears, a retired gene auto-disappears.
  This is the first time the auto-derived property actually
  mattered — the housekeeping sweep wanted to verify that every
  gene added in Sessions 3/5/6/7 is still in the view, and the
  answer was "yes, by construction, nothing to check".

- **`plans/arch-exploration/**` is frozen in the strictest sense.**
  The Session 10 exit criterion "grep for `observation_window_ticks`
  returns zero hits outside of `arch-exploration/lessons_learnt.md`"
  is impossible to satisfy literally — the phase's `master_todo.md`,
  `progress.md`, `purpose.md`, `session_1_reward_plumbing.md`,
  `session_6_transformer_arch.md`, and `ui_additions.md` all
  legitimately mention the retired name. The session prompt's
  "do not touch `plans/arch-exploration/`" rule wins over the
  narrower grep-zero exit criterion. Future housekeeping sweeps
  should read the exit criterion as "zero hits in *live* files"
  and ignore the frozen folder entirely.

- **`.claude/worktrees/` is scratch and should be ignored by
  audits.** Two worktrees (`elegant-fermi`, `magical-diffie`)
  hold historical snapshots with the retired names still in their
  vendored `PLAN.md` / `TODO.md` / `config.yaml` / `tests/`. These
  are disposable and are not part of the live repo. `.gitignore`
  already excludes them from commits; audits should exclude them
  too. (Not strictly a new lesson, but worth stating explicitly.)
