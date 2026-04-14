# UI Additions — Arb Improvements

Running list of UI / wizard / training-monitor work discovered
during each session. Session 8 consolidates everything in this file
into a single frontend pass. Each earlier session is responsible
for appending its UI tasks here before being marked complete.

The rule mirrors `plans/arch-exploration/ui_additions.md`: **no gene
or knob added anywhere in this plan is complete until the user can
change it from the UI**. Developer-editing-YAML is not a finished
state.

---

## Session 1 — Reward & advantage clipping

- [ ] Wizard: expose `reward.reward_clip`, `training.advantage_clip`,
      `training.value_loss_clip` in the constraints / parameters
      step. Default values of `0` render as "off". Help text
      explains they're safety nets to prevent training collapse
      from outlier races.
- [ ] Training monitor: per-episode display of
      `clipped_reward_total` alongside `total_reward` when the clip
      is active. Small badge ("clipped") next to the reward when
      `|clipped_reward_total - total_reward| > epsilon`.

## Session 2 — Entropy floor

- [ ] Wizard: expose `training.entropy_floor`. Help text:
      "minimum policy entropy before the entropy bonus is raised
      to encourage exploration". Default 0 = off.
- [ ] Training monitor: per-head entropy panel (signal, stake,
      aggression, cancel, arb_spread). Colour-code heads whose
      entropy is below the floor for > N batches ("collapsing"
      warning).

## Session 3 — Signal-bias warmup & bet-rate sparkline

- [ ] Wizard: expose `training.signal_bias_warmup` (epochs) and
      `training.signal_bias_magnitude`. Help text: "biases the
      agent toward placing bets during the first N epochs so it
      doesn't collapse into abstention before learning".
- [ ] Training monitor: bet-rate sparkline per agent across
      episodes. Arb-rate sparkline underneath. These are the
      two headline diagnostics — make them visible at a glance.

## Sessions 4 & 5 — Arb features

- [ ] No direct UI work. Features are always-on inputs. Optional:
      model-detail page shows the new feature names alongside
      other RUNNER_KEYS (automatic if the schema-driven UI is
      already reading the list).

## Session 6 — Oracle scan

- [ ] New admin action: "Scan days for arb oracle samples". Takes
      a date range, runs the scan, reports sample counts per day.
      Runs asynchronously — link to status via the existing
      progress infrastructure if possible.
- [ ] Scoreboard / model detail: show `oracle_density` (samples /
      ticks) per training day used by the model, so low-density
      days are visible.

## Session 7 — BC pretrainer

- [ ] Wizard: expose `training.bc_pretrain_steps` on the scalping
      step. Greyed out when scalping mode is off. Help text:
      "pretrain on every real arb moment in the training data
      before RL begins. Recommended 500–2000 for scalping; 0 for
      directional".
- [ ] Training monitor: show "BC warmup" phase indicator before
      PPO begins for agents that have `bc_pretrain_steps > 0`.
      BC loss curve logged alongside.

## Session 8 — Consolidation

- [ ] Work through every checkbox above. Any that are not yet
      ticked get implemented this session.
- [ ] Manual verification: run wizard end-to-end, flip every new
      knob, start a training run, confirm each knob's value
      appears in the `progress` event / run record.
- [ ] `ng build` clean.

## Session 9 — Aux head (optional)

- [ ] Wizard: expose `training.aux_arb_head` as a boolean toggle
      on the scalping step. Default off. Greyed out when scalping
      mode is off.
- [ ] Training monitor: aux loss curve alongside policy and value
      loss when aux head is enabled.

## Session 10 — Verification

- [ ] No UI work (analysis + documentation session).
