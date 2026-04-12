# 06 — Training Plans Help Text

## What

Add explanatory help text to every field and section in the training
plans editor. Follow the same pattern as the training wizard (which
already has `help-text` paragraphs and `field-help` spans under each
input). Consider a side-panel or inline help box layout so the
explanations don't push the form too far down the page.

## Why

- The training plans editor has bare technical labels
  (`population_size`, `min_arch_samples`, `seed`) with no explanation
  of what they do or why you'd change them.
- The user is learning reinforcement learning. Every field should
  explain: what it controls, what good values look like, and what
  happens if you get it wrong.
- The wizard page already does this well — the training plans page
  just never got the same treatment.
