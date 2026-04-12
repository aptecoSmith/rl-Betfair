# Lessons Learnt — Training Plans Help Text

## From discussion

- The wizard page already has good help text patterns (`.help-text`,
  `.field-help`). The training plans editor was built in a later sprint
  and just never received the same treatment.
- The user is learning RL. Technical labels like `min_arch_samples`
  mean nothing without context. Every field needs: what it does, what
  good values are, and what goes wrong if you choose badly.
- Budget deserves special attention — models trained at one budget can
  behave very differently at another, and this isn't intuitive.
