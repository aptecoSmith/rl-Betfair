# Master TODO — Training Plans Help Text

## Session 1: Add help text + layout

### Layout decision

- [ ] Choose layout: inline help spans (like wizard) vs side-panel
      that shows context for the focused field. Side-panel is richer
      but more work. Inline is consistent with wizard. Could also do
      a hybrid: inline short help + expandable "learn more" that opens
      a side panel or tooltip.
- [ ] Whichever layout, keep it consistent with the wizard's existing
      `help-text` / `field-help` CSS classes

### Page-level introduction

- [ ] Add intro paragraph at top of editor explaining what a training
      plan is and how it differs from the quick-start wizard:
      "A training plan gives you full control over the hyperparameter
      search space. Use this when you want to systematically explore
      different configurations rather than using the defaults."

### Field-level help text

- [ ] **name** — "A descriptive name for this plan. Use something
      you'll recognise later, e.g. 'high-lr-lstm-exploration' or
      'conservative-budget-run'."

- [ ] **population_size** — "How many agents train simultaneously.
      Each starts with different random hyperparameters. Larger
      populations explore more of the search space but multiply
      training time linearly. 10-20 for quick experiments, 30-50
      for serious runs."

- [ ] **min_arch_samples** — "Minimum agents per architecture.
      Ensures each architecture gets a fair shot — if you select
      3 architectures with min_arch_samples=5, you need at least
      15 agents. Prevents the genetic algorithm from accidentally
      eliminating an architecture before it's had enough samples
      to show its potential."

- [ ] **seed** — "Random seed for reproducibility. Same seed +
      same settings = same initial population. Useful for
      comparing two plans that differ in one variable. Leave blank
      for a fresh random start each time."

- [ ] **budget per race** — "How much virtual money each agent
      starts with per race during training. Lower budgets (£10-20)
      force the agent to be selective. Higher budgets (£100+)
      let it explore more freely. The default is £100. Models
      trained at one budget can behave differently at another —
      a £10-trained model may be too cautious with £100."

- [ ] **exploration strategy** — expand the existing hint text:
      - **Random**: "Each agent's hyperparameters are sampled
        independently and uniformly. Simple and unbiased, but can
        leave gaps in the search space — especially with small
        populations."
      - **Sobol**: "Uses a quasi-random Sobol sequence to spread
        points evenly across the search space. Better coverage than
        random with the same number of agents. Best for first runs
        where you want broad exploration."
      - **Coverage-biased**: "Looks at all your previous training
        runs and targets the gaps — hyperparameter regions that
        haven't been explored yet. Best after 10+ prior runs when
        you want to fill in blind spots."
      - **Manual**: "You specify the exact hyperparameter values
        for the initial seed point. Other agents mutate from this
        starting point. Use when you have a specific configuration
        you want to test and explore around."

- [ ] **architectures** — "Which neural network architectures
      compete in this population. If you select multiple, agents
      with different architectures breed together — the genetic
      algorithm explores architecture choice as another variable.
      Select one for a focused search, multiple for broader
      exploration."

- [ ] **notes** — "Free-text notes for your future self. Record
      why you're running this plan, what you expect to learn,
      or what you changed from the last run."

### Section-level help text

- [ ] **Hyperparameter ranges** — "Each gene controls one aspect
      of the agent's behaviour or learning process. The ranges
      define the search space — wider ranges explore more but take
      longer to converge. Narrowing a range focuses the search on
      values you think are promising. Hover over a gene name to see
      what it controls."

- [ ] **Bias toward uncovered** — expand the button's hint:
      "Analyses your historical training runs and tightens ranges
      toward hyperparameter regions that haven't been explored yet.
      Review the adjusted ranges before saving — the bias is a
      suggestion, not a guarantee."

- [ ] **arch_lr_ranges** — "Override the learning rate range for
      specific architectures. Transformers typically need lower
      learning rates (1e-5 to 1e-4) than LSTMs (1e-4 to 1e-3).
      Without an override, all architectures share the same
      learning_rate gene range."

### Gene-level tooltips (if not already present)

- [ ] Check if `app-gene-editor` component already shows gene
      descriptions. If not, add a tooltip or subtitle showing the
      gene's description from the schema (already available in the
      gene spec data)

### Styling

- [ ] Help text should be visually distinct but not overwhelming —
      smaller font, muted colour, consistent with wizard's
      `.help-text` and `.field-help` classes
- [ ] Ensure help text doesn't break the form layout on narrow
      screens

### Verify

- [ ] `cd frontend && ng build` — clean
- [ ] Visual check: every field has help text, nothing looks cramped
