# Hard Constraints — Training Stop Options

Non-negotiables. Violation of any of these is rejected at review.

## Correctness

1. **"Stop immediately" must not corrupt state.** The current
   `stop_event` behaviour waits for the current agent to finish its
   step (training or eval).  This must remain true — never kill
   mid-write to the registry or mid-checkpoint-save.

2. **"Evaluate all" must evaluate every model in the current
   population.** If the user chose this option, skipping models is
   a bug.  The only exception is if a subsequent escalation to
   "immediate" overrides it.

3. **"Finish current eval" must produce valid results for that
   model.** The single model that was mid-evaluation must complete
   all its test days and get a valid score written to the registry.
   A partial evaluation (e.g. 3 of 9 test days) is not acceptable.

4. **Escalation is always allowed.** Once a stop granularity is
   chosen, the user can escalate to a more aggressive option
   (eval_all → eval_current → immediate).  De-escalation (immediate
   → eval_all) is not supported — you can't un-ring the bell.

## Backward compatibility

5. **`POST /training/stop` without granularity = immediate.**
   Existing callers (including any scripts or tests) must not break.

6. **`POST /training/finish` continues to work.** It maps to
   eval_all behaviour (complete current gen + evaluate).

7. **WebSocket event format unchanged.** New events can be added
   but existing event types must keep their shape.

## UX

8. **The dialog must show time estimates.** Choosing between options
   without knowing the cost is useless.  Estimates can be rough
   but must be present.

9. **No modal trap.** The dialog must have a Cancel button that
   closes it without sending any command.  The user might open it
   just to see the estimates.

## Scope

10. **No changes to the training loop itself.** This plan adds
    shutdown granularity, not training features.
11. **No changes to scoring or evaluation logic.** The evaluator
    runs the same way — we're just controlling *which* models get
    evaluated before shutdown.

## Documentation

12. **Every session updates `progress.md`.**
13. **Every surprising finding goes in `lessons_learnt.md`.**
