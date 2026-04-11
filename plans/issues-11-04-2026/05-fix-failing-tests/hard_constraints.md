# Hard Constraints — Fix Failing Tests

1. **Do not delete tests to make the suite green.** Fix the root
   cause or add a proper skip condition with a reason string.
2. **Skipped tests must have a reason.** Every `pytest.mark.skip`
   or `pytest.mark.skipif` must document why.
3. **Timeout increases must be justified.** Don't set everything to
   10 minutes. Match the timeout to the expected runtime + margin.
4. **No changes to production code.** This plan is test-only.
5. **Every session updates `progress.md`.**
