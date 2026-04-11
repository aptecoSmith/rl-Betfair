# Hard Constraints — Training Log Auto-Scroll

1. **Manual scroll must not be overridden.** If the user scrolls up
   to read history, auto-scroll must not yank them back down.
2. **Default is on.** New page load → auto-scroll enabled.
3. **No performance regression.** Scrolling should not trigger
   excessive change detection. Use `effect()` or
   `afterNextRender()`, not `AfterViewChecked` polling.
4. **Scope: activity log only.** No other UI changes.
