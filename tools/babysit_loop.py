"""Babysit the rl-betfair experiment loop.

Designed to run hourly via Windows Task Scheduler. Each invocation is
ONE iteration that:

1. Checks current GPU state (any cohort.runner running?)
2. Identifies completed cells since last invocation
3. Pulls metrics, identifies leader, appends to monitoring_notes.md
4. If GPU is idle AND there's a next round queued: launches it
5. Stops cleanly when budget exhausted or deploy candidate confirmed

State persisted to ``plans/recipe-expansion-and-robustness/babysit_state.json``
so successive invocations build on each other.

Run manually: ``python tools/babysit_loop.py``
Run via task scheduler: see ``tools/install_babysit_task.ps1``.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN_DIR = REPO / "plans" / "recipe-expansion-and-robustness"
STATE_FILE = PLAN_DIR / "babysit_state.json"
MONITOR_LOG = PLAN_DIR / "monitoring_notes.md"
BABYSIT_LOG = PLAN_DIR / "babysit_log.txt"

# Ordered list of rounds to launch when GPU goes idle. Each entry is
# the wrapper path relative to repo root. The script consumes them
# in order and skips entries whose wrapper log already says "fan-out
# complete". To add a new round: write the wrapper script, then
# append its path here.
ROUND_QUEUE = [
    "plans/recipe-expansion-and-robustness/run_round6.sh",
    "plans/recipe-expansion-and-robustness/run_round6_5.sh",
    "plans/recipe-expansion-and-robustness/run_round7.sh",
    # 2026-05-27 06:50 BST REORDER: Round 6.5 CONFIRMED fc=0 unlocks
    # positive day_pnl (20/20 agents positive, mean +£200). Round 9
    # (fc=0 sweep) is now higher priority than Round 8 (which still
    # uses fc=120 throughout). Round 8 deprioritized.
    "plans/recipe-expansion-and-robustness/run_round9.sh",
    "plans/recipe-expansion-and-robustness/run_round10.sh",
    "plans/recipe-expansion-and-robustness/run_round11.sh",
    # Round 8 last — uses fc=120 which is known-suboptimal now.
    "plans/recipe-expansion-and-robustness/run_round8.sh",
]

# Acceptance criteria — count how many a cell passes.
ACCEPTANCE = {
    "opens_min": 100,
    "opens_max": 180,
    "mat_pct_min": 5.0,
    "fc_pct_max": 50.0,
    "day_pnl_min": -100.0,
    "locked_over_sigma_naked_min": 0.5,
}


def log(msg: str) -> None:
    """Append to babysit_log.txt with timestamp."""
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line)
    BABYSIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with BABYSIT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "last_analyzed_cells": [],
        "rounds_launched": [],
        "stopped": False,
        "stop_reason": "",
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def _process_cmdlines() -> list[str]:
    """Yield every running process's command line as a single string.

    Uses psutil for portability — `ps` isn't on PATH under Windows
    Task Scheduler's restricted environment.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        log("psutil not available — process probes will be unreliable")
        return []
    out: list[str] = []
    for proc in psutil.process_iter(attrs=["cmdline", "name"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            out.append(" ".join(cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def cohort_running() -> bool:
    """True if any cohort.runner python process is alive."""
    for cmd in _process_cmdlines():
        if "cohort.runner" in cmd:
            return True
    return False


def wrapper_running(wrapper_path: str) -> bool:
    """True if a bash process is currently executing this wrapper.

    Strict match: the cmdline must consist of exactly ``bash <path>``
    (possibly with the full bash path). Loose matches against the
    shell-snapshot eval strings in Claude Code's own bash sessions
    were producing false positives — those contain wrapper paths in
    their argument lists but aren't actually running the wrappers.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return False
    name = Path(wrapper_path).name
    for proc in psutil.process_iter(attrs=["cmdline", "name"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if len(cmdline) < 2:
                continue
            proc_name = (proc.info.get("name") or "").lower()
            if "bash" not in proc_name and "bash" not in cmdline[0].lower():
                continue
            # Look for `bash <wrapper>` — the script path must appear
            # as its own argv element (token), not embedded inside
            # another arg like a snapshot-script's eval string.
            for tok in cmdline[1:]:
                if tok.endswith(name) or tok.endswith(name.replace("\\", "/")):
                    return True
        except Exception:
            continue
    return False


def wrapper_completed(wrapper_path: str) -> bool:
    """True if wrapper's log contains a fan-out complete sentinel."""
    wrapper = Path(wrapper_path)
    log_name = wrapper.name.replace("run_", "_").replace(".sh", "_wrapper.log")
    log_path = REPO / "registry" / log_name
    if not log_path.exists():
        return False
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return "fan-out complete" in txt


def all_completed_cells() -> list[Path]:
    """Return a list of cell-output dirs from all round wrappers
    that have a scoreboard.jsonl file (i.e., finished)."""
    cells: list[Path] = []
    reg = REPO / "registry"
    for d in reg.glob("_round*_*"):
        if d.is_dir() and (d / "scoreboard.jsonl").exists():
            cells.append(d)
    return sorted(cells)


def cell_metrics(cell_dir: Path) -> dict | None:
    """Compute per-cell aggregate metrics from scoreboard.jsonl."""
    sb = cell_dir / "scoreboard.jsonl"
    try:
        rows = [json.loads(l) for l in sb.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return None
    if not rows:
        return None
    n = len(rows)

    def avg(k):
        return sum((r.get(k) or 0) for r in rows) / n

    pairs = avg("eval_pairs_opened") or 0
    mat = 100 * avg("eval_arbs_completed") / pairs if pairs else 0
    cls_ = 100 * avg("eval_arbs_closed") / pairs if pairs else 0
    fc = 100 * avg("eval_arbs_force_closed") / pairs if pairs else 0
    locked = [r.get("eval_locked_pnl", 0) for r in rows]
    naked = [r.get("eval_naked_pnl", 0) for r in rows]
    sn = statistics.stdev(naked) if len(naked) > 1 else 0.0
    locked_over_sn = (sum(locked) / n / sn) if sn > 0 else float("inf")
    return {
        "cell": cell_dir.name,
        "n_agents": n,
        "day_pnl": avg("eval_day_pnl"),
        "locked_pnl": avg("eval_locked_pnl"),
        "naked_pnl": avg("eval_naked_pnl"),
        "force_closed_pnl": avg("eval_force_closed_pnl"),
        "closed_pnl": avg("eval_closed_pnl"),
        "bets": avg("eval_bet_count"),
        "opens": pairs,
        "mat_pct": mat,
        "cls_pct": cls_,
        "fc_pct": fc,
        "locked_over_sigma_naked": locked_over_sn,
        "passes": acceptance_passes(
            pairs, mat, fc, avg("eval_day_pnl"), locked_over_sn,
        ),
    }


def acceptance_passes(opens, mat_pct, fc_pct, day_pnl, lsn):
    """Return number of acceptance criteria the cell passes."""
    return sum([
        ACCEPTANCE["opens_min"] <= opens <= ACCEPTANCE["opens_max"],
        mat_pct >= ACCEPTANCE["mat_pct_min"],
        fc_pct <= ACCEPTANCE["fc_pct_max"],
        day_pnl >= ACCEPTANCE["day_pnl_min"],
        lsn >= ACCEPTANCE["locked_over_sigma_naked_min"],
    ])


def format_metric_row(m: dict) -> str:
    return (
        f"{m['cell']:<50} "
        f"pnl={m['day_pnl']:+7.1f} "
        f"locked={m['locked_pnl']:+5.1f} "
        f"opens={m['opens']:>4.0f} "
        f"mat={m['mat_pct']:>4.1f}% "
        f"cls={m['cls_pct']:>4.1f}% "
        f"fc={m['fc_pct']:>4.1f}% "
        f"L/σN={m['locked_over_sigma_naked']:>5.2f} "
        f"passes={m['passes']}/5"
    )


def append_monitor_entry(timestamp: str, content: str) -> None:
    """Append a dated entry to monitoring_notes.md."""
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = f"\n### {timestamp} — babysit iteration\n\n{content}\n"
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(entry)


def launch_wrapper(wrapper_rel: str) -> bool:
    """Launch a wrapper script as a fully-detached background process.

    Root cause of prior silent failures: Windows Task Scheduler puts
    its hosted python.exe inside a JOB OBJECT with the
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE flag. When babysit_loop.py
    exits, Windows kills the entire job tree, including any bash
    children — regardless of DETACHED_PROCESS / CREATE_NEW_PROCESS_GROUP.
    The fix is ``CREATE_BREAKAWAY_FROM_JOB`` (0x01000000) which lets
    the child escape the job object.

    Also: wait up to 30s for the wrapper log to appear; under Task
    Scheduler context bash startup can be slow.
    """
    wrapper = REPO / wrapper_rel
    if not wrapper.exists():
        log(f"wrapper not found: {wrapper}")
        return False
    log(f"launching wrapper: {wrapper}")
    try:
        if sys.platform == "win32":
            CREATE_BREAKAWAY_FROM_JOB = 0x01000000
            creationflags = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | CREATE_BREAKAWAY_FROM_JOB
            )
        else:
            creationflags = 0
        proc = subprocess.Popen(
            ["bash", str(wrapper)],
            cwd=str(REPO),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            creationflags=creationflags,
        )
        log(f"  Popen returned, child pid={proc.pid}")
        # Poll up to 30s for the wrapper log to appear. The wrapper
        # creates its log on the very first line (`exec >> $LOG 2>&1`)
        # so if it appears at all, launch succeeded.
        import time
        log_name = wrapper.name.replace("run_", "_").replace(".sh", "_wrapper.log")
        log_path = REPO / "registry" / log_name
        for i in range(30):
            time.sleep(1)
            if log_path.exists() and log_path.stat().st_size > 0:
                log(f"  wrapper log appeared after {i+1}s; launch CONFIRMED")
                return True
        # Final check: is the bash process still alive?
        try:
            import psutil
            if psutil.pid_exists(proc.pid):
                p = psutil.Process(proc.pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    log(f"  WARN: log absent after 30s but bash pid={proc.pid} still running; assuming launch OK")
                    return True
        except Exception:
            pass
        log(f"  FAIL: wrapper log absent after 30s, child not alive")
        return False
    except Exception as e:
        log(f"launch failed: {e}")
        return False


def main() -> int:
    BABYSIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("babysit iteration start")

    state = load_state()
    if state.get("stopped"):
        log(f"babysit stopped (reason: {state.get('stop_reason', 'unknown')}); exiting")
        return 0

    # --- Analyze any new cells since last iteration -------------
    seen = set(state.get("last_analyzed_cells", []))
    cells = all_completed_cells()
    new_cells = [c for c in cells if c.name not in seen]

    log(f"found {len(cells)} total cells, {len(new_cells)} new since last iteration")

    if new_cells:
        rows = []
        for c in new_cells:
            m = cell_metrics(c)
            if m is None:
                continue
            rows.append(m)
        if rows:
            # Sort by passes desc, then day_pnl desc
            rows.sort(key=lambda r: (-r["passes"], -r["day_pnl"]))
            content_lines = ["**New cells since last iteration:**", ""]
            for m in rows:
                content_lines.append(format_metric_row(m))
            # Identify leader (max passes, max day_pnl)
            leader = rows[0]
            content_lines.append("")
            content_lines.append(
                f"**Leader of this batch:** `{leader['cell']}` — {leader['passes']}/5, day_pnl={leader['day_pnl']:+.1f}"
            )

            # Overall leader across ALL completed cells
            all_m = [cell_metrics(c) for c in cells]
            all_m = [m for m in all_m if m is not None]
            all_m.sort(key=lambda r: (-r["passes"], -r["day_pnl"]))
            overall_leader = all_m[0]
            content_lines.append(
                f"**Overall leader so far:** `{overall_leader['cell']}` — "
                f"{overall_leader['passes']}/5, day_pnl={overall_leader['day_pnl']:+.1f}"
            )

            # Stop ONLY when positive day_pnl with reasonable
            # bet counts. The user wants a recipe that REGULARLY
            # hits positive day_pnl — so we want consistency, not
            # just one lucky cell. Keep launching until we see
            # POSITIVE day_pnl + opens in band + passes >= 4.
            if (
                overall_leader["day_pnl"] > 0
                and overall_leader["passes"] >= 4
                and 100 <= overall_leader["opens"] <= 180
            ):
                content_lines.append("")
                content_lines.append(
                    f"**POSITIVE DAY_PNL ACHIEVED** with {overall_leader['passes']}/5 acceptance. "
                    f"Continuing to queue replicate cells via existing rounds; do not auto-stop."
                )
                # Don't auto-stop — we want REGULAR positive day_pnl,
                # which means seeing it across multiple seeds /
                # variants. The pre-queued rounds will provide that.

            ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            append_monitor_entry(ts, "\n".join(content_lines))
            log(f"appended analysis for {len(rows)} cells; overall leader {overall_leader['cell']} ({overall_leader['passes']}/5)")

    state["last_analyzed_cells"] = sorted(set(seen) | {c.name for c in cells})

    # --- Decide whether to launch a new round --------------------
    if state.get("stopped"):
        save_state(state)
        log("babysit stopped after analysis; not launching")
        return 0

    # Is any wrapper currently mid-flight? If a cohort.runner is
    # running, a wrapper is alive.
    if cohort_running():
        log("cohort.runner is running; not launching")
        save_state(state)
        return 0

    # Are any wrappers themselves alive without a cohort.runner?
    # Could be between-cells lull. Don't double-launch.
    for w in ROUND_QUEUE:
        if wrapper_running(w):
            log(f"wrapper {w} is alive (between cells); not launching new")
            save_state(state)
            return 0

    # SAFETY: don't launch a wrapper whose log says it started but
    # hasn't logged "fan-out complete" yet — the wrapper may be alive
    # but ps detection failed (e.g., unicode error). This catches
    # wrappers that are running but not visible.
    for w in ROUND_QUEUE:
        wrapper = Path(w)
        log_name = wrapper.name.replace("run_", "_").replace(".sh", "_wrapper.log")
        log_path = REPO / "registry" / log_name
        if log_path.exists():
            txt = log_path.read_text(encoding="utf-8", errors="ignore")
            if "wrapper started" in txt and "fan-out complete" not in txt:
                log(f"wrapper {w} started but not complete; assuming alive; not launching")
                save_state(state)
                return 0

    # All wrappers idle. Find the next un-launched, un-completed round.
    rounds_launched = set(state.get("rounds_launched", []))
    for w in ROUND_QUEUE:
        if w in rounds_launched and wrapper_completed(w):
            continue
        # If we previously launched it but it isn't complete yet AND
        # nothing's running, the wrapper may have died. Re-launch.
        if w in rounds_launched and not wrapper_completed(w):
            log(f"wrapper {w} previously launched but not complete; re-launching")
            if launch_wrapper(w):
                save_state(state)
                return 0
            else:
                continue
        # Never launched — launch it.
        if launch_wrapper(w):
            rounds_launched.add(w)
            state["rounds_launched"] = sorted(rounds_launched)
            save_state(state)
            return 0
        else:
            log(f"failed to launch {w}; trying next")
            continue

    # No more rounds to launch.
    log("no further rounds queued; nothing to launch")
    state["stopped"] = True
    state["stop_reason"] = "round queue exhausted"
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
