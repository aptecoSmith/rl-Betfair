"""Autonomous cohort runner for the arb-signal-cleanup 3-cohort probe.

Polls the three plan files. When one completes, archives its logs, flips
``config.yaml`` for the next cohort, restarts the stack, and launches the
next plan via the training API. Idempotent — safe to call repeatedly; does
nothing if no transition is needed. After cohort C completes, restores
``config.yaml`` floors.

Usage (called from ScheduleWakeup or directly):

    python scripts/autorun_cohorts.py

Exit codes:
    0  — nothing to do (current cohort still running / in flight / draft
         waiting for worker), or transition succeeded and next cohort
         launched, or terminal success after cohort C completes + floors
         restored.
    1  — transition failed; manual intervention needed. Stack may be in
         an unclean state.
    2  — a plan is in ``status='failed'``; DO NOT advance. Alert user.

All state changes are logged to stdout with ISO timestamps. The caller
(ScheduleWakeup) captures stdout so the runbook is readable from the
conversation log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_PLANS = REPO_ROOT / "registry" / "training_plans"
CONFIG_PATH = REPO_ROOT / "config.yaml"
EPISODES_LOG = REPO_ROOT / "logs" / "training" / "episodes.jsonl"
STOP_SCRIPT = REPO_ROOT / "stop-all.bat"
START_SCRIPT = REPO_ROOT / "start-all.bat"
API_STATUS_URL = "http://localhost:8001/training/status"
API_START_URL = "http://localhost:8001/training/start"
API_READY_TIMEOUT_S = 90
API_READY_POLL_S = 2
STACK_STOP_SETTLE_S = 5
PLAN_LAUNCH_VERIFY_S = 10

COHORTS = {
    "A": {
        "plan_id": "8eff137d-37c4-4c60-80df-24f1f033efde",
        "force_close": 30,
        "warmup_eps": 10,
    },
    "B": {
        "plan_id": "04006f4f-e6cb-4539-a8f3-9f22b81a3535",
        "force_close": 0,
        "warmup_eps": 0,
    },
    "C": {
        "plan_id": "149440cb-1ad3-4262-9e52-c15931baf13f",
        "force_close": 30,
        "warmup_eps": 10,
    },
}
COHORT_ORDER = ("A", "B", "C")


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def _load_plan(plan_id: str) -> dict:
    path = TRAINING_PLANS / f"{plan_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _flip_config(force_close: int, warmup_eps: int) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    out = []
    fc_done = False
    wu_done = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if not fc_done and stripped.startswith("force_close_before_off_seconds:"):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}force_close_before_off_seconds: {force_close}")
            fc_done = True
        elif not wu_done and stripped.startswith("shaped_penalty_warmup_eps:"):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}shaped_penalty_warmup_eps: {warmup_eps}")
            wu_done = True
        else:
            out.append(line)
    if not fc_done or not wu_done:
        raise RuntimeError(
            f"config.yaml edit missed a field "
            f"(force_close={fc_done}, warmup={wu_done})"
        )
    CONFIG_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    _log(
        f"config.yaml flipped: force_close={force_close}, "
        f"warmup_eps={warmup_eps}"
    )


def _archive_episodes(cohort: str) -> Path | None:
    if not EPISODES_LOG.exists() or EPISODES_LOG.stat().st_size == 0:
        _log("episodes.jsonl empty/missing — nothing to archive")
        return None
    iso = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = EPISODES_LOG.parent / (
        f"episodes.cohort-{cohort}-complete-{iso}.jsonl"
    )
    EPISODES_LOG.replace(dst)
    EPISODES_LOG.touch()
    _log(f"archived log: {dst.name} ({dst.stat().st_size:,} bytes)")
    return dst


def _run_bat(script_path: Path, label: str) -> None:
    _log(f"{label}: {script_path.name}")
    result = subprocess.run(
        ["cmd", "/c", str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    for line in (result.stdout or "").splitlines():
        _log(f"  [stdout] {line}")
    for line in (result.stderr or "").splitlines():
        _log(f"  [stderr] {line}")
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} exited {result.returncode}"
        )


def _wait_api_ready() -> None:
    deadline = time.monotonic() + API_READY_TIMEOUT_S
    last_err = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(API_STATUS_URL, timeout=3) as resp:
                if resp.status == 200:
                    _log(f"API ready at {API_STATUS_URL}")
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = repr(e)
        time.sleep(API_READY_POLL_S)
    raise RuntimeError(
        f"API not ready after {API_READY_TIMEOUT_S}s: {last_err}"
    )


def _launch_plan(plan_id: str, cohort: str) -> None:
    body = json.dumps({
        "plan_id": plan_id,
        "smoke_test_first": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        API_START_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_txt = ""
        try:
            body_txt = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"POST {API_START_URL} {plan_id[:8]}... failed "
            f"({e.code}): {body_txt or e.reason}"
        )
    _log(
        f"launched cohort {cohort} plan {plan_id[:8]}... "
        f"response keys: {sorted(payload.keys())}"
    )
    # Brief settle, then verify the plan status flipped.
    time.sleep(PLAN_LAUNCH_VERIFY_S)
    plan = _load_plan(plan_id)
    _log(
        f"cohort {cohort} plan status post-launch: {plan['status']!r}"
    )
    if plan["status"] not in ("running", "draft"):
        raise RuntimeError(
            f"cohort {cohort} plan status unexpectedly {plan['status']!r} "
            f"after launch (expected running/draft)"
        )


def _transition(src: str, dst: str) -> None:
    dst_cfg = COHORTS[dst]
    _log(f"=== TRANSITION cohort {src} -> cohort {dst} ===")
    _run_bat(STOP_SCRIPT, "stop stack")
    time.sleep(STACK_STOP_SETTLE_S)
    _archive_episodes(src)
    _flip_config(dst_cfg["force_close"], dst_cfg["warmup_eps"])
    _run_bat(START_SCRIPT, "start stack")
    _wait_api_ready()
    _launch_plan(dst_cfg["plan_id"], dst)
    _log(f"=== TRANSITION {src} -> {dst} COMPLETE ===")


def _finalise() -> None:
    """Called after cohort C completes. Stop stack, archive C's logs,
    restore config.yaml floors.
    """
    _log("=== FINALISE: cohort C complete, restoring floors ===")
    _run_bat(STOP_SCRIPT, "stop stack (final)")
    time.sleep(STACK_STOP_SETTLE_S)
    _archive_episodes("C")
    _flip_config(force_close=0, warmup_eps=0)
    _log("config.yaml restored to floors. All three cohorts complete.")


def main() -> int:
    if not TRAINING_PLANS.exists():
        _log(f"FATAL: {TRAINING_PLANS} missing")
        return 1

    # Load current state of all three plans.
    states: dict[str, str] = {}
    for c in COHORT_ORDER:
        try:
            plan = _load_plan(COHORTS[c]["plan_id"])
        except FileNotFoundError:
            _log(f"FATAL: cohort {c} plan file missing")
            return 1
        states[c] = plan.get("status", "?")

    _log(
        "cohort states: "
        + ", ".join(f"{c}={states[c]}" for c in COHORT_ORDER)
    )

    # Fail gate: any cohort failed → stop chain.
    for c in COHORT_ORDER:
        if states[c] == "failed":
            _log(
                f"ALERT: cohort {c} is status='failed' — halting chain. "
                "Manual intervention required."
            )
            return 2

    # Nothing to do while the current cohort is in flight. ``paused`` is
    # the between-sessions state — with ``auto_continue: true`` the
    # worker auto-resumes, so we must NOT transition; just wait.
    in_flight = {"running", "paused"}
    for c in COHORT_ORDER:
        if states[c] in in_flight:
            _log(f"cohort {c} is {states[c]!r} (in-flight) — no action")
            return 0

    # Find the cohort to transition TO (first non-completed cohort after
    # the last completed one).
    completed: list[str] = []
    for c in COHORT_ORDER:
        if states[c] == "completed":
            completed.append(c)
            continue
        break

    if len(completed) == 0:
        # Cohort A hasn't started yet. User should have launched A
        # manually; this runner only handles transitions.
        _log(
            "cohort A not yet completed and not running — nothing to do. "
            "(Did you launch cohort A manually?)"
        )
        return 0

    if len(completed) == len(COHORT_ORDER):
        # All three done.
        if states["C"] == "completed":
            _finalise()
        return 0

    src = completed[-1]
    dst = COHORT_ORDER[len(completed)]
    try:
        _transition(src, dst)
    except Exception as e:
        _log(f"TRANSITION FAILED: {e!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
