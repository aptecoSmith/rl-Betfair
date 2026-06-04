"""BC must SKIP (not crash) when the oracle's obs config doesn't match.

pbt-breeding 2026-06-04: lean-obs fresh blood (obs_dim 574) crashed the
multiprocess pool at gen 0 because BC's oracle load does a STRICT obs_dim
check and the only oracle cache is full-obs (2254): it RAISED instead of
skipping. With lean obs now a fresh-blood gene (some agents lean, some full)
and only a full-obs oracle present, a mismatch is EXPECTED and must be
graceful — the agent trains via PPO without BC. This is the regression guard
for that (a 'missing test' the operator flagged after the OOM postmortem).
"""

from __future__ import annotations

from pathlib import Path

from training_v2.cohort import worker


def test_skips_bc_on_oracle_obs_dim_mismatch(monkeypatch):
    def _raise(**_kw):
        raise ValueError("Cache obs_dim=2254 but caller expects 574. "
                         "Re-run oracle scan against the current shim/scorer")
    monkeypatch.setattr(worker, "load_oracle_samples_for_dates", _raise)
    out = worker._load_bc_oracle_or_skip(
        dates=["2026-04-19"], data_dir=Path("data/processed"),
        obs_dim=574, agent_id="lean-agent")
    assert out is None  # graceful skip — did NOT raise / crash the worker


def test_returns_samples_when_oracle_matches(monkeypatch):
    sentinel = [object(), object(), object()]
    monkeypatch.setattr(
        worker, "load_oracle_samples_for_dates", lambda **_kw: sentinel)
    out = worker._load_bc_oracle_or_skip(
        dates=["2026-04-19"], data_dir=Path("data/processed"),
        obs_dim=2254, agent_id="full-agent")
    assert out is sentinel  # full-obs agent: oracle matches -> BC runs


def test_non_valueerror_still_propagates(monkeypatch):
    # A genuine bug (not an obs-dim mismatch) must NOT be swallowed.
    def _boom(**_kw):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(worker, "load_oracle_samples_for_dates", _boom)
    import pytest
    with pytest.raises(RuntimeError, match="disk on fire"):
        worker._load_bc_oracle_or_skip(
            dates=["2026-04-19"], data_dir=Path("x"), obs_dim=574,
            agent_id="a")
