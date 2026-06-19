"""Guard: the static_obs cache atomic rename tolerates transient Windows
file interference (the 2026-06-18 mid-prebuild FileNotFoundError).

A freshly-written ``*.tmp`` can be briefly locked/quarantined by Defender or a
search indexer between write and ``os.replace``; the rename then raises
PermissionError (WinError 5/32) or FileNotFoundError (WinError 2). These are
transient. ``_atomic_replace`` retries, and treats "tmp gone but dst already
present" as success (idempotent).
"""
from __future__ import annotations

import pytest

from training_v2.cohort import static_obs_cache as soc
from training_v2.cohort.static_obs_cache import _atomic_replace


def test_happy_path_renames(tmp_path):
    src = tmp_path / "x.tmp"
    dst = tmp_path / "x"
    src.write_bytes(b"data")
    _atomic_replace(src, dst)
    assert dst.read_bytes() == b"data"
    assert not src.exists()


def test_tmp_gone_but_dst_present_is_success(tmp_path):
    # Simulate: rename already landed elsewhere — tmp absent, dst present.
    src = tmp_path / "x.tmp"  # does NOT exist
    dst = tmp_path / "x"
    dst.write_bytes(b"already-there")
    _atomic_replace(src, dst)  # must NOT raise
    assert dst.read_bytes() == b"already-there"


def test_transient_permission_error_then_success(tmp_path, monkeypatch):
    src = tmp_path / "x.tmp"
    dst = tmp_path / "x"
    src.write_bytes(b"payload")

    real_replace = type(src).replace
    calls = {"n": 0}

    def flaky_replace(self, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("WinError 32 — file locked by AV")
        return real_replace(self, target)

    monkeypatch.setattr(type(src), "replace", flaky_replace)
    monkeypatch.setattr(soc.time, "sleep", lambda *_: None)  # no real delay
    _atomic_replace(src, dst)
    assert dst.read_bytes() == b"payload"
    assert calls["n"] == 3


def test_persistent_failure_with_no_dst_raises(tmp_path, monkeypatch):
    src = tmp_path / "x.tmp"
    dst = tmp_path / "x"
    src.write_bytes(b"payload")

    def always_fail(self, target):
        raise PermissionError("locked forever")

    monkeypatch.setattr(type(src), "replace", always_fail)
    monkeypatch.setattr(soc.time, "sleep", lambda *_: None)
    with pytest.raises(PermissionError):
        _atomic_replace(src, dst, attempts=3)
