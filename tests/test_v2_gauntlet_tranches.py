"""Guard: gauntlet tranches are FIXED-SIZE — the trailing remainder is dropped,
NOT folded into the last tranche (the 2026-06-20 bug: 38 days → 10/10/18).

Lockstep's make_rotations folds the remainder into the last fold by design; the
gauntlet must keep every tranche exactly per=train+eval days so every recipe
faces the identical T1..TN (hard_constraints.md). The leftover <per days become
T(N+1) once per more days bank.
"""
from __future__ import annotations

from training_v2.cohort.runner import _gauntlet_tranche_days


def _days(n):
    return [f"2026-04-{i+1:02d}" for i in range(n)]


def test_remainder_dropped_not_folded_into_last():
    # 38 days, per=10 → 3 tranches of EXACTLY 10 (was 10/10/18 before the fix).
    tr = _gauntlet_tranche_days(_days(38), 6, 4)
    assert [len(t) for t in tr] == [10, 10, 10]
    # 8-day remainder dropped (not in any tranche).
    used = [d for t in tr for d in t]
    assert len(used) == 30
    assert _days(38)[30:] == [d for d in _days(38) if d not in used]


def test_all_tranches_equal_per():
    for n in (30, 31, 39, 40, 41):
        tr = _gauntlet_tranche_days(_days(n), 6, 4)
        assert all(len(t) == 10 for t in tr), f"n={n}: {[len(t) for t in tr]}"
        assert len(tr) == n // 10


def test_old_anchored_t1_is_oldest():
    tr = _gauntlet_tranche_days(_days(25), 6, 4)
    assert tr[0] == _days(25)[:10]   # T1 = oldest block
    assert tr[1] == _days(25)[10:20]
    assert len(tr) == 2              # the newest 5 are dropped (not a tranche)


def test_exact_multiple_uses_all():
    tr = _gauntlet_tranche_days(_days(30), 6, 4)
    assert [len(t) for t in tr] == [10, 10, 10]
    assert sum(len(t) for t in tr) == 30  # nothing dropped when it divides
