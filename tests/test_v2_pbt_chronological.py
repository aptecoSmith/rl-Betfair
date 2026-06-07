"""Tick-Tock rotation-rework — chronological folds + N-tier ladder.

Gates the going-forward data structure:

* ``make_rotations(mode="chronological")`` — old-anchored date-blocks so
  R1..R(n-1) are FIXED as data accumulates (new data extends the high end),
  the LAST rotation absorbs the remainder, and each block evals on its
  NEWEST ``eval_per`` days (train old → eval new).
* ``PbtConfig`` N-tier (R4+) helpers + validation.
* ``breed_pbt`` generic N-tier ladder — builds R1..RN, freezes the top tier.
* BYTE-IDENTITY: with a 3-tier config + ``mode="random"`` the new generic
  code reproduces the pre-rework ``pbt.py`` (exec'd from git HEAD) bit-for-bit
  on genes/tier/role and on the random folds.
"""

from __future__ import annotations

import random
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from training_v2.cohort.pbt import (
    PbtConfig,
    breed_pbt,
    init_pbt_population,
    make_rotations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── chronological make_rotations ───────────────────────────────────────────


def _days(n: int, start_day: int = 1) -> list[str]:
    # n consecutive 2026-04/05 dates, chronological.
    out = []
    d = start_day
    mo = 4
    for _ in range(n):
        if d > 28:
            d -= 28
            mo += 1
        out.append(f"2026-{mo:02d}-{d:02d}")
        d += 1
    return out


class TestChronologicalRotations:
    def test_old_anchored_blocks_eval_is_newest(self):
        pool = _days(48)
        rots = make_rotations(pool, cohort_seed=1, n_rotations=4,
                              train_per_rotation=6, eval_per_rotation=6,
                              mode="chronological")
        assert len(rots) == 4
        srt = sorted(pool)
        # R1 = oldest 12, eval = newest 6 OF THAT BLOCK, train = oldest 6.
        assert set(rots[0].train_days) == set(srt[0:6])
        assert set(rots[0].eval_days) == set(srt[6:12])
        # R4 = newest 12.
        assert set(rots[3].train_days) | set(rots[3].eval_days) == set(srt[36:48])
        # disjoint train/eval within each rotation.
        for r in rots:
            assert not (set(r.train_days) & set(r.eval_days))

    def test_last_rotation_absorbs_remainder(self):
        pool = _days(50)  # 4×12 = 48, remainder 2
        rots = make_rotations(pool, cohort_seed=1, n_rotations=4,
                              train_per_rotation=6, eval_per_rotation=6,
                              mode="chronological")
        sizes = [len(r.train_days) + len(r.eval_days) for r in rots]
        assert sizes == [12, 12, 12, 14]  # last absorbs the 2 extra
        # eval stays 6 even in the bigger last block; train grows.
        assert len(rots[3].eval_days) == 6
        assert len(rots[3].train_days) == 8

    def test_r1_to_r3_fixed_as_data_grows(self):
        """The keystone: adding newer data leaves R1..R(n-1) untouched; only
        the top rotation grows (until the operator adds another tier)."""
        pool48 = _days(48)
        pool54 = pool48 + _days(6, start_day=200)  # 6 strictly-newer days
        a = make_rotations(pool48, cohort_seed=1, n_rotations=4,
                           train_per_rotation=6, eval_per_rotation=6,
                           mode="chronological")
        b = make_rotations(pool54, cohort_seed=1, n_rotations=4,
                           train_per_rotation=6, eval_per_rotation=6,
                           mode="chronological")
        for i in range(3):  # R1, R2, R3 identical
            assert a[i].train_days == b[i].train_days
            assert a[i].eval_days == b[i].eval_days
        # R4 grew (absorbed the 6 new days).
        a4 = len(a[3].train_days) + len(a[3].eval_days)
        b4 = len(b[3].train_days) + len(b[3].eval_days)
        assert b4 == a4 + 6

    def test_raises_on_too_small_pool(self):
        with pytest.raises(ValueError):
            make_rotations(_days(20), cohort_seed=1, n_rotations=4,
                          train_per_rotation=6, eval_per_rotation=6,
                          mode="chronological")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            make_rotations(_days(48), cohort_seed=1, n_rotations=4,
                          mode="bogus")


# ── PbtConfig N-tier helpers + validation ──────────────────────────────────


class TestNTierConfig:
    def test_three_tier_defaults_unchanged(self):
        c = PbtConfig(n_agents=16, n_rotations=3, r2_size=6, r3_size=4,
                      promote_from_r1=3, promote_from_r2=2, freeze_top_r3=2)
        assert c.n_tiers == 3
        assert c.tier_size(2) == 6 and c.tier_size(3) == 4
        assert c.promote_from(1) == 3 and c.promote_from(2) == 2
        assert c.freeze_count() == 2
        c.validate()  # legacy path, no raise

    def test_four_tier_helpers(self):
        c = PbtConfig(n_agents=16, n_rotations=4, tier_sizes=(6, 4, 3),
                      promote_counts=(3, 2, 2), freeze_top=2)
        assert c.n_tiers == 4
        assert [c.tier_size(t) for t in (2, 3, 4)] == [6, 4, 3]
        assert [c.promote_from(t) for t in (1, 2, 3)] == [3, 2, 2]
        assert c.freeze_count() == 2
        c.validate()

    def test_validate_rejects_bad_ntier(self):
        # promote_from(1)=7 > tier_size(2)=6
        with pytest.raises(ValueError, match="promote_from"):
            PbtConfig(n_agents=16, n_rotations=4, tier_sizes=(6, 4, 3),
                      promote_counts=(7, 2, 2), freeze_top=2).validate()
        # n_rotations != n_tiers
        with pytest.raises(ValueError, match="n_rotations"):
            PbtConfig(n_agents=16, n_rotations=3, tier_sizes=(6, 4, 3),
                      promote_counts=(3, 2, 2), freeze_top=2).validate()
        # sum of tiers > n_agents
        with pytest.raises(ValueError, match="sum of tier_sizes"):
            PbtConfig(n_agents=8, n_rotations=4, tier_sizes=(6, 4, 3),
                      promote_counts=(3, 2, 2), freeze_top=2).validate()


# ── N-tier breed ladder ────────────────────────────────────────────────────


@dataclass
class _Res:
    model_id: str
    weights_path: str
    score: float


def _score(r: _Res) -> float:
    return r.score


def _pair(specs, base=0):
    return [(s, _Res(f"m{base}_{i}", f"w{base}_{i}.pt", score=-i))
            for i, s in enumerate(specs)]


class TestNTierBreed:
    _CFG = PbtConfig(
        n_agents=16, n_rotations=4, train_per_rotation=6, eval_per_rotation=6,
        tier_sizes=(6, 4, 3), promote_counts=(3, 2, 2), freeze_top=2)

    def test_pipeline_fills_to_four_tiers_and_freezes_r4(self):
        rng = random.Random(0)
        specs = init_pbt_population(rng, self._CFG, enabled_set=frozenset())
        assert len(specs) == 16 and all(s.tier == 1 for s in specs)
        froze_r4 = False
        for gen in range(6):
            nxt, frozen = breed_pbt(_pair(specs, base=gen), random.Random(gen + 1),
                                    self._CFG, score_fn=_score)
            assert len(nxt) == 16
            counts = {t: sum(1 for s in nxt if s.tier == t) for t in (1, 2, 3, 4)}
            assert sum(counts.values()) == 16
            if frozen:
                # only the TOP tier (R4) freezes, capped at freeze_top=2.
                assert all(s.tier == 4 for s, _ in frozen)
                assert len(frozen) <= 2
                froze_r4 = True
            specs = nxt
        # By gen 6 the pipeline has filled R4 and frozen champions from it.
        assert froze_r4
        assert counts[2] == 6 and counts[3] == 4 and counts[4] == 3
        assert counts[1] == 16 - 6 - 4 - 3  # R1 absorbs slack = 3

    def test_r4_agents_have_seen_four_rotations(self):
        rng = random.Random(0)
        specs = init_pbt_population(rng, self._CFG, enabled_set=frozenset())
        for gen in range(6):
            specs, _ = breed_pbt(_pair(specs, base=gen), random.Random(gen + 1),
                                 self._CFG, score_fn=_score)
        r4 = [s for s in specs if s.tier == 4]
        assert r4
        for s in r4:  # climbed R1→R2→R3→R4
            assert {1, 2, 3, 4} <= s.rotations_seen


# ── BYTE-IDENTITY: generic 3-tier == pre-rework pbt.py (git HEAD) ───────────


def _load_head_pbt():
    try:
        src = subprocess.run(
            ["git", "show", "HEAD:training_v2/cohort/pbt.py"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO_ROOT))
    except (OSError, FileNotFoundError):  # pragma: no cover
        pytest.skip("git unavailable")
    if src.returncode != 0 or not src.stdout:  # pragma: no cover
        pytest.skip("HEAD pbt.py unavailable")
    mod = types.ModuleType("pbt_head")
    mod.__dict__["__name__"] = "pbt_head"
    sys.modules["pbt_head"] = mod
    exec(compile(src.stdout, "<pbt_head>", "exec"), mod.__dict__)
    return mod


class TestThreeTierByteIdentical:
    _SCALAR = dict(n_agents=16, n_rotations=3, train_per_rotation=6,
                   eval_per_rotation=4, r2_size=6, r3_size=4,
                   promote_from_r1=3, promote_from_r2=2, freeze_top_r3=2)

    def test_random_make_rotations_matches_head(self):
        head = _load_head_pbt()
        pool = _days(40)
        for seed in range(20):
            h = head.make_rotations(pool, cohort_seed=seed, n_rotations=3)
            c = make_rotations(pool, cohort_seed=seed, n_rotations=3,
                               mode="random")
            assert [(r.train_days, r.eval_days) for r in h] == \
                   [(r.train_days, r.eval_days) for r in c]
        sys.modules.pop("pbt_head", None)

    def test_breed_3tier_matches_head_on_genes_tier_role(self):
        head = _load_head_pbt()
        try:
            for seed in range(12):
                hc = head.PbtConfig(**self._SCALAR)
                cc = PbtConfig(**self._SCALAR)
                pop_h = head.init_pbt_population(
                    random.Random(seed), hc, enabled_set=frozenset())
                pop_c = init_pbt_population(
                    random.Random(seed), cc, enabled_set=frozenset())
                # Same gene draws (lineage uuids differ — exclude them).
                assert [g.genes.to_dict() for g in pop_h] == \
                       [g.genes.to_dict() for g in pop_c]
                nh, fh = head.breed_pbt(_pair(pop_h), random.Random(99), hc,
                                        score_fn=_score, enabled_set=frozenset())
                nc, fc = breed_pbt(_pair(pop_c), random.Random(99), cc,
                                   score_fn=_score, enabled_set=frozenset())
                sig_h = [(s.genes.to_dict(), s.tier, s.role) for s in nh]
                sig_c = [(s.genes.to_dict(), s.tier, s.role) for s in nc]
                assert sig_h == sig_c, f"breed diverged at seed {seed}"
                assert len(fh) == len(fc)
        finally:
            sys.modules.pop("pbt_head", None)
