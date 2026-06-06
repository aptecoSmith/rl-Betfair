"""Tick-Tock piece A — fresh-blood band-seed (the keystone).

Gates the seed mechanism that turns a full-width "Tick" era into a
hypothesis-warm-started "Tock":

* :func:`parse_seed_gene` — type-aware ``--seed-gene NAME=LO:HI`` parsing +
  range/choice validation (a typo never silently mis-seeds an era).
* :func:`draw_seeded_gene` — in-band draw respecting gene type; point pin.
* :func:`sample_fresh_blood_genes(..., seed_bands=...)` — the seed lands in
  the genome AND is recorded; the unseeded path is BYTE-IDENTICAL to before.
* ``pbt`` threading — a structural seed holds era-wide (gen-0 + R1 refill);
  a non-structural seed DRIFTS under breeding (not reset to default).
* :func:`runner._resolve_seed_bands` — auto-enable non-structural seeds,
  collision-guard vs --reward-overrides, gate⇄predictor coupling, breeding
  guard (the canonical ``_resolve_<knob>`` both-sources test).

Load-bearing: ``test_unseeded_*`` proves a no-seed Tick is bit-for-bit the
pre-Tick-Tock sampler (CLAUDE.md byte-identity rule).
"""

from __future__ import annotations

import random
import subprocess
import types
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from training_v2.cohort import genes as cur
from training_v2.cohort.genes import (
    PHASE5_GENE_DEFAULTS,
    PHASE5_GENE_NAMES,
    STOP_LOSS_PNL_THRESHOLD_RANGE,
    CohortGenes,
    draw_seeded_gene,
    parse_seed_gene,
    sample_fresh_blood_genes,
)
from training_v2.cohort.pbt import (
    STRUCTURAL_GENE_NAMES,
    PbtConfig,
    breed_pbt,
    init_pbt_population,
    make_offspring,
)
from training_v2.cohort.runner import _resolve_seed_bands

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── parse_seed_gene ────────────────────────────────────────────────────────


class TestParseSeedGene:
    def test_float_band(self):
        name, band = parse_seed_gene("stop_loss_pnl_threshold=0.18:0.26")
        assert name == "stop_loss_pnl_threshold"
        assert band == (0.18, 0.26)

    def test_float_point_pin(self):
        name, band = parse_seed_gene("stop_loss_pnl_threshold=0.22")
        assert band == (0.22, 0.22)

    def test_bool_true_false(self):
        assert parse_seed_gene("use_direction_predictor=true") == (
            "use_direction_predictor", (True, True))
        assert parse_seed_gene("direction_gate_enabled=false") == (
            "direction_gate_enabled", (False, False))

    def test_str_categorical_point(self):
        assert parse_seed_gene("architecture=transformer") == (
            "architecture", ("transformer", "transformer"))

    def test_int_choice_point_and_band(self):
        assert parse_seed_gene("bc_pretrain_steps=500") == (
            "bc_pretrain_steps", (500.0, 500.0))
        name, band = parse_seed_gene("hidden_size=128:256")
        assert name == "hidden_size" and band == (128.0, 256.0)

    def test_log_uniform_float_band(self):
        # bc_learning_rate ∈ [1e-5, 1e-3]; a high sub-band is valid.
        name, band = parse_seed_gene("bc_learning_rate=3e-4:8e-4")
        assert name == "bc_learning_rate"
        assert band == pytest.approx((3e-4, 8e-4))

    @pytest.mark.parametrize("bad", [
        "no_equals_here",                          # missing '='
        "not_a_gene=0.5",                          # unknown gene
        "use_direction_predictor=0:1",             # bool as a band
        "use_direction_predictor=maybe",           # bad bool token
        "architecture=lstm:transformer",           # str as a band
        "architecture=cnn",                        # invalid str choice
        "direction_gate_threshold=0.1:0.9",        # outside [0.20, 0.50]
        "stop_loss_pnl_threshold=0.3:0.1",         # lo > hi
        "transformer_heads=3:3",                   # no valid choice in band
        "stop_loss_pnl_threshold=abc",             # non-numeric
    ])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            parse_seed_gene(bad)


# ── draw_seeded_gene ───────────────────────────────────────────────────────


class TestDrawSeededGene:
    def test_point_pins(self):
        rng = random.Random(0)
        assert draw_seeded_gene(rng, "use_direction_predictor", (True, True)) is True
        assert draw_seeded_gene(rng, "architecture", ("transformer",) * 2) == "transformer"
        assert draw_seeded_gene(rng, "bc_pretrain_steps", (500, 500)) == 500
        assert draw_seeded_gene(rng, "stop_loss_pnl_threshold", (0.22, 0.22)) == 0.22

    def test_band_stays_in_range(self):
        rng = random.Random(1)
        for _ in range(500):
            v = draw_seeded_gene(rng, "stop_loss_pnl_threshold", (0.18, 0.26))
            assert 0.18 <= v <= 0.26
            lr = draw_seeded_gene(rng, "bc_learning_rate", (3e-4, 8e-4))
            assert 3e-4 <= lr <= 8e-4
            h = draw_seeded_gene(rng, "hidden_size", (128, 256))
            assert h in (128, 256)


# ── sample_fresh_blood_genes(seed_bands=...) ───────────────────────────────


class TestSampleFreshBloodSeeded:
    def test_band_seed_lands_in_range_and_is_recorded(self):
        """A seeded band draws inside the band AND shows up in to_dict() — the
        recorded ``hyperparameters`` the scoreboard/register persist."""
        bands = {"stop_loss_pnl_threshold": (0.18, 0.26)}
        en = frozenset({"stop_loss_pnl_threshold"})
        for s in range(200):
            g = sample_fresh_blood_genes(random.Random(s), enabled_set=en,
                                         seed_bands=bands)
            assert 0.18 <= g.stop_loss_pnl_threshold <= 0.26
            # Recorded in the persisted dict at the drawn value.
            assert g.to_dict()["stop_loss_pnl_threshold"] == \
                g.stop_loss_pnl_threshold

    def test_structural_bool_seed_holds_for_every_draw(self):
        """A structural seed (use_direction_predictor=true) is True on EVERY
        fresh-blood draw — the era-wide pin."""
        bands = {"use_direction_predictor": (True, True),
                 "direction_gate_enabled": (True, True)}
        for s in range(200):
            g = sample_fresh_blood_genes(random.Random(s), seed_bands=bands)
            assert g.use_direction_predictor is True
            assert g.direction_gate_enabled is True

    def test_coupling_raises_when_gate_seeded_without_predictor(self):
        bands = {"direction_gate_enabled": (True, True),
                 "use_direction_predictor": (False, False)}
        with pytest.raises(ValueError, match="use_direction_predictor=True"):
            sample_fresh_blood_genes(random.Random(0), seed_bands=bands)

    def test_seed_overrides_phase5_default_even_when_disabled(self):
        """A seeded PHASE5 gene takes the seed value even if (defensively) it
        is not in enabled_set — the seed wins over the disabled-default."""
        bands = {"stop_loss_pnl_threshold": (0.25, 0.25)}
        g = sample_fresh_blood_genes(random.Random(0), enabled_set=frozenset(),
                                     seed_bands=bands)
        assert g.stop_loss_pnl_threshold == 0.25
        assert g.stop_loss_pnl_threshold != PHASE5_GENE_DEFAULTS[
            "stop_loss_pnl_threshold"]


# ── BYTE-IDENTITY (load-bearing) ───────────────────────────────────────────


class TestUnseededByteIdentical:
    def test_seed_bands_none_equals_empty_and_is_deterministic(self):
        for s in range(60):
            a = sample_fresh_blood_genes(random.Random(s))
            b = sample_fresh_blood_genes(random.Random(s), seed_bands=None)
            c = sample_fresh_blood_genes(random.Random(s), seed_bands={})
            assert a.to_dict() == b.to_dict() == c.to_dict()

    def test_seed_bands_none_does_not_shift_rng_stream(self):
        """The unseeded path must consume the IDENTICAL RNG draws, so the very
        next number off the stream matches with/without the seed_bands kwarg."""
        for s in range(40):
            ra, rb = random.Random(s), random.Random(s)
            ga = sample_fresh_blood_genes(ra)
            gb = sample_fresh_blood_genes(rb, seed_bands=None)
            assert ga.to_dict() == gb.to_dict()
            assert ra.random() == rb.random()  # stream position identical

    def test_unseeded_matches_git_head(self):
        """Strongest guard: exec the pre-change genes.py from git HEAD (it
        imports only math/random/dataclasses — no package internals) and
        compare a batch of unseeded genomes. Proves the seed_bands plumbing
        did not perturb the sampler. Skips gracefully without git/HEAD."""
        try:
            src = subprocess.run(
                ["git", "show", "HEAD:training_v2/cohort/genes.py"],
                capture_output=True, text=True, cwd=str(REPO_ROOT),
            )
        except (OSError, FileNotFoundError):  # pragma: no cover
            pytest.skip("git unavailable")
        if src.returncode != 0 or not src.stdout:  # pragma: no cover
            pytest.skip("HEAD genes.py unavailable")
        import sys
        head = types.ModuleType("genes_head")
        head.__dict__["__name__"] = "genes_head"
        # @dataclass (py3.14) resolves sys.modules[cls.__module__] — the temp
        # module must be registered there for the exec to define CohortGenes.
        sys.modules["genes_head"] = head
        try:
            exec(compile(src.stdout, "<genes_head>", "exec"), head.__dict__)
            if not hasattr(head, "sample_fresh_blood_genes"):  # pragma: no cover
                pytest.skip("HEAD predates sample_fresh_blood_genes")
            for enabled in (frozenset(),
                            frozenset(head.PHASE5_GENE_NAMES)):
                for s in range(80):
                    gh = head.sample_fresh_blood_genes(
                        random.Random(s), enabled_set=enabled)
                    gc = sample_fresh_blood_genes(
                        random.Random(s), enabled_set=enabled)
                    assert gh.to_dict() == gc.to_dict(), (s, sorted(enabled)[:1])
        finally:
            sys.modules.pop("genes_head", None)


# ── PBT threading (era-wide pin + drift) ───────────────────────────────────


@dataclass
class _Res:
    model_id: str
    weights_path: str
    score: float


def _score(r: _Res) -> float:
    return r.score


def _pair(specs):
    # score = -i so specs[0] ranks highest (deterministic winners).
    return [(s, _Res(model_id=f"m{i}", weights_path=f"w{i}.pt", score=-i))
            for i, s in enumerate(specs)]


def _cfg(**kw) -> PbtConfig:
    base = dict(n_agents=30, r2_size=10, r3_size=6,
                promote_from_r1=5, promote_from_r2=3, freeze_top_r3=3)
    base.update(kw)
    return PbtConfig(**base)


class TestPbtSeedThreading:
    _STRUCT = {"use_direction_predictor": (True, True),
               "direction_gate_enabled": (True, True)}
    _NONSTRUCT = {"stop_loss_pnl_threshold": (0.18, 0.26)}
    _EN = frozenset({"stop_loss_pnl_threshold"})

    def test_gen0_structural_seed_era_wide(self):
        pop = init_pbt_population(random.Random(0), _cfg(),
                                  enabled_set=frozenset(),
                                  seed_bands=self._STRUCT)
        assert len(pop) == 30
        assert all(s.genes.use_direction_predictor is True for s in pop)
        assert all(s.genes.direction_gate_enabled is True for s in pop)

    def test_gen0_band_seed_in_range(self):
        pop = init_pbt_population(random.Random(0), _cfg(),
                                  enabled_set=self._EN,
                                  seed_bands=self._NONSTRUCT)
        assert all(0.18 <= s.genes.stop_loss_pnl_threshold <= 0.26
                   for s in pop)

    def test_r1_refill_is_seeded(self):
        """breed_pbt's rookie injection (R1 fresh blood) also band-seeds, so
        the hypothesis keeps entering the population every generation."""
        cfg = _cfg()
        pop = init_pbt_population(random.Random(0), cfg,
                                  enabled_set=frozenset(),
                                  seed_bands=self._STRUCT)
        nxt, _ = breed_pbt(_pair(pop), random.Random(1), cfg,
                           score_fn=_score, enabled_set=frozenset(),
                           seed_bands=self._STRUCT)
        fresh = [s for s in nxt if s.role == "fresh"]
        assert fresh, "expected R1 fresh blood in the next gen"
        assert all(s.genes.use_direction_predictor is True for s in fresh)
        assert all(s.genes.direction_gate_enabled is True for s in fresh)

    def test_offspring_drift_from_seed_not_default(self):
        """A non-structural seed, AUTO-ENABLED, DRIFTS under breeding: the
        offspring's value is a ±frac perturbation of the seeded parent value,
        NOT reset to the gene default (0.0)."""
        rng = random.Random(3)
        parent = sample_fresh_blood_genes(rng, enabled_set=self._EN,
                                          seed_bands=self._NONSTRUCT)
        assert 0.18 <= parent.stop_loss_pnl_threshold <= 0.26
        for s in range(50):
            child = make_offspring(parent, random.Random(s),
                                   enabled_set=self._EN, frac=0.20)
            # Drifted, not the default.
            assert child.stop_loss_pnl_threshold != \
                PHASE5_GENE_DEFAULTS["stop_loss_pnl_threshold"]
            # Within ±20% of the parent (clamped to the gene range).
            lo = max(STOP_LOSS_PNL_THRESHOLD_RANGE[0],
                     parent.stop_loss_pnl_threshold * 0.80)
            hi = min(STOP_LOSS_PNL_THRESHOLD_RANGE[1],
                     parent.stop_loss_pnl_threshold * 1.20)
            assert lo - 1e-9 <= child.stop_loss_pnl_threshold <= hi + 1e-9

    def test_structural_seed_survives_breeding(self):
        """A structural seed is inherited verbatim by make_offspring (frozen
        per lineage) — the era-wide pin holds through the gauntlet."""
        rng = random.Random(4)
        parent = sample_fresh_blood_genes(rng, seed_bands=self._STRUCT)
        child = make_offspring(parent, random.Random(9),
                               enabled_set=frozenset(), frac=0.20)
        for name in STRUCTURAL_GENE_NAMES:
            assert getattr(child, name) == getattr(parent, name), name
        assert child.use_direction_predictor is True
        assert child.direction_gate_enabled is True


# ── runner._resolve_seed_bands (the _resolve_<knob> both-sources test) ─────


class TestResolveSeedBands:
    def test_no_seeds_is_noop(self):
        en = frozenset({"open_cost"})
        bands, en2 = _resolve_seed_bands([], breeding="pbt",
                                         reward_overrides={}, enabled_set=en)
        assert bands == {} and en2 == en

    def test_nonstructural_seed_auto_enabled(self):
        bands, en = _resolve_seed_bands(
            ["stop_loss_pnl_threshold=0.18:0.26"], breeding="pbt",
            reward_overrides={}, enabled_set=frozenset())
        assert "stop_loss_pnl_threshold" in bands
        assert "stop_loss_pnl_threshold" in en  # auto-enabled → drifts

    def test_structural_seed_not_auto_enabled(self):
        """Structural seeds are frozen-inherited, not PHASE5 genes — they must
        NOT be added to enabled_set."""
        bands, en = _resolve_seed_bands(
            ["use_direction_predictor=true"], breeding="pbt",
            reward_overrides={}, enabled_set=frozenset())
        assert "use_direction_predictor" in bands
        assert "use_direction_predictor" not in en
        assert "use_direction_predictor" not in PHASE5_GENE_NAMES

    def test_requires_breeding_pbt(self):
        with pytest.raises(SystemExit, match="breeding pbt"):
            _resolve_seed_bands(["stop_loss_pnl_threshold=0.2:0.3"],
                                breeding="ga", reward_overrides={},
                                enabled_set=frozenset())

    def test_collision_with_reward_overrides(self):
        with pytest.raises(ValueError, match="source of truth"):
            _resolve_seed_bands(
                ["stop_loss_pnl_threshold=0.2:0.3"], breeding="pbt",
                reward_overrides={"stop_loss_pnl_threshold": 0.2},
                enabled_set=frozenset())

    def test_gate_seed_requires_predictor_seed(self):
        with pytest.raises(ValueError, match="use_direction_predictor=true"):
            _resolve_seed_bands(["direction_gate_enabled=true"],
                                breeding="pbt", reward_overrides={},
                                enabled_set=frozenset())

    def test_gate_with_predictor_ok(self):
        bands, en = _resolve_seed_bands(
            ["use_direction_predictor=true", "direction_gate_enabled=true"],
            breeding="pbt", reward_overrides={}, enabled_set=frozenset())
        assert bands["direction_gate_enabled"] == (True, True)
        assert bands["use_direction_predictor"] == (True, True)

    def test_duplicate_seed_rejected(self):
        with pytest.raises(ValueError, match="more than once"):
            _resolve_seed_bands(
                ["open_cost=1.0", "open_cost=2.0"], breeding="pbt",
                reward_overrides={}, enabled_set=frozenset())
