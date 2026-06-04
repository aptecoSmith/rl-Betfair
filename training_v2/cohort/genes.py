"""Cohort gene schema — Phase 3 (7 legacy) + Phase 5 (11 promoted).

Phase 3 (Session 03) locked a deliberately small 7-gene set
(``learning_rate, entropy_coeff, clip_range, gae_lambda,
value_coeff, mini_batch_size, hidden_size``) — see
``plans/rewrite/phase-3-cohort/purpose.md``.

Phase 5 (``plans/rewrite/phase-5-restore-genes/``, opened
2026-05-03) promotes 11 additional knobs that were already
designed as per-agent genes by their own plans but never landed
on ``CohortGenes``. They are evolved per-agent only when the
operator opts in via the cohort runner's ``--enable-gene NAME``
flag; disabled genes stay frozen at their pre-Phase-5 cohort-wide
default value, preserving byte-identity for legacy launches.

Phase 5 gene table (all bounds inclusive):

================================ ============ =============== ====================
Gene                             Range        Distribution    Default-when-disabled
================================ ============ =============== ====================
open_cost                        [0.0, 2.0]   uniform         0.0
matured_arb_bonus_weight         [0.0, 5.0]   uniform         0.0
mark_to_market_weight            [0.0, 0.10]  uniform         0.05
naked_loss_scale                 [0.0, 1.0]   uniform         1.0
stop_loss_pnl_threshold          [0.0, 0.30]  uniform         0.0
arb_spread_target_lock_pct       [0.005, 0.05] uniform        0.02
fill_prob_loss_weight            [0.0, 0.30]  uniform         0.0
mature_prob_loss_weight          [1.0, 5.0]   uniform         0.0
risk_loss_weight                 [0.0, 0.30]  uniform         0.0
alpha_lr                         [1e-2, 1e-1] log-uniform     1e-2
reward_clip                      [1.0, 10.0]  uniform         10.0
================================ ============ =============== ====================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields


# ── Phase 3 range / choice constants ──────────────────────────────────────


LEARNING_RATE_RANGE: tuple[float, float] = (1e-5, 1e-3)
ENTROPY_COEFF_RANGE: tuple[float, float] = (1e-4, 1e-1)
CLIP_RANGE_RANGE: tuple[float, float] = (0.1, 0.3)
GAE_LAMBDA_RANGE: tuple[float, float] = (0.9, 0.99)
VALUE_COEFF_RANGE: tuple[float, float] = (0.25, 1.0)
MINI_BATCH_SIZE_CHOICES: tuple[int, ...] = (32, 64, 128)
# The gene-only GA + every existing launch sample this (UNCHANGED ->
# byte-identity, HC#1). Fresh blood uses the per-architecture *_SAMPLE sets
# below; assert_in_range accepts the wider _VALID set.
HIDDEN_SIZE_CHOICES: tuple[int, ...] = (64, 128, 256)
# Valid hidden sizes (assert_in_range). Widened 2026-06-04 so fresh-blood
# LSTMs can go larger (operator: bigger LSTMs ranked higher in gen-0 R1).
HIDDEN_SIZE_VALID: tuple[int, ...] = (64, 128, 256, 512, 1024)
# Fresh-blood LSTM sampling: larger sizes allowed (LSTM is cheap per tick on
# CPU vs the transformer's per-tick encoder). NB larger = slower gens.
HIDDEN_SIZE_LSTM_SAMPLE: tuple[int, ...] = (64, 128, 256, 512, 1024)
# Fresh-blood transformer d_model. Un-capped to 512 (2026-06-04) for the GPU
# lane — a d512 transformer is affordable on GPU (the lane routes its
# forward+update to CUDA). Every value divides the head counts {2,4,8} (512%8=0)
# so d_model % n_heads == 0 always holds. d512 raises GPU memory (the semaphore
# caps concurrency); 1024 left out (the activations balloon as O(mb*heads*ctx^2)).
HIDDEN_SIZE_TRANSFORMER_SAMPLE: tuple[int, ...] = (64, 128, 256, 512)


# ── Architecture genes (pbt-breeding Step 1b, 2026-06-03) ─────────────────
# STRUCTURAL genes: set ONLY at fresh-blood birth and FROZEN within a
# lineage (warm-start weight inheritance requires matching weight shapes —
# HC#10). The gauntlet's hall-of-fame records each champion's architecture
# so the system *reports which architecture wins* (design.md). The
# defaults below (``"lstm"`` + the v1 transformer defaults) reproduce the
# pre-pbt schema byte-identically: the base ``sample_genes`` / gene-only GA
# never sample these (``_sample_field`` pins them), so an existing launch
# is unchanged. Only ``sample_fresh_blood_genes`` draws across the choices.
ARCHITECTURE_CHOICES: tuple[str, ...] = ("lstm", "transformer")
# ``hidden_size`` doubles as the transformer's ``d_model``; every
# (hidden_size, n_heads) pair from the choice lists divides evenly
# (64/128/256 % 2/4/8 == 0) so d_model % n_heads == 0 always holds.
TRANSFORMER_DEPTH_CHOICES: tuple[int, ...] = (1, 2, 3, 4, 6)
TRANSFORMER_HEADS_CHOICES: tuple[int, ...] = (2, 4, 8)
TRANSFORMER_CTX_TICKS_CHOICES: tuple[int, ...] = (32, 64, 128, 256)
# FFN expansion: dim_feedforward = mult * d_model (gene 2026-06-04). 2x is the
# current narrow default; 4x is the standard transformer width (more per-token
# capacity, more cost). STRUCTURAL (weight shapes) -> frozen per lineage.
TRANSFORMER_FFN_MULT_CHOICES: tuple[int, ...] = (2, 4)
# Positional encoding (gene 2026-06-04): "learned" = the current absolute
# nn.Embedding; "rope" = rotary RELATIVE encoding (better for a time-series
# where "N ticks ago" matters). STRUCTURAL (different params) -> frozen.
TRANSFORMER_POS_ENCODING_CHOICES: tuple[str, ...] = ("learned", "rope")

# Fresh-blood SAMPLING (2026-06-04, UN-CAPPED 2026-06-04 for the GPU lane).
# History: a ctx256/depth3 transformer took ~74 min/agent ON CPU and gated the
# whole generation, so sampling was capped to ctx{32,64}/depth{1,2}. The GPU
# policy lane (plans/pbt-gpu-forward) removes that cap's reason: big-ctx
# transformers (ctx>=128) route their forward+update to CUDA (measured 6.3x on
# the ctx256 forward), collapsing their agent-day toward the env-sim floor —
# LSTM-comparable. So fresh blood may again draw the full range; the lane keeps
# the big ones affordable while small transformers + LSTMs stay on pure CPU.
TRANSFORMER_DEPTH_SAMPLE: tuple[int, ...] = (1, 2, 3, 4, 6)
TRANSFORMER_CTX_TICKS_SAMPLE: tuple[int, ...] = (32, 64, 128, 256)
TRANSFORMER_FFN_MULT_SAMPLE: tuple[int, ...] = (2, 4)
# "rope" stays OUT of sampling until the policy implements the custom RoPE
# attention (task #8) — gene infra + valid CHOICES are in place; flip this to
# ("learned", "rope") the moment RoPE is built + smoke-ablated, and the campaign
# picks it up on its next relaunch. Keeps fresh blood from drawing a path the
# policy can't build yet.
TRANSFORMER_POS_ENCODING_SAMPLE: tuple[str, ...] = ("learned",)

# GPU policy lane (plans/pbt-gpu-forward, 2026-06-04). A transformer whose
# context window is at least this routes its policy FORWARD + batched PPO
# UPDATE to CUDA (the env always stays on CPU). Below this — and for every LSTM
# — the batch=1 forward is launch-bound on GPU (measured: ctx64 GPU ~= CPU; the
# LSTM's batch=1 GEMV loses on GPU), so they stay on the pure-CPU R5 path.
GPU_LANE_MIN_CTX: int = 128

# ── Transformer size spec for the GPU lane (2026-06-04) ───────────────────────
# GRADUATE TO GPU: architecture == transformer AND ctx_ticks >= GPU_LANE_MIN_CTX
#   (128). Keyed on ctx — the dominant O(ctx^2) attention cost. ctx 32/64
#   transformers + EVERY LSTM stay CPU (batch=1 forward is launch-bound on GPU).
# CEILING on ctx = 256. Races are ~150-250 ticks, so ctx256 already spans the
#   FULL race; ctx > 256 is just padding, no signal gain. Do not raise it.
# ROOM TO GROW with the lane is d_model + depth (NOT ctx). MEASURED:
#   ctx256/d256/depth3/heads8 = ~200s/agent-day on GPU, ~4GB (~= a full-obs
#   LSTM). CONSERVATIVE MAX to explore next: ctx256, d_model 512, depth 4-6,
#   heads 8-16 (est ~1.5-2x -> ~300-450s; beyond d512/depth6 ~600-800s starts
#   to gate gens). Only ctx256/d256/depth3 is measured — smoke-bench a bigger
#   build (plans/pbt-gpu-forward/_measure/bench_ctx256_threads.py) before a
#   campaign. To enable: widen HIDDEN_SIZE_TRANSFORMER_SAMPLE (caps d_model at
#   256 today) to add 512, and TRANSFORMER_DEPTH_SAMPLE/_CHOICES for depth > 3.
#
# NON-SIZE CONFIG LEVERS (operator 2026-06-04 — make these genes so the
#   gauntlet explores them; both currently HARDCODED in DiscreteTransformerPolicy):
#   * FFN ratio: dim_feedforward is 2x d_model today (vs the standard 4x) — a
#     per-token capacity knob -> `transformer_ffn_mult` gene {2, 4}. STRUCTURAL
#     (changes weight shapes) -> frozen per lineage, in ARCHITECTURE_GENE_NAMES.
#   * Positional encoding: learned-absolute nn.Embedding today. RELATIVE
#     encodings (RoPE / ALiBi) usually beat absolute on time-series (price
#     dynamics care about "N ticks ago") -> `transformer_pos_encoding` gene
#     {learned, rope}. STRUCTURAL (different params) -> frozen per lineage.
#     RoPE needs a custom attention path -> build carefully + smoke-ablate.
#   Plus KV-cache the rollout (efficiency, not a gene) — O(ctx^2)/tick re-encode
#     is the dominant ctx256 forward cost; pbt-gpu-forward flagged it for ctx>=256.
# ─────────────────────────────────────────────────────────────────────────────


def is_gpu_lane_eligible(genes) -> bool:
    """Whether this agent's policy compute should run on CUDA. True only for
    big-context transformers (ctx_ticks >= GPU_LANE_MIN_CTX) — the case where
    the O(ctx^2) attention forward is large enough that GPU beats CPU even at
    batch=1. Env stays on CPU regardless; only forward + update move. Duck-typed
    on ``.architecture`` / ``.transformer_ctx_ticks`` so it also accepts a
    trainer_hp-style mapping via getattr-friendly objects."""
    return (str(getattr(genes, "architecture", "lstm")) == "transformer"
            and int(getattr(genes, "transformer_ctx_ticks", 0)) >= GPU_LANE_MIN_CTX)

#: The structural gene names — frozen within a lineage, sampled only at
#: fresh-blood birth. ``hidden_size`` (a legacy gene) is structural too
#: under warm-start (mutating it breaks weight inheritance) but stays in
#: its own legacy handling; this set covers the pbt-added ones.
ARCHITECTURE_GENE_NAMES: frozenset[str] = frozenset({
    "architecture",
    "transformer_depth",
    "transformer_heads",
    "transformer_ctx_ticks",
    "transformer_ffn_mult",      # FFN width -> weight shapes (structural)
    "transformer_pos_encoding",  # learned vs RoPE -> different params (structural)
    # Obs representation is structural too (different obs_dim/runner_dim ->
    # different weight shapes -> warm-start can't cross it). Sampled at
    # fresh-blood birth, frozen for the lineage, never perturbed in breeding.
    "predictor_lean_obs",
})


# ── Phase 5 range constants (2026-05-03) ──────────────────────────────────


OPEN_COST_RANGE: tuple[float, float] = (0.0, 2.0)
MATURED_ARB_BONUS_WEIGHT_RANGE: tuple[float, float] = (0.0, 5.0)
MARK_TO_MARKET_WEIGHT_RANGE: tuple[float, float] = (0.0, 0.10)
NAKED_LOSS_SCALE_RANGE: tuple[float, float] = (0.0, 1.0)
STOP_LOSS_PNL_THRESHOLD_RANGE: tuple[float, float] = (0.0, 0.30)
# Price-adaptive arb_spread, redesigned 2026-05-23 (plans/force_close_and_arb_spread/).
# Fraction of aggressive stake the agent wants locked per scalped pair.
# The env passes this directly to ``min_arb_ticks_for_profit`` as the
# ``profit_floor`` argument:
#   arb_ticks = min_arb_ticks_for_profit(agg_price, side, commission,
#                                         profit_floor=target_lock_pct)
# Returns None (pair refused) if the target lock can't be reached
# within MAX_ARB_TICKS at the current price + commission. Otherwise
# returns the smallest tick offset that delivers >= target_lock_pct
# locked per £1 aggressive stake. Naturally adapts across the price
# ladder — same gene value gives roughly the same % locked at every
# price (different tick counts, same target %).
#
# Phenotype handle:
#   0.005 (0.5%) — "fill-seeker": tight passives, high fill rate,
#                  tiny profit per pair
#   0.02  (2.0%) — balanced default
#   0.05  (5.0%) — "profit-seeker": wide passives, lower fill rate,
#                  big lock per pair when they do fill
# Range upper bound 0.05 chosen because higher targets are
# commonly unreachable within MAX_ARB_TICKS at typical scalping
# prices under 5% commission.
ARB_SPREAD_TARGET_LOCK_PCT_RANGE: tuple[float, float] = (0.005, 0.05)
FILL_PROB_LOSS_WEIGHT_RANGE: tuple[float, float] = (0.0, 0.30)
MATURE_PROB_LOSS_WEIGHT_RANGE: tuple[float, float] = (1.0, 5.0)
RISK_LOSS_WEIGHT_RANGE: tuple[float, float] = (0.0, 0.30)
ALPHA_LR_RANGE: tuple[float, float] = (1e-2, 1e-1)
REWARD_CLIP_RANGE: tuple[float, float] = (1.0, 10.0)
# scalping-tight-naked-variance Phase 2A (2026-05-15). L2 symmetric
# per-pair naked-pnl variance penalty coefficient. Applied to the
# SHAPED reward channel only. Default 0.0 = byte-identical pre-plan.
# Range upper bound matches env/betfair_env.py::
# NAKED_VARIANCE_PENALTY_BETA_MAX. See plans/scalping-tight-naked-
# variance/hard_constraints.md §7-§11.
NAKED_VARIANCE_PENALTY_BETA_RANGE: tuple[float, float] = (0.0, 0.10)
# Phase-14 S03 (2026-05-07). Direction-gate threshold. Lower bound
# is the gate's "no-op floor" semantic. Upper bound clamps the
# strictest gene draw — at 0.99+ an agent never opens, starving
# PPO of training signal (per phase-14 hard_constraints §10).
#
# 2026-05-25 RECALIBRATION (recipe-sensitivity-sweep / shared
# direction head C11). The original (0.5, 0.95) range was set when
# the per-agent direction head was pos_weighted (outputs cluster
# near 0.5 by construction). The frozen shared C11 head is trained
# UNWEIGHTED so its outputs are calibrated to the actual ~18 %
# positive rate — observed `direction_back_prob` / `direction_lay_prob`
# on placed bets show mean ~0.26, max ~0.84, with `max(back, lay)`
# per-runner mean ~0.32. At threshold 0.5+ the gate refuses 95-99 %
# of opens (effectively NOOP-only), starving PPO. New range
# (0.20, 0.50) puts the strictest draw at "refuse ~95 % of opens"
# and the loosest at "refuse ~30 % of opens" — meaningful response
# curve across the gene's range. See
# `plans/recipe-sensitivity-sweep/purpose.md` for the discovery
# trail.
DIRECTION_GATE_THRESHOLD_RANGE: tuple[float, float] = (0.20, 0.50)
# Phase-15 (2026-05-24). Promoted from operator-pinned to GA-evolvable
# after the 2026-05-24 direction signal probe found:
#   * positive class rate ~15-20 % across training days
#   * pos-weighted random-uniform-0.5 floor for that imbalance ~1.13
#   * standardised logistic regression on raw obs descends to ~1.00
#     (10-12 % relative descent) — signal IS in obs
#   * probe2's reported `train_mean_direction_back/lay_bce` ~1.14 sat
#     AT the floor → head undertrained at the pinned 0.05 weight
# Range floor 0.1 = 2x the broken pin (lowest agent still in
# "actually trains" territory). Ceiling 2.0 = comparable to
# mature_prob_loss_weight's mid-range — the GA tuning precedent
# (CLAUDE.md 2026-05-04 raised mature_prob from [0.0, 0.30] to
# [1.0, 5.0] for the same "head not training under small weight"
# diagnosis). 20x ratio gives the GA room to find the curve.
DIRECTION_PROB_LOSS_WEIGHT_RANGE: tuple[float, float] = (0.1, 2.0)
# Phase-15 (2026-05-24). Promoted from operator-pinned at the same
# time as `direction_prob_loss_weight`. The 2026-05-24 probe log line
# showed `post_bc_dir_bce_back=0.7483 lay=0.6975` — both at the
# unweighted random floor of ln(2)~0.693, meaning BC at the pinned
# weight 0.1 did not move the direction head either. BC's blended
# loss is `(1 - w) * oracle_ce + w * direction_ce`; range floor 0.1
# = current pin (preserves a "control" agent matching probe2);
# ceiling 0.5 = aggressive but oracle still dominates the loss
# tug-of-war (above 0.5 risks turning BC into "predict direction"
# instead of "imitate oracle action").
BC_DIRECTION_TARGET_WEIGHT_RANGE: tuple[float, float] = (0.1, 0.5)

# Predictor-integration Session 03 (plans/predictor-integration/
# integration_contract.md §4). Five new genes:
#   predictor_feature_gain   — scalar 0..1 scaling the predictor obs
#                              columns in actor_input. 0 = ignore
#                              predictors entirely; 1 = full strength.
#                              Cross-mode (arb / value_*).
#   value_edge_threshold     — value_win mode only. Minimum
#                              (champion_p_win - implied_p_win) for
#                              the policy to consider a bet.
#                              Manifest's value_spotting_at_inference_time
#                              recommends 0.05.
#   value_kelly_fraction     — value_win mode only. Fraction of full
#                              Kelly the agent stakes at. 0 ≈ ignore
#                              edge; 1 = full-Kelly.
#   each_way_edge_threshold  — value_each_way mode only.
#   each_way_kelly_fraction  — value_each_way mode only.
# Non-applicable-to-mode genes are still present (zero-effect when
# the env's reward gate doesn't read them) so cross-mode breeding
# stays trivial. Path A pattern from CLAUDE.md §"v2 stack consumes
# aux-head loss weights" §"v2-specific worker plumbing".
PREDICTOR_FEATURE_GAIN_RANGE: tuple[float, float] = (0.0, 1.0)
VALUE_EDGE_THRESHOLD_RANGE: tuple[float, float] = (0.02, 0.10)
VALUE_KELLY_FRACTION_RANGE: tuple[float, float] = (0.0, 1.0)
EACH_WAY_EDGE_THRESHOLD_RANGE: tuple[float, float] = (0.02, 0.10)
EACH_WAY_KELLY_FRACTION_RANGE: tuple[float, float] = (0.0, 1.0)


#: Default value applied to a Phase 5 gene whose name is NOT in the cohort's
#: ``enabled_set``. Each value matches the pre-Phase-5 cohort-wide default
#: so a launch with no ``--enable-gene`` flags is byte-identical to a
#: pre-plan run at the same seed.
PHASE5_GENE_DEFAULTS: dict[str, float] = {
    "open_cost": 0.0,
    "matured_arb_bonus_weight": 0.0,
    "mark_to_market_weight": 0.05,
    "naked_loss_scale": 1.0,
    "stop_loss_pnl_threshold": 0.0,
    # Price-adaptive arb_spread, redesigned 2026-05-23. Default 0.02 =
    # "lock 2% of stake per pair". Always active in the env (no opt-in
    # flag) — operator-pinning via --arb-spread-target-lock-pct,
    # GA evolution via --enable-gene arb_spread_target_lock_pct.
    "arb_spread_target_lock_pct": 0.02,
    "fill_prob_loss_weight": 0.0,
    "mature_prob_loss_weight": 0.0,
    "risk_loss_weight": 0.0,
    "alpha_lr": 1e-2,
    "reward_clip": 10.0,
    # scalping-tight-naked-variance Phase 2A (2026-05-15). Default
    # 0.0 = no penalty = byte-identical to pre-plan.
    "naked_variance_penalty_beta": 0.0,
    # Phase-14 S03 (2026-05-07). Default 0.5 = the gate's no-op
    # floor: when ``direction_gate_enabled=False`` (the cohort-wide
    # default) this value is unread; when enabled, 0.5 is the
    # value at which the gate filters the fewest rows.
    "direction_gate_threshold": 0.5,
    # Phase-15 (2026-05-24). Promoted from operator-pinned. Default
    # 0.0 = direction head receives no supervised gradient; the head
    # initialises near sigmoid(0)=0.5 and contributes a near-constant
    # column to actor_head's input (benign no-signal). When > 0 the
    # head learns to predict label_back / label_lay from the obs
    # vector (and the predictor's direction-predictor signal in
    # particular). See "Phase-15 direction signal probe" comment.
    "direction_prob_loss_weight": 0.0,
    # Phase-15 (2026-05-24). Promoted alongside
    # `direction_prob_loss_weight` (same diagnosis). Default 0.0 =
    # BC pretrain uses oracle-only target = byte-identical to pre-
    # Phase-15 launches that didn't set --reward-overrides for this
    # knob.
    "bc_direction_target_weight": 0.0,
    # Predictor-integration Session 03 (2026-05-10). Defaults per
    # integration_contract.md §4. Cross-mode breeding-friendly: every
    # mode's CohortGenes.to_dict() carries all 5 keys at the documented
    # defaults; the env's reward gate / action surface decides which
    # ones are read.
    "predictor_feature_gain": 1.0,
    "value_edge_threshold": 0.05,
    "value_kelly_fraction": 0.25,
    "each_way_edge_threshold": 0.05,
    "each_way_kelly_fraction": 0.25,
}


#: Frozenset of Phase 5 gene names. Used to dispatch enable/disable
#: behaviour in ``sample_genes`` / ``mutate`` / ``crossover``. The 7
#: legacy genes are NOT in this set — they always evolve.
PHASE5_GENE_NAMES: frozenset[str] = frozenset(PHASE5_GENE_DEFAULTS)


#: Phase 5 ranges keyed by gene name, used by ``assert_in_range``.
_PHASE5_RANGES: dict[str, tuple[float, float]] = {
    "open_cost": OPEN_COST_RANGE,
    "matured_arb_bonus_weight": MATURED_ARB_BONUS_WEIGHT_RANGE,
    "mark_to_market_weight": MARK_TO_MARKET_WEIGHT_RANGE,
    "naked_loss_scale": NAKED_LOSS_SCALE_RANGE,
    "stop_loss_pnl_threshold": STOP_LOSS_PNL_THRESHOLD_RANGE,
    "arb_spread_target_lock_pct": ARB_SPREAD_TARGET_LOCK_PCT_RANGE,
    "fill_prob_loss_weight": FILL_PROB_LOSS_WEIGHT_RANGE,
    "mature_prob_loss_weight": MATURE_PROB_LOSS_WEIGHT_RANGE,
    "risk_loss_weight": RISK_LOSS_WEIGHT_RANGE,
    "alpha_lr": ALPHA_LR_RANGE,
    "reward_clip": REWARD_CLIP_RANGE,
    "naked_variance_penalty_beta": NAKED_VARIANCE_PENALTY_BETA_RANGE,
    "direction_gate_threshold": DIRECTION_GATE_THRESHOLD_RANGE,
    "direction_prob_loss_weight": DIRECTION_PROB_LOSS_WEIGHT_RANGE,
    "bc_direction_target_weight": BC_DIRECTION_TARGET_WEIGHT_RANGE,
    "predictor_feature_gain": PREDICTOR_FEATURE_GAIN_RANGE,
    "value_edge_threshold": VALUE_EDGE_THRESHOLD_RANGE,
    "value_kelly_fraction": VALUE_KELLY_FRACTION_RANGE,
    "each_way_edge_threshold": EACH_WAY_EDGE_THRESHOLD_RANGE,
    "each_way_kelly_fraction": EACH_WAY_KELLY_FRACTION_RANGE,
}


# Floats sampled log-uniform on the [lo, hi] range. Floats absent from
# this set are sampled uniform.
_LOG_UNIFORM_FLOATS: frozenset[str] = frozenset({
    "learning_rate", "entropy_coeff", "alpha_lr",
})


# ── Public dataclass ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CohortGenes:
    """One agent's hyperparameters.

    Frozen so the dataclass is hashable and the GA can't accidentally
    mutate a parent in-place during breeding. Use :func:`crossover` /
    :func:`mutate` to produce children.

    Phase 5 fields (added 2026-05-03) come AT THE END of the dataclass.
    Their defaults match ``PHASE5_GENE_DEFAULTS`` so code constructing
    a ``CohortGenes`` with only the 7 legacy fields specified still
    works and reproduces pre-Phase-5 behaviour.
    """

    # Phase 3 (legacy 7) — always evolved.
    learning_rate: float
    entropy_coeff: float
    clip_range: float
    gae_lambda: float
    value_coeff: float
    mini_batch_size: int
    hidden_size: int

    # Phase 5 (promoted 11, 2026-05-03). Default values match cohort-wide
    # pre-plan defaults so unused genes stay neutral.
    open_cost: float = 0.0
    matured_arb_bonus_weight: float = 0.0
    mark_to_market_weight: float = 0.05
    naked_loss_scale: float = 1.0
    stop_loss_pnl_threshold: float = 0.0
    # Price-adaptive arb_spread, redesigned 2026-05-23
    # (plans/force_close_and_arb_spread/). See
    # ARB_SPREAD_TARGET_LOCK_PCT_RANGE doc-comment. Replaced the prior
    # arb_spread_scale + arb_spread_headroom_ticks pair from earlier
    # the same day — both became redundant once the gene expresses
    # the locked-profit target directly.
    arb_spread_target_lock_pct: float = 0.02
    fill_prob_loss_weight: float = 0.0
    mature_prob_loss_weight: float = 0.0
    risk_loss_weight: float = 0.0
    alpha_lr: float = 1e-2
    reward_clip: float = 10.0
    # scalping-tight-naked-variance Phase 2A (added 2026-05-15).
    # L2 symmetric per-pair naked-pnl variance penalty coefficient.
    # Default 0.0 = byte-identical to pre-plan. Range [0, 0.005].
    naked_variance_penalty_beta: float = 0.0

    # Phase 8 (added 2026-05-05). BC pretrain knobs. Defaults are
    # inert — ``bc_pretrain_steps = 0`` is the no-op switch; the other
    # two are unread when steps == 0. Adding fields with defaults to
    # the frozen dataclass is backward-compatible at the registry
    # layer (rows persist via ``to_dict``; old rows missing these keys
    # round-trip through ``CohortGenes(**existing_dict)`` because every
    # new field has a default).
    bc_pretrain_steps: int = 0
    bc_learning_rate: float = 3e-4
    bc_target_entropy_warmup_eps: int = 5

    # Phase-13 (added 2026-05-06). Direction-prob aux head — phase-13
    # S03. ``direction_prob_loss_weight = 0.0`` is byte-identical to
    # pre-plan: head exists in the network (architecture-hash break)
    # but contributes no BCE term to total_loss. The three label-
    # defining knobs resolve the offline cache stem at trainer init
    # — they MUST match the values used to scan the labels (S02 CLI).
    direction_prob_loss_weight: float = 0.0
    direction_horizon_ticks: int = 60
    direction_threshold_ticks: int = 5
    direction_force_close_seconds: float = 60.0
    # Phase-13 S05 (2026-05-06). Direction-targeted BC pretrain. With
    # weight 0.0 the BC step runs oracle-only (byte-identical to
    # phase-8 BC); with weight w > 0 the loss interpolates as
    # ``(1 - w) * oracle_ce + w * direction_ce``. Operator-controlled
    # via ``--reward-overrides bc_direction_target_weight=X``.
    bc_direction_target_weight: float = 0.0

    # Phase-14 S03 (2026-05-07). Direction-confidence gate. The
    # ``direction_gate_enabled`` flag turns the policy-side hard mask
    # ON; ``direction_gate_threshold`` controls how strict the gate
    # is. Defaults: enabled=False (byte-identical to phase-14
    # S01+S02), threshold=0.5 (the no-op floor when the gate IS
    # enabled — the policy clamps the actual value into
    # [DIRECTION_GATE_THRESHOLD_MIN=0.5,
    #  DIRECTION_GATE_THRESHOLD_MAX=0.95]).
    direction_gate_enabled: bool = False
    direction_gate_threshold: float = 0.5
    # Phase-14 S06 (2026-05-07). Threshold-warmup window in episodes.
    # The trainer linearly anneals the policy's effective threshold
    # from 0.5 to the agent's gene value across this many episodes
    # (mirrors bc_target_entropy_warmup_eps). Default 5 — operator-
    # controlled, not GA-evolved. Set to 0 to disable warmup
    # entirely (gene value applies from episode 0).
    direction_gate_warmup_eps: int = 5

    # Predictor-integration Session 03 (added 2026-05-10).
    # Per integration_contract.md §4. Cross-mode (arb / value_win /
    # value_each_way) breeding-friendly: every gene carries a
    # default value, every mode's CohortGenes.to_dict() emits all 5
    # keys, the env / trainer reads only the ones relevant to the
    # active strategy_mode. See `plans/predictor-integration/
    # session_prompts/03_strategy_mode_switch.md` §"New CohortGenes
    # fields" for the per-mode usage map.
    predictor_feature_gain: float = 1.0
    value_edge_threshold: float = 0.05
    value_kelly_fraction: float = 0.25
    each_way_edge_threshold: float = 0.05
    each_way_kelly_fraction: float = 0.25

    # Architecture genes (pbt-breeding Step 1b, 2026-06-03). STRUCTURAL —
    # sampled only at fresh-blood birth, frozen within a lineage (HC#10).
    # Defaults reproduce the pre-pbt all-LSTM schema; ``hidden_size``
    # (above) doubles as the transformer's ``d_model``. ``transformer_*``
    # are unread when ``architecture == "lstm"``.
    architecture: str = "lstm"
    transformer_depth: int = 2
    transformer_heads: int = 4
    transformer_ctx_ticks: int = 32
    transformer_ffn_mult: int = 2          # dim_feedforward = mult * d_model
    transformer_pos_encoding: str = "learned"   # "learned" | "rope"
    # Predictor obs representation — STRUCTURAL (changes obs_dim + runner_dim
    # -> weight shapes, so frozen within a lineage). A fresh-blood OPTION so
    # the gauntlet explores BOTH the lean predictor obs (~370-d, well-scaled,
    # fast) and the full obs (~2254-d, BC sets its input-norm). Default False
    # (full) keeps the gene-only GA + existing launches unchanged.
    predictor_lean_obs: bool = False

    def to_dict(self) -> dict:
        """Plain-dict form for registry persistence + scoreboard rows."""
        return {
            "learning_rate": float(self.learning_rate),
            "entropy_coeff": float(self.entropy_coeff),
            "clip_range": float(self.clip_range),
            "gae_lambda": float(self.gae_lambda),
            "value_coeff": float(self.value_coeff),
            "mini_batch_size": int(self.mini_batch_size),
            "hidden_size": int(self.hidden_size),
            "open_cost": float(self.open_cost),
            "matured_arb_bonus_weight": float(self.matured_arb_bonus_weight),
            "mark_to_market_weight": float(self.mark_to_market_weight),
            "naked_loss_scale": float(self.naked_loss_scale),
            "stop_loss_pnl_threshold": float(self.stop_loss_pnl_threshold),
            "arb_spread_target_lock_pct": float(self.arb_spread_target_lock_pct),
            "fill_prob_loss_weight": float(self.fill_prob_loss_weight),
            "mature_prob_loss_weight": float(self.mature_prob_loss_weight),
            "risk_loss_weight": float(self.risk_loss_weight),
            "alpha_lr": float(self.alpha_lr),
            "reward_clip": float(self.reward_clip),
            "naked_variance_penalty_beta": float(
                self.naked_variance_penalty_beta,
            ),
            "bc_pretrain_steps": int(self.bc_pretrain_steps),
            "bc_learning_rate": float(self.bc_learning_rate),
            "bc_target_entropy_warmup_eps": int(
                self.bc_target_entropy_warmup_eps,
            ),
            "direction_prob_loss_weight": float(
                self.direction_prob_loss_weight,
            ),
            "direction_horizon_ticks": int(self.direction_horizon_ticks),
            "direction_threshold_ticks": int(
                self.direction_threshold_ticks,
            ),
            "direction_force_close_seconds": float(
                self.direction_force_close_seconds,
            ),
            "bc_direction_target_weight": float(
                self.bc_direction_target_weight,
            ),
            "direction_gate_enabled": bool(self.direction_gate_enabled),
            "direction_gate_threshold": float(
                self.direction_gate_threshold,
            ),
            "direction_gate_warmup_eps": int(
                self.direction_gate_warmup_eps,
            ),
            "predictor_feature_gain": float(self.predictor_feature_gain),
            "value_edge_threshold": float(self.value_edge_threshold),
            "value_kelly_fraction": float(self.value_kelly_fraction),
            "each_way_edge_threshold": float(self.each_way_edge_threshold),
            "each_way_kelly_fraction": float(self.each_way_kelly_fraction),
            "architecture": str(self.architecture),
            "transformer_depth": int(self.transformer_depth),
            "transformer_heads": int(self.transformer_heads),
            "transformer_ctx_ticks": int(self.transformer_ctx_ticks),
            "transformer_ffn_mult": int(self.transformer_ffn_mult),
            "transformer_pos_encoding": str(self.transformer_pos_encoding),
            "predictor_lean_obs": bool(self.predictor_lean_obs),
        }


# ── Sampling ──────────────────────────────────────────────────────────────


def _sample_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return float(math.exp(rng.uniform(log_lo, log_hi)))


def _sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _sample_field(rng: random.Random, field_name: str):
    """Sample one gene value following the locked schema."""
    if field_name == "learning_rate":
        return _sample_log_uniform(rng, *LEARNING_RATE_RANGE)
    if field_name == "entropy_coeff":
        return _sample_log_uniform(rng, *ENTROPY_COEFF_RANGE)
    if field_name == "clip_range":
        return _sample_uniform(rng, *CLIP_RANGE_RANGE)
    if field_name == "gae_lambda":
        return _sample_uniform(rng, *GAE_LAMBDA_RANGE)
    if field_name == "value_coeff":
        return _sample_uniform(rng, *VALUE_COEFF_RANGE)
    if field_name == "mini_batch_size":
        return int(rng.choice(MINI_BATCH_SIZE_CHOICES))
    if field_name == "hidden_size":
        return int(rng.choice(HIDDEN_SIZE_CHOICES))
    if field_name in _PHASE5_RANGES:
        lo, hi = _PHASE5_RANGES[field_name]
        if field_name in _LOG_UNIFORM_FLOATS:
            return _sample_log_uniform(rng, lo, hi)
        return _sample_uniform(rng, lo, hi)
    # Phase 8 (2026-05-05). BC pretrain knobs are operator-controlled
    # (cohort runner ``--bc-pretrain-steps`` flag) rather than
    # GA-evolved per-agent. Sampling pins them to their pre-S02 inert
    # defaults so a fresh CohortGenes draw is byte-identical to a
    # pre-S02 draw at the same seed. If a future plan wants the GA to
    # explore BC settings, replace these returns with proper random
    # samplers and add the genes to ``_PHASE5_RANGES``.
    if field_name == "bc_pretrain_steps":
        return 0
    if field_name == "bc_learning_rate":
        return 3e-4
    if field_name == "bc_target_entropy_warmup_eps":
        return 5
    # Phase-13 (2026-05-06) / Phase-15 (2026-05-24).
    # `direction_prob_loss_weight` and `bc_direction_target_weight`
    # are now Phase 5 genes and dispatched via the `_PHASE5_RANGES`
    # branch above. The remaining direction_* knobs below stay
    # operator-controlled (constants surfaced via reward-overrides /
    # cohort-runner flags) and pin to inert defaults so an unsampled
    # cohort is byte-identical to pre-S03 at the same seed.
    if field_name == "direction_horizon_ticks":
        return 60
    if field_name == "direction_threshold_ticks":
        return 5
    if field_name == "direction_force_close_seconds":
        return 60.0
    # Phase-14 S03 (2026-05-07). ``direction_gate_enabled`` is a
    # cohort-wide bool, never sampled per-agent — operator turns it
    # on via ``--reward-overrides direction_gate_enabled=true``.
    # Sampling pins it to False; the cohort runner overlays the
    # operator value before constructing the policy. The threshold
    # itself IS sampled when in the ``enabled_set`` (it lives in
    # _PHASE5_RANGES).
    if field_name == "direction_gate_enabled":
        return False
    # Phase-14 S06 (2026-05-07). Warmup eps is operator-controlled,
    # never sampled per-agent. Pin to default 5.
    if field_name == "direction_gate_warmup_eps":
        return 5
    # Architecture genes (pbt-breeding Step 1b). The BASE sampler (used by
    # the gene-only GA + every existing launch) PINS these to the LSTM
    # defaults so it stays byte-identical to the pre-pbt schema — only
    # ``sample_fresh_blood_genes`` draws across the architecture choices.
    if field_name == "architecture":
        return "lstm"
    if field_name == "transformer_depth":
        return 2
    if field_name == "transformer_heads":
        return 4
    if field_name == "transformer_ctx_ticks":
        return 32
    if field_name == "transformer_ffn_mult":
        return 2
    if field_name == "transformer_pos_encoding":
        return "learned"
    if field_name == "predictor_lean_obs":
        return False
    raise KeyError(f"Unknown gene field: {field_name!r}")


def sample_genes(
    rng: random.Random,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Sample one fresh agent's genes from the locked schema.

    The 7 legacy genes (PPO + architecture) ALWAYS evolve. Phase 5
    genes evolve only when their name is in ``enabled_set``;
    otherwise they take the cohort-wide default from
    ``PHASE5_GENE_DEFAULTS``.
    """
    kwargs: dict = {}
    for f in fields(CohortGenes):
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            kwargs[f.name] = PHASE5_GENE_DEFAULTS[f.name]
        else:
            kwargs[f.name] = _sample_field(rng, f.name)
    return CohortGenes(**kwargs)


def _sample_architecture_field(rng: random.Random, field_name: str):
    """Draw one STRUCTURAL architecture gene across its full choice set."""
    if field_name == "architecture":
        return rng.choice(ARCHITECTURE_CHOICES)
    if field_name == "transformer_depth":
        return int(rng.choice(TRANSFORMER_DEPTH_SAMPLE))
    if field_name == "transformer_heads":
        return int(rng.choice(TRANSFORMER_HEADS_CHOICES))
    if field_name == "transformer_ctx_ticks":
        return int(rng.choice(TRANSFORMER_CTX_TICKS_SAMPLE))
    if field_name == "transformer_ffn_mult":
        return int(rng.choice(TRANSFORMER_FFN_MULT_SAMPLE))
    if field_name == "transformer_pos_encoding":
        return str(rng.choice(TRANSFORMER_POS_ENCODING_SAMPLE))
    if field_name == "predictor_lean_obs":
        # ~50/50 lean vs full so the gauntlet explores both representations.
        return bool(rng.random() < 0.5)
    raise KeyError(f"Not an architecture gene: {field_name!r}")


def sample_fresh_blood_genes(
    rng: random.Random,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """PBT fresh blood (pbt-breeding Step 1b/2, HC#9).

    Samples the 7 legacy genes + the STRUCTURAL architecture genes
    (``architecture`` ∈ {lstm, transformer} + transformer
    depth/heads/ctx_ticks) across their full choice sets — this is the
    architecture tournament's entry point, and the only sampler that
    draws an architecture (the base :func:`sample_genes` pins them to the
    LSTM default so the gene-only GA stays byte-identical). Phase 5 genes
    follow the same ``enabled_set`` convention as :func:`sample_genes`.

    The drawn structural genes are then FROZEN for the lineage's life —
    the breed step (:func:`make_offspring`) perturbs only non-structural
    genes, because warm-start weight inheritance needs matching weight
    shapes (HC#10).
    """
    # Sample the architecture FIRST so hidden_size can be conditioned on it:
    # an LSTM may go large (cheap per tick on CPU); a transformer's d_model
    # is capped at 256 (its per-tick encoder makes a big d_model gate whole
    # generations on the CPU-only multiprocess path).
    arch = _sample_architecture_field(rng, "architecture")
    hidden_choices = (
        HIDDEN_SIZE_LSTM_SAMPLE if arch == "lstm"
        else HIDDEN_SIZE_TRANSFORMER_SAMPLE
    )
    kwargs: dict = {}
    for f in fields(CohortGenes):
        if f.name == "architecture":
            kwargs[f.name] = arch
        elif f.name == "hidden_size":
            kwargs[f.name] = int(rng.choice(hidden_choices))
        elif f.name in ARCHITECTURE_GENE_NAMES:
            kwargs[f.name] = _sample_architecture_field(rng, f.name)
        elif f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            kwargs[f.name] = PHASE5_GENE_DEFAULTS[f.name]
        else:
            kwargs[f.name] = _sample_field(rng, f.name)
    return CohortGenes(**kwargs)


# ── Crossover / mutation ──────────────────────────────────────────────────


def crossover(
    parent_a: CohortGenes,
    parent_b: CohortGenes,
    rng: random.Random,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Uniform per-gene crossover. 50/50 parent pick on each gene.

    Disabled Phase 5 genes always take the cohort-wide default —
    never inherit a parent's value — keeping the cohort-default
    invariant under breeding.
    """
    child: dict = {}
    for f in fields(CohortGenes):
        if f.name in ARCHITECTURE_GENE_NAMES:
            # Structural — frozen within a lineage (HC#10). Inherit from
            # parent_a verbatim with NO rng draw, so adding these genes
            # leaves the pre-pbt RNG stream untouched (HC#1 byte-identity
            # when --breeding pbt is off; both gene-only-GA parents are
            # LSTM so this is a content no-op there too).
            child[f.name] = getattr(parent_a, f.name)
            continue
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            child[f.name] = PHASE5_GENE_DEFAULTS[f.name]
            continue
        if rng.random() < 0.5:
            child[f.name] = getattr(parent_a, f.name)
        else:
            child[f.name] = getattr(parent_b, f.name)
    return CohortGenes(**child)


def mutate(
    genes: CohortGenes,
    rng: random.Random,
    mutation_rate: float = 0.1,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Per-gene mutation. Each enabled gene is re-sampled with
    ``mutation_rate`` probability.

    Re-sampling ignores the parent value — for log-uniform floats, the
    mutated value is a fresh draw on the full log-uniform range; for
    categoricals, it's a fresh ``rng.choice`` (which can re-pick the
    same value, same as v1's mutation). ``mutation_rate=0`` is identity;
    ``mutation_rate=1`` always re-samples every enabled gene.

    Disabled Phase 5 genes are pinned to the cohort-wide default and
    never touched by mutation.
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            f"mutation_rate must be in [0, 1], got {mutation_rate}",
        )
    out: dict = {}
    for f in fields(CohortGenes):
        if f.name in ARCHITECTURE_GENE_NAMES:
            # Structural — never mutated (frozen within a lineage, HC#10).
            # NO rng draw → preserves the pre-pbt RNG stream (HC#1).
            out[f.name] = getattr(genes, f.name)
            continue
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            out[f.name] = PHASE5_GENE_DEFAULTS[f.name]
            continue
        if rng.random() < mutation_rate:
            out[f.name] = _sample_field(rng, f.name)
        else:
            out[f.name] = getattr(genes, f.name)
    return CohortGenes(**out)


# ── Validation helper ────────────────────────────────────────────────────


def assert_in_range(genes: CohortGenes) -> None:
    """Sanity-check every gene lands in the locked schema's range."""
    lo, hi = LEARNING_RATE_RANGE
    if not lo <= genes.learning_rate <= hi:
        raise ValueError(
            f"learning_rate {genes.learning_rate} outside [{lo}, {hi}]",
        )
    lo, hi = ENTROPY_COEFF_RANGE
    if not lo <= genes.entropy_coeff <= hi:
        raise ValueError(
            f"entropy_coeff {genes.entropy_coeff} outside [{lo}, {hi}]",
        )
    lo, hi = CLIP_RANGE_RANGE
    if not lo <= genes.clip_range <= hi:
        raise ValueError(
            f"clip_range {genes.clip_range} outside [{lo}, {hi}]",
        )
    lo, hi = GAE_LAMBDA_RANGE
    if not lo <= genes.gae_lambda <= hi:
        raise ValueError(
            f"gae_lambda {genes.gae_lambda} outside [{lo}, {hi}]",
        )
    lo, hi = VALUE_COEFF_RANGE
    if not lo <= genes.value_coeff <= hi:
        raise ValueError(
            f"value_coeff {genes.value_coeff} outside [{lo}, {hi}]",
        )
    if genes.mini_batch_size not in MINI_BATCH_SIZE_CHOICES:
        raise ValueError(
            f"mini_batch_size {genes.mini_batch_size} not in "
            f"{MINI_BATCH_SIZE_CHOICES}",
        )
    if genes.hidden_size not in HIDDEN_SIZE_VALID:
        raise ValueError(
            f"hidden_size {genes.hidden_size} not in {HIDDEN_SIZE_VALID}",
        )
    for name, (lo, hi) in _PHASE5_RANGES.items():
        value = getattr(genes, name)
        # Disabled-gene values land on the cohort-wide default. When the
        # GA's sampled range doesn't include the default (e.g.
        # ``mature_prob_loss_weight`` range [1.0, 5.0] with default 0.0),
        # accept the default explicitly so a disabled gene's pinned-zero
        # value isn't rejected by the validator. Enabled genes get a
        # fresh draw via ``_sample_field`` which lives in the declared
        # range; this branch only covers the disabled-default case.
        if value == PHASE5_GENE_DEFAULTS[name]:
            continue
        if not lo <= value <= hi:
            raise ValueError(
                f"{name} {value} outside [{lo}, {hi}]",
            )
    # Architecture genes (pbt-breeding Step 1b). Structural choices.
    if genes.architecture not in ARCHITECTURE_CHOICES:
        raise ValueError(
            f"architecture {genes.architecture!r} not in "
            f"{ARCHITECTURE_CHOICES}",
        )
    if genes.transformer_depth not in TRANSFORMER_DEPTH_CHOICES:
        raise ValueError(
            f"transformer_depth {genes.transformer_depth} not in "
            f"{TRANSFORMER_DEPTH_CHOICES}",
        )
    if genes.transformer_heads not in TRANSFORMER_HEADS_CHOICES:
        raise ValueError(
            f"transformer_heads {genes.transformer_heads} not in "
            f"{TRANSFORMER_HEADS_CHOICES}",
        )
    if genes.transformer_ctx_ticks not in TRANSFORMER_CTX_TICKS_CHOICES:
        raise ValueError(
            f"transformer_ctx_ticks {genes.transformer_ctx_ticks} not in "
            f"{TRANSFORMER_CTX_TICKS_CHOICES}",
        )
    if genes.transformer_ffn_mult not in TRANSFORMER_FFN_MULT_CHOICES:
        raise ValueError(
            f"transformer_ffn_mult {genes.transformer_ffn_mult} not in "
            f"{TRANSFORMER_FFN_MULT_CHOICES}",
        )
    if genes.transformer_pos_encoding not in TRANSFORMER_POS_ENCODING_CHOICES:
        raise ValueError(
            f"transformer_pos_encoding {genes.transformer_pos_encoding!r} not in "
            f"{TRANSFORMER_POS_ENCODING_CHOICES}",
        )
    if not isinstance(genes.predictor_lean_obs, bool):
        raise ValueError(
            f"predictor_lean_obs must be bool, got "
            f"{genes.predictor_lean_obs!r}",
        )
