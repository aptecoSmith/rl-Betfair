"""Discrete-action policy classes for the v2 trainer.

Phase 1, Session 02 deliverable. Drops the v1 70-dim continuous-multi-
head action space in favour of a single masked categorical over
``{NOOP, OPEN_BACK_i, OPEN_LAY_i, CLOSE_i}`` plus a small Beta-sized
``stake`` continuous head and a **per-runner value head**.

Locked design (see
``plans/rewrite/phase-1-policy-and-env-wiring/purpose.md``):

- The categorical sees a ``-inf`` mask on illegal actions; sampling
  from ``Categorical(logits=masked_logits)`` therefore never returns
  a masked index. **No post-hoc rejection.**
- The Beta heads emit ``alpha, beta > 1`` (via ``softplus + 1``) so
  the distribution is unimodal — no near-0/1 hairpin pathologies.
  The action caller draws ``s = Beta(alpha, beta).sample()`` and
  re-scales to ``[MIN_BET_STAKE, max_stake_cap]`` outside the policy.
- The value head is plain linear, no activation, output dim
  ``max_runners`` — Phase 2's GAE consumes per-runner returns directly.
- The hidden-state pack/slice helpers mirror the v1 contract
  (``CLAUDE.md`` §"Recurrent PPO: hidden-state protocol on update")
  so Phase 2's PPO update can batch per-transition states without
  knowing the architecture's batch-axis.

**Hard constraints (Session 02 prompt):**

* No training code lives here — no optimiser, no loss, no GAE.
* No imports from ``agents/`` (parallel tree; rewrite README §3).
* No env edits.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical

from agents_v2.action_space import DiscreteActionSpace
from env.betfair_env import MARKET_DIM, RUNNER_DIM, VELOCITY_DIM


__all__ = [
    "RISK_LOG_VAR_MIN",
    "RISK_LOG_VAR_MAX",
    "DiscretePolicyOutput",
    "BaseDiscretePolicy",
    "DiscreteLSTMPolicy",
]


# Clamp bounds on the risk head's log-var output. Ported verbatim from
# v1 ``agents/policy_network.py`` (Session 03 of the scalping-active-
# management plan). ``stddev = exp(0.5 * log_var)`` then spans ~£0.02
# to ~£7.39 on £100 stakes — covering realistic locked-P&L stddev while
# keeping ``exp(log_var)`` numerically well-behaved inside the NLL.
RISK_LOG_VAR_MIN: float = -8.0
RISK_LOG_VAR_MAX: float = 4.0


@dataclass(frozen=True)
class DiscretePolicyOutput:
    """Forward-pass result for any :class:`BaseDiscretePolicy` subclass.

    Attributes
    ----------
    logits:
        Raw categorical logits, shape ``(batch, action_space.n)``.
        These are pre-mask — useful for diagnostics that want to see
        the "would-pick" preference order including illegal actions.
    masked_logits:
        ``logits`` with ``-inf`` at every False entry of the mask
        passed to ``forward``. If no mask is supplied this equals
        ``logits``. Shape ``(batch, action_space.n)``.
    action_dist:
        :class:`torch.distributions.Categorical` built from
        ``masked_logits``. Sampling from it never returns a masked
        index (PyTorch handles the ``-inf`` softmax cleanly as long
        as at least one logit is finite — NOOP is always legal under
        the locked layout, so this invariant always holds).
    stake_alpha, stake_beta:
        Beta-distribution parameters for the stake continuous head,
        shape ``(batch,)`` each, both strictly ``> 1`` so the Beta
        is unimodal.
    value_per_runner:
        Per-runner critic, shape ``(batch, max_runners)``. Phase 2's
        GAE uses this directly; the global scalar value can be
        recovered as a sum-over-runners.
    new_hidden_state:
        Architecture-specific tuple. For LSTM-family policies:
        ``(h, c)`` each ``(num_layers, batch, hidden_size)``; for the
        transformer (Phase 1 follow-on): ``(buffer, valid_count)``.
        The exact shape is private to each subclass — only the
        2-tuple-of-tensors contract is shared.
    fill_prob_per_runner:
        Sigmoid output of ``fill_prob_head``, shape
        ``(batch, max_runners)``. Auxiliary BCE-trained per-runner
        forecast of "will the pair's second leg fill". Phase 7 S01
        port from v1 — feeds ``actor_head`` via column-concat.
    mature_prob_per_runner:
        Sigmoid output of ``mature_prob_head``, shape
        ``(batch, max_runners)``. Strict per-runner forecast of
        "will the pair mature naturally OR be closed by agent
        signal" — force-closed pairs are the negative class. Feeds
        ``actor_head`` via column-concat.
    direction_back_prob_per_runner, direction_lay_prob_per_runner:
        Sigmoid outputs of the per-side direction head, shape
        ``(batch, max_runners)`` each. BCE-trained on offline
        threshold-crossing labels (phase-13 S02): "did LTP move
        favourably for a back-first / lay-first scalp within the
        close horizon". Feed ``actor_head`` via column-concat.
    direction_back_logits_per_runner, direction_lay_logits_per_runner:
        Raw logits before sigmoid, shape ``(batch, max_runners)``
        each. Carried so the BCE loss can use
        ``binary_cross_entropy_with_logits(..., pos_weight=...)``
        for numerical stability + class balance.
    predicted_locked_pnl_per_runner:
        Mean channel of ``risk_head``, shape ``(batch, max_runners)``.
        Per-runner predicted locked-P&L. Does NOT feed ``actor_head``;
        only shapes the shared backbone via the Gaussian NLL gradient.
    predicted_locked_log_var_per_runner:
        Log-variance channel of ``risk_head``, shape
        ``(batch, max_runners)``, clamped to
        ``[RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX]`` at the forward
        boundary so NLL consumers never see an unsafe value.
    """

    logits: torch.Tensor
    masked_logits: torch.Tensor
    action_dist: Categorical
    stake_alpha: torch.Tensor
    stake_beta: torch.Tensor
    value_per_runner: torch.Tensor
    new_hidden_state: tuple[torch.Tensor, ...]
    fill_prob_per_runner: torch.Tensor
    mature_prob_per_runner: torch.Tensor
    predicted_locked_pnl_per_runner: torch.Tensor
    predicted_locked_log_var_per_runner: torch.Tensor
    direction_back_prob_per_runner: torch.Tensor
    direction_lay_prob_per_runner: torch.Tensor
    direction_back_logits_per_runner: torch.Tensor
    direction_lay_logits_per_runner: torch.Tensor
    # fc-cost-probe D (2026-05-17): per-runner strict-fc head outputs.
    # ``None`` when the head is disabled (default) — keeps the
    # PolicyOutput contract backward-compatible. When enabled, both
    # are shape ``(batch, max_runners)``.
    fc_prob_per_runner: torch.Tensor | None = None
    fc_prob_logits_per_runner: torch.Tensor | None = None


class BaseDiscretePolicy(nn.Module, abc.ABC):
    """Abstract base for v2 discrete-action policies.

    The class is intentionally thin: it pins the ``__init__`` signature
    (``obs_dim``, ``action_space``, ``hidden_size``) and the forward
    contract (returns :class:`DiscretePolicyOutput`). Subclasses pick
    a backbone (LSTM, TimeLSTM, Transformer …) and implement
    ``forward`` + ``init_hidden`` + the pack/slice helpers.

    The pack/slice helpers default to ``dim=0`` concat / index-select
    — correct for the transformer's ``(buffer, valid_count)`` hidden
    state. The LSTM subclass overrides them to use ``dim=1`` (the
    ``nn.LSTM`` batch axis).
    """

    def __init__(
        self,
        obs_dim: int,
        action_space: DiscreteActionSpace,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim!r}")
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size!r}",
            )
        self.obs_dim = int(obs_dim)
        self.action_space = action_space
        self.max_runners = action_space.max_runners
        self.hidden_size = int(hidden_size)

    # ── Interface ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    def init_hidden(
        self, batch: int = 1,
    ) -> tuple[torch.Tensor, ...]:
        """Return a zero-initialised hidden state for ``batch`` sequences."""

    @abc.abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, ...] | None = None,
        mask: torch.Tensor | None = None,
    ) -> DiscretePolicyOutput:
        """Run the backbone + heads.

        Parameters
        ----------
        obs:
            ``(batch, obs_dim)`` for a single timestep, or
            ``(batch, ctx, obs_dim)`` for a sequence — subclasses use
            the last timestep's output for the heads.
        hidden_state:
            Architecture-specific tuple, or ``None`` for a fresh
            zero-init state. The default-init path moves the state to
            ``obs.device`` so callers don't have to.
        mask:
            ``(batch, action_space.n)`` boolean. ``True`` = legal,
            ``False`` = masked. Applied at the logits level
            (``-inf``), never as post-hoc rejection.
        """

    # ── Hidden-state batching (PPO update path) ───────────────────────────

    @staticmethod
    def pack_hidden_states(
        states: list[tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        """Concatenate per-transition hidden states into one batched tuple.

        Default implementation concatenates each tuple slot along
        ``dim=0`` — correct for transformer-style states whose batch
        axis is dim 0. The LSTM subclass overrides this to use
        ``dim=1``.
        """
        if not states:
            raise ValueError("pack_hidden_states: states list is empty")
        n_slots = len(states[0])
        return tuple(
            torch.cat([s[i] for s in states], dim=0) for i in range(n_slots)
        )

    @staticmethod
    def slice_hidden_states(
        packed: tuple[torch.Tensor, ...],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Slice a packed hidden state by integer ``indices`` along dim 0."""
        return tuple(t.index_select(0, indices) for t in packed)

    @staticmethod
    def pack_hidden_buffer(
        buffers: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Pack pre-stacked time-axis-0 buffers into the per-policy packed form.

        Phase 4 Session 06 (2026-05-02). Equivalent to
        ``pack_hidden_states([tuple(buf[t] for buf in buffers) for t in
        range(n_steps)])`` but skips the per-tick list-of-tuples
        construction and the N-way ``torch.cat`` over small slices —
        the input buffers already hold the per-tick states in
        contiguous memory.

        Default implementation handles BasePolicy-style states whose
        per-tick element shape has the batch axis at dim 0 (e.g.
        transformer state ``(1, ctx, d_model)``). The pre-stacked
        buffer's shape is ``(n_steps, 1, *rest)`` and the packed form
        is ``(n_steps, *rest)`` — equivalent to concat along dim=0 of
        N copies of a ``(1, *rest)`` tensor. ``.squeeze(1)`` is a view
        (no copy).

        The LSTM subclass overrides this to handle the
        ``(n_steps, num_layers, 1, hidden)`` → ``(num_layers, n_steps,
        hidden)`` reshape.
        """
        if not buffers:
            raise ValueError("pack_hidden_buffer: buffers tuple is empty")
        return tuple(buf.squeeze(1) for buf in buffers)


# ── Discrete LSTM ──────────────────────────────────────────────────────────


class DiscreteLSTMPolicy(BaseDiscretePolicy):
    """Single-layer LSTM for the v2 discrete action space.

    Architecture (Phase 7 S01)::

        Linear(obs_dim → hidden) → ReLU
        LSTM(hidden, hidden, num_layers=1, batch_first=True)
            → lstm_last : (batch, hidden)

        Auxiliary per-runner heads (all consume lstm_last):
            fill_prob_head   : Linear(hidden → max_runners)
            mature_prob_head : Linear(hidden → max_runners)
            risk_head        : Linear(hidden → max_runners * 2)

        Per-runner actor (feeds the categorical's per-runner classes):
            runner_slot_embedding : Embedding(max_runners → embed_dim)
            actor_head            : Sequential(
                                       Linear(embed + hidden + 2 → mlp),
                                       ReLU,
                                       Linear(mlp → 3),
                                    )
            input per slot i = [slot_emb_i, lstm_last,
                                fill_prob_i, mature_prob_i]
            output per slot i = [OB_i, OL_i, CL_i] logits
        NOOP logit:
            noop_head : Linear(hidden → 1)
        Final logits = cat([NOOP, OB_0..R-1, OL_0..R-1, CL_0..R-1])

        Other heads (unchanged):
            stake_alpha_head : Linear(hidden → 1) → softplus → +1
            stake_beta_head  : Linear(hidden → 1) → softplus → +1
            value_head       : Linear(hidden → max_runners)

    The per-runner actor pathway ports v1's contract: ``fill_prob`` and
    ``mature_prob`` are concat'd column-wise into the per-runner actor
    input so the BCE-trained heads' confidence scores feed the action
    selection for the runner they describe. ``risk_head`` does NOT feed
    the actor — its only training signal is the Gaussian NLL auxiliary
    (Phase 7 S02) which shapes the shared LSTM backbone via gradient.

    Hidden state ``(h, c)``, each ``(num_layers=1, batch, hidden_size)``
    — the standard ``nn.LSTM`` layout.
    """

    architecture_name = "discrete_lstm_v2"
    description = (
        "v2 discrete-action policy: single-layer LSTM (hidden=128 by "
        "default), per-runner actor MLP fed by fill_prob + mature_prob "
        "BCE heads, risk head emitting locked-P&L (mean, log_var), "
        "Beta stake head, per-runner value head."
    )

    DEFAULT_RUNNER_EMBED_DIM: int = 16
    DEFAULT_ACTOR_MLP_HIDDEN: int = 64

    # Phase-14 S03 (2026-05-07). The direction-gate gene's allowed
    # range. Lower bound 0.5 = effectively-no-gate (positive-class
    # density ~22% means few rows have max(P_back, P_lay) ≥ 0.5
    # at fresh init). Upper bound 0.95 caps the strictest gene draw
    # at the level where the supervised probe still sees ~233-1554
    # opens/day — preventing an agent from drawing 0.99+ and
    # starving PPO of training signal. See phase-14
    # hard_constraints §10.
    DIRECTION_GATE_THRESHOLD_MIN: float = 0.5
    DIRECTION_GATE_THRESHOLD_MAX: float = 0.95

    def __init__(
        self,
        obs_dim: int,
        action_space: DiscreteActionSpace,
        hidden_size: int = 128,
        runner_embed_dim: int | None = None,
        actor_mlp_hidden: int | None = None,
        direction_gate_enabled: bool = False,
        direction_gate_threshold: float = 0.5,
        enable_fc_prob_head: bool = False,
        runner_dim: int | None = None,
        frozen_direction_head_path: "Path | None" = None,
    ) -> None:
        super().__init__(obs_dim, action_space, hidden_size)
        self.num_layers = 1
        # Phase-15 fix (2026-05-24): which RUNNER_KEYS variant the env
        # uses (full=143 or lean=23) drives the per-slot input dim
        # for direction_prob_head + the runner_block slicing in
        # forward(). Default to the module-level RUNNER_DIM (full obs)
        # for back-compat with existing test callers; lean-obs
        # callers (the cohort, after Phase-15) MUST pass
        # runner_dim=23 (or env.active_runner_dim). Bug history:
        # before this, the head was built with LayerNorm(143) +
        # Linear(143, …) regardless of obs layout, and the runner-
        # block extractor fell into a zero-pad test-mode fallback
        # under lean-obs — head input was structurally garbage.
        self._runner_dim = int(
            runner_dim if runner_dim is not None else RUNNER_DIM,
        )
        if self._runner_dim <= 0:
            raise ValueError(
                f"runner_dim must be positive, got {self._runner_dim!r}",
            )
        self.runner_embed_dim = int(
            runner_embed_dim
            if runner_embed_dim is not None
            else self.DEFAULT_RUNNER_EMBED_DIM,
        )
        self.actor_mlp_hidden = int(
            actor_mlp_hidden
            if actor_mlp_hidden is not None
            else self.DEFAULT_ACTOR_MLP_HIDDEN,
        )
        if self.runner_embed_dim <= 0:
            raise ValueError(
                f"runner_embed_dim must be positive, got "
                f"{self.runner_embed_dim!r}",
            )
        if self.actor_mlp_hidden <= 0:
            raise ValueError(
                f"actor_mlp_hidden must be positive, got "
                f"{self.actor_mlp_hidden!r}",
            )

        self.input_proj = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Auxiliary per-runner heads (Phase 7 S01).
        # fill_prob_head + mature_prob_head feed actor_head via
        # column-concat; risk_head does not (CLAUDE.md "fill_prob feeds
        # actor_head" / "mature_prob_head feeds actor_head" /
        # purpose.md §3 risk-head-as-side-channel).
        self.fill_prob_head = nn.Linear(
            self.hidden_size, self.max_runners,
        )
        self.mature_prob_head = nn.Linear(
            self.hidden_size, self.max_runners,
        )
        self.risk_head = nn.Linear(
            self.hidden_size, self.max_runners * 2,
        )
        # fc-cost-probe D (2026-05-17): per-runner aux head with
        # STRICT label `1.0 if force_closed else 0.0`. Feeds
        # actor_head via column-concat (5th aux column) when
        # enabled. Default disabled → architecture byte-identical
        # to pre-probe-D so existing weight loads still work.
        # When enabled, actor_input grows by 1 column and the
        # arch-hash check refuses pre-probe-D weights (by design;
        # probe D launches with fresh init).
        self.enable_fc_prob_head: bool = bool(enable_fc_prob_head)
        if self.enable_fc_prob_head:
            self.fc_prob_head = nn.Linear(
                self.hidden_size, self.max_runners,
            )
        # Phase-15 S01 (2026-05-08). Per-runner direction head fed
        # the runner's RAW per-runner FEATURE SLICE rather than
        # ``concat([slot_emb_i, lstm_last])``. The supervised
        # ``tools/direction_features_probe.py`` extracted 24-94×
        # top-quintile lift on raw per-runner inputs, vs phase-14's
        # LSTM-bottlenecked pathway which the smoke + probeAB
        # showed could not learn at cohort scale (BCE flat ~1.04
        # across all generations). See
        # ``plans/rewrite/phase-15-direction-head-feature-slice/
        # purpose.md`` for the diagnosis.
        #
        # Input shape per slot: ``(RUNNER_DIM,)`` — the raw
        # ``RUNNER_KEYS`` block for that runner, sliced from obs.
        # Output: 2 logits per slot ``(direction_back_logit_i,
        # direction_lay_logit_i)``. Sigmoid feeds actor_head via
        # column-concat (the existing +4 wiring is preserved —
        # actor_head's input dim is unchanged).
        #
        # Architecture-hash break vs phase-14: the first Linear's
        # input dim shrinks from ``runner_embed + hidden`` to
        # ``RUNNER_DIM``. Pre-S01 checkpoints fail strict load by
        # design — see ``plans/rewrite/phase-15-direction-head-
        # feature-slice/hard_constraints.md §1``.
        #
        # Phase-15 S01 (amendment 2026-05-08): prepend a LayerNorm
        # over the per-runner feature slice before the first
        # Linear. The S02 smoke surfaced direction BCE saturating
        # at 4-12 (vs the probe's 0.4-0.6) because raw obs values
        # for vol_delta_60 sit in the [10², 10³] range — kaiming-
        # init weights on the first Linear push pre-activations
        # into the thousands and the sigmoid saturates against
        # truth. Other v2 heads (fill / mature / risk / value)
        # read ``lstm_last``, which is post-``input_proj``'s
        # learned linear scaling, so they don't see the raw
        # heavy-tail scales. The probe used per-day pd.std
        # normalisation; LayerNorm here achieves the equivalent
        # squash without dataset-stats bookkeeping
        # (sense_check.md item 2 spec correction; see also
        # lessons_learnt.md "Saturation from raw obs scales").
        # Default head: one hidden layer at actor_mlp_hidden. Used
        # when no frozen-head manifest is supplied. When a manifest
        # IS supplied, the head is REBUILT below to match whatever
        # `architecture.hidden_dims` the manifest specifies — the
        # sweep variants (C0-C20) explored 1- and 2-hidden-layer
        # designs with varying widths, and the winner (C11) is a
        # 2-layer head (256 → 128). See
        # `plans/direction-head-architecture-sweep/`.
        self.direction_prob_head = nn.Sequential(
            nn.LayerNorm(self._runner_dim),
            nn.Linear(self._runner_dim, self.actor_mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.actor_mlp_hidden, 2),
        )

        # 2026-05-24: optional load of a pre-trained shared direction
        # head, frozen. See ``plans/shared-direction-head/``. The
        # caller (cohort worker) passes a directory containing
        # ``weights.pt`` + ``manifest.json``. The manifest's
        # ``architecture.hidden_dims`` field determines the head's
        # shape — when it differs from the default, we REBUILD the
        # head before loading weights. Frozen via
        # ``requires_grad_(False)`` so gradient flows THROUGH the
        # head (actor_head reads its output) but weights stay fixed.
        self._frozen_direction_head = False
        if frozen_direction_head_path is not None:
            from pathlib import Path as _Path
            import torch as _torch
            import json as _json
            mp = _Path(frozen_direction_head_path)
            if mp.is_dir():
                weights_path = mp / "weights.pt"
                manifest_path = mp / "manifest.json"
            else:
                weights_path = mp
                manifest_path = mp.parent / "manifest.json"
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"frozen_direction_head_path: weights.pt not "
                    f"found at {weights_path}",
                )

            # Read the manifest (when present) to recover the head's
            # architecture. Pre-sweep manifests with a single
            # hidden layer continue to work because hidden_dims
            # defaults match the policy's default head shape.
            hidden_dims: list[int] = [self.actor_mlp_hidden]
            input_dim_manifest: int | None = None
            if manifest_path.exists():
                _manifest = _json.loads(
                    manifest_path.read_text(encoding="utf-8"),
                )
                _arch = _manifest.get("architecture", {}) or {}
                _hd = _arch.get("hidden_dims")
                if _hd:
                    hidden_dims = [int(h) for h in _hd]
                input_dim_manifest = (
                    int(_arch["input_dim"])
                    if "input_dim" in _arch else None
                )
                # Refuse to load a head trained for a different
                # per-runner input dim — the lean-obs/full-obs
                # bug class (see plans/direction-predictor-label-
                # alignment/) bit us with exactly this kind of
                # silent dim mismatch.
                if (
                    input_dim_manifest is not None
                    and input_dim_manifest != self._runner_dim
                ):
                    raise ValueError(
                        f"direction-head manifest input_dim="
                        f"{input_dim_manifest} does not match the "
                        f"policy's per-runner dim {self._runner_dim}. "
                        f"Re-train the head against the runner-dim "
                        f"the policy uses (lean obs = 23, full obs "
                        f"= 143)."
                    )

            # Rebuild the head to match the manifest's hidden_dims.
            # Layer pattern: LayerNorm(input) -> [Linear -> ReLU]+ ->
            # Linear(_, 2). Same shape as the training script's
            # SimpleMLP / DeepMLP / WideMLP — see
            # ``scripts/train_direction_head.py``.
            layers: list[nn.Module] = [nn.LayerNorm(self._runner_dim)]
            prev = self._runner_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, 2))
            self.direction_prob_head = nn.Sequential(*layers)

            state = _torch.load(
                weights_path, map_location="cpu", weights_only=True,
            )
            self.direction_prob_head.load_state_dict(state, strict=True)
            for param in self.direction_prob_head.parameters():
                param.requires_grad_(False)
            self._frozen_direction_head = True

        # Phase-15 S01: precompute the obs-slice offsets for the
        # per-runner feature block. The env's static obs is laid
        # out as ``[market_vec, vel_vec, runner_vec, agent_state,
        # position_vec]``; the runner block starts at
        # ``MARKET_DIM + VELOCITY_DIM`` and spans
        # ``max_runners * RUNNER_DIM`` floats.
        #
        # ``_runner_block_full_size`` is always
        # ``max_runners * RUNNER_DIM`` — the head's per-slot
        # input dim. If ``obs_dim`` is smaller than the env's
        # natural layout (typically a test placeholder), the
        # forward path zero-pads the available slice up to that
        # size so the head still receives a well-shaped tensor.
        # Real production callers pass obs_dim derived from
        # ``env.observation_space``, which is always >= the
        # layout requirement; the pad only activates in unit
        # tests that build a minimal LSTM without caring about
        # feature semantics. The strict phase-15 regression
        # tests use a properly-sized obs (see
        # ``tests/test_v2_direction_prob_in_actor.py``).
        natural_offset = MARKET_DIM + VELOCITY_DIM
        natural_size = self.max_runners * self._runner_dim
        self._runner_block_full_size = natural_size
        if natural_offset + natural_size <= self.obs_dim:
            self._runner_block_offset = natural_offset
            self._runner_block_size = natural_size
        else:
            # Test-mode fallback: anchor at 0, take whatever fits.
            self._runner_block_offset = 0
            self._runner_block_size = max(0, min(self.obs_dim, natural_size))
            # 2026-05-24 guard: in production this fallback is the
            # symptom of an env/policy obs-layout mismatch (which
            # silently bricked the Phase-15 cohort for 16 days
            # before being caught). Tests deliberately use small
            # obs_dim to skip the env build — heuristic threshold of
            # 256 distinguishes "intentional test" from "production
            # mistake." Adjust if a future obs config ever lives
            # below this threshold legitimately.
            _PROD_OBS_DIM_THRESHOLD = 256
            if self.obs_dim >= _PROD_OBS_DIM_THRESHOLD:
                import warnings as _warnings
                _warnings.warn(
                    f"DiscreteLSTMPolicy fell into the runner-block "
                    f"test-mode fallback at obs_dim={self.obs_dim}. "
                    f"Expected per-runner layout "
                    f"max_runners={self.max_runners} × "
                    f"runner_dim={self._runner_dim} = {natural_size} "
                    f"would not fit after MARKET+VELOCITY offset "
                    f"{natural_offset} within obs_dim. Most likely "
                    f"the caller passed the wrong runner_dim — "
                    f"production callers should pass "
                    f"runner_dim=env.active_runner_dim. See "
                    f"plans/direction-predictor-label-alignment/.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Per-runner actor (Phase 7 S01). The flat categorical from
        # Phase 1 is now produced by stitching: NOOP from noop_head + a
        # per-runner head producing [OB_i, OL_i, CL_i] per slot. The
        # per-runner head sees a learned slot embedding so it can
        # distinguish runner identities (v1 carries a per-slot runner
        # embedding for the same reason).
        self.runner_slot_embedding = nn.Embedding(
            self.max_runners, self.runner_embed_dim,
        )
        # +4 (was +2): fill_prob, mature_prob, direction_back_prob,
        # direction_lay_prob (phase-13 S03).
        # +5 when enable_fc_prob_head=True: adds fc_prob column (probe D).
        _aux_cols = 5 if self.enable_fc_prob_head else 4
        actor_input_dim = self.runner_embed_dim + self.hidden_size + _aux_cols
        self.actor_head = nn.Sequential(
            nn.Linear(actor_input_dim, self.actor_mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.actor_mlp_hidden, 3),  # [OB, OL, CL]
        )
        self.noop_head = nn.Linear(self.hidden_size, 1)

        self.stake_alpha_head = nn.Linear(self.hidden_size, 1)
        self.stake_beta_head = nn.Linear(self.hidden_size, 1)
        self.value_head = nn.Linear(self.hidden_size, self.max_runners)

        # Phase-14 S03 (2026-05-07). Direction-gate config. Disabled
        # by default — when False the policy is byte-identical to
        # phase-14 S01+S02 without S03 (no mask applied). When
        # enabled, OPEN_BACK_i / OPEN_LAY_i logits are masked
        # (-inf) where ``max(P_back_i, P_lay_i) < threshold``.
        # NOOP and CLOSE_i are NEVER gated — see hard_constraints
        # §14, §15. Threshold clamped to
        # [DIRECTION_GATE_THRESHOLD_MIN, _MAX] at construction so
        # callers passing wider values don't bypass the cap.
        self.direction_gate_enabled = bool(direction_gate_enabled)
        self.direction_gate_threshold = float(
            max(
                self.DIRECTION_GATE_THRESHOLD_MIN,
                min(
                    self.DIRECTION_GATE_THRESHOLD_MAX,
                    float(direction_gate_threshold),
                ),
            ),
        )
        # Phase-14 S06 (2026-05-07). Effective gate threshold —
        # operator-controlled warmup. The trainer pokes a value in
        # via ``set_effective_gate_threshold`` once per episode;
        # absent that call the policy uses the gene value directly.
        # Mirrors bc_target_entropy_warmup_eps's per-episode poke
        # pattern. The smoke surfaced the cold-start collapse:
        # 3 of 4 agents at gate threshold ≥0.88 emitted ZERO bets
        # because fresh-init head sigmoid sits ~0.5 and T≥0.88
        # masks everything. Anneal from 0.5 → gene over the first
        # N updates lets PPO see opens at cold-start.
        self._effective_gate_threshold: float | None = None

    # ── Hidden state ──────────────────────────────────────────────────────

    def init_hidden(
        self, batch: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch, self.hidden_size)
        c = torch.zeros(self.num_layers, batch, self.hidden_size)
        return h, c

    @staticmethod
    def pack_hidden_states(
        states: list[tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LSTM hidden state batches along dim 1, not dim 0."""
        if not states:
            raise ValueError("pack_hidden_states: states list is empty")
        h0 = torch.cat([s[0] for s in states], dim=1)
        h1 = torch.cat([s[1] for s in states], dim=1)
        return h0, h1

    @staticmethod
    def slice_hidden_states(
        packed: tuple[torch.Tensor, ...],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            packed[0].index_select(1, indices),
            packed[1].index_select(1, indices),
        )

    @staticmethod
    def pack_hidden_buffer(
        buffers: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LSTM-specific: ``(n_steps, num_layers, 1, hidden)`` → ``(num_layers, n_steps, hidden)``.

        Phase 4 Session 06 (2026-05-02). The pre-stacked buffer's shape
        is ``(n_steps, num_layers, 1, hidden)`` (Session 04). The packed
        form expected by ``slice_hidden_states`` (and downstream by
        ``policy.forward``) is ``(num_layers, n_steps, hidden)`` —
        equivalent to ``torch.cat([s[k] for s in states], dim=1)`` over
        N states each ``(num_layers, 1, hidden)``.

        ``.squeeze(2)`` drops the singleton ``1`` (the per-tick batch
        axis); ``.permute(1, 0, 2)`` swaps the leading time axis with
        the layers axis. Both are views — no data movement.
        """
        if not buffers or len(buffers) != 2:
            raise ValueError(
                f"pack_hidden_buffer: expected 2-tuple (h, c), got "
                f"{len(buffers) if buffers else 0} buffers",
            )
        h_buf, c_buf = buffers
        return (
            h_buf.squeeze(2).permute(1, 0, 2),
            c_buf.squeeze(2).permute(1, 0, 2),
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: tuple[torch.Tensor, ...] | None = None,
        mask: torch.Tensor | None = None,
        apply_direction_gate: bool | None = None,
    ) -> DiscretePolicyOutput:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
        if obs.dim() != 3:
            raise ValueError(
                f"obs must be (batch, obs_dim) or (batch, ctx, obs_dim), "
                f"got shape {tuple(obs.shape)}",
            )
        batch, _ctx, obs_dim = obs.shape
        if obs_dim != self.obs_dim:
            raise ValueError(
                f"obs_dim mismatch: expected {self.obs_dim}, got {obs_dim}",
            )

        if hidden_state is None:
            h0, c0 = self.init_hidden(batch)
            hidden_state = (h0.to(obs.device), c0.to(obs.device))

        # Backbone
        flat = obs.reshape(batch * obs.shape[1], obs_dim)
        proj = self.input_proj(flat).reshape(batch, obs.shape[1], -1)
        lstm_out, new_hidden = self.lstm(proj, hidden_state)
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden)

        # ── Auxiliary heads (Phase 7 S01) ─────────────────────────────
        # fill_prob and mature_prob feed actor_head; risk_head does not.
        fill_logit = self.fill_prob_head(lstm_last)        # (batch, R)
        mature_logit = self.mature_prob_head(lstm_last)    # (batch, R)
        fill_prob = torch.sigmoid(fill_logit)
        mature_prob = torch.sigmoid(mature_logit)
        # fc-cost-probe D (2026-05-17): strict-fc aux head.
        if self.enable_fc_prob_head:
            fc_prob_logit = self.fc_prob_head(lstm_last)   # (batch, R)
            fc_prob = torch.sigmoid(fc_prob_logit)
        else:
            fc_prob_logit = None
            fc_prob = None

        # Phase-14 S01 / phase-15 S01 — per-runner direction head.
        # Slot embeddings are still computed here because actor_head
        # uses them below; the direction head no longer reads them
        # (phase-15 feeds the head the runner's raw feature slice
        # instead — see hard_constraints §2).
        slot_idx = torch.arange(self.max_runners, device=lstm_last.device)
        runner_embs = self.runner_slot_embedding(slot_idx)  # (R, embed)
        runner_embs_b = runner_embs.unsqueeze(0).expand(
            batch, -1, -1,
        )  # (batch, R, embed)
        lstm_expanded = lstm_last.unsqueeze(1).expand(
            -1, self.max_runners, -1,
        )  # (batch, R, hidden)

        # Phase-15 S01: slice the per-runner feature block out of
        # obs and feed it to the direction head DIRECTLY. We use
        # the LAST timestep's obs (``obs[:, -1, :]``) because the
        # head's forecast is "given current state, where is price
        # going?" — past timesteps are encoded in lstm_last for
        # actor_head's use, but the direction head wants only the
        # current per-runner numbers, matching the supervised
        # probe's regime.
        #
        # ``runner_feats_raw`` shape: (batch, R, RUNNER_DIM). Same
        # block v1's policies extract for their actor pathway —
        # see ``agents/policy_network.py:691-706``.
        obs_last = obs[:, -1, :]  # (batch, obs_dim)
        runners_flat = obs_last[
            :,
            self._runner_block_offset:
            self._runner_block_offset + self._runner_block_size,
        ]
        # Zero-pad if obs is smaller than the env's natural layout
        # (test-mode fallback). Production-sized obs hits the
        # ``size == full_size`` fast path.
        if self._runner_block_size < self._runner_block_full_size:
            pad_width = (
                self._runner_block_full_size - self._runner_block_size
            )
            pad = torch.zeros(
                batch, pad_width,
                dtype=runners_flat.dtype,
                device=runners_flat.device,
            )
            runners_flat = torch.cat([runners_flat, pad], dim=-1)
        runner_feats_raw = runners_flat.view(
            batch, self.max_runners, self._runner_dim,
        )

        # Direction head reads the per-runner feature slice
        # directly. NO concat with slot_emb / lstm_last — the
        # whole point of phase 15 is to bypass that bottleneck
        # (hard_constraints §2).
        direction_input_flat = runner_feats_raw.reshape(
            batch * self.max_runners, self._runner_dim,
        )
        direction_logits_flat = self.direction_prob_head(
            direction_input_flat,
        )  # (batch * R, 2)
        direction_logits = direction_logits_flat.view(
            batch, self.max_runners, 2,
        )
        direction_back_logits = direction_logits[..., 0]
        direction_lay_logits = direction_logits[..., 1]
        direction_back_prob = torch.sigmoid(direction_back_logits)
        direction_lay_prob = torch.sigmoid(direction_lay_logits)

        # Risk head: (batch, R * 2) → (batch, R, 2). Clamp log_var at
        # the forward boundary so PolicyOutput consumers (UI, parquet,
        # NLL) never see an unsafe value.
        risk_out = self.risk_head(lstm_last)
        risk_out = risk_out.view(batch, self.max_runners, 2)
        risk_mean = risk_out[..., 0]
        risk_log_var = risk_out[..., 1].clamp(
            RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX,
        )

        # ── Per-runner actor (Phase 7 S01) ────────────────────────────
        # Build per-runner inputs: [slot_emb_i, lstm_last,
        # fill_prob_i, mature_prob_i, direction_back_i, direction_lay_i]
        # for each runner i.
        #
        # ``fill_prob`` / ``mature_prob`` are NOT detached — surrogate-
        # loss gradient flows back through fill_prob_head /
        # mature_prob_head (CLAUDE.md "fill_prob feeds actor_head"
        # §"Do not detach"). Those heads do not gate actions, so the
        # actor's "learn to use the prediction" pathway is benign.
        #
        # Phase-15 S01 (amendment 2026-05-08): ``direction_back_prob``
        # / ``direction_lay_prob`` ARE detached when feeding actor_input.
        # The direction head's output also drives the gate — masking
        # OPEN_BACK/OPEN_LAY logits below threshold — which creates a
        # self-referential loop: PPO learns "high direction_prob keeps
        # OPEN actions legal" and pulls the head's outputs high
        # regardless of truth. BCE pulls toward (mostly zero) labels.
        # The two pulls reach an equilibrium near p≈0.65 with
        # end-of-day BCE ≈ -log(0.35) ≈ 1.05 — observed in S02 smoke
        # v3 where 30× more BCE weight produced only 0.003 BCE delta.
        # Detaching breaks the self-referential loop: BCE becomes the
        # SOLE training signal for direction_prob_head, mirroring the
        # supervised probe's regime. The gate still reads the
        # un-detached values for masking (no loss flows through the
        # mask), but the actor's surrogate-loss gradient terminates at
        # the detach. Reverse this when / if direction head calibrates
        # to the probe's 0.4-0.6 BCE — at that point the actor's
        # "learn to use the prediction" pathway has value to add. See
        # ``plans/rewrite/phase-15-direction-head-feature-slice/
        # lessons_learnt.md`` "PPO/BCE self-referential gate loop".
        _actor_input_parts = [
            runner_embs_b,
            lstm_expanded,
            fill_prob.unsqueeze(-1),
            mature_prob.unsqueeze(-1),
            direction_back_prob.detach().unsqueeze(-1),
            direction_lay_prob.detach().unsqueeze(-1),
        ]
        # fc-cost-probe D (2026-05-17): fc_prob feeds actor_head when
        # enabled, on the same NOT-detached pattern as fill_prob /
        # mature_prob (surrogate-loss gradient flows back through
        # fc_prob_head so the actor can learn to use the prediction).
        if self.enable_fc_prob_head:
            _actor_input_parts.append(fc_prob.unsqueeze(-1))
        actor_input = torch.cat(_actor_input_parts, dim=-1)  # (batch, R, embed + hidden + 4|5)
        per_runner_logits = self.actor_head(actor_input)  # (batch, R, 3)
        ob_logits = per_runner_logits[..., 0]  # (batch, R)
        ol_logits = per_runner_logits[..., 1]
        cl_logits = per_runner_logits[..., 2]

        # NOOP logit (no per-runner conditioning).
        noop_logit = self.noop_head(lstm_last)  # (batch, 1)

        # Stitch into the flat action layout that DiscreteActionSpace
        # uses: [NOOP, OB_0..R-1, OL_0..R-1, CL_0..R-1].
        logits = torch.cat(
            [noop_logit, ob_logits, ol_logits, cl_logits], dim=-1,
        )  # (batch, 1 + 3 * R)
        masked_logits = self._apply_mask(logits, mask)
        # Phase-14 S03: direction-confidence gate. When enabled,
        # blocks OPEN_BACK_i / OPEN_LAY_i actions whose runner's
        # max(P_back, P_lay) sits below the configured threshold.
        # NOOP and CLOSE_i are NEVER gated. The mask AND-s with
        # the legality mask (both must pass) so the gate never
        # overrides legality. See hard_constraints §11–§15.
        #
        # Phase-14 S05: ``apply_direction_gate`` lets the caller
        # opt out of the in-forward gate recomputation. When the
        # caller has supplied a pre-captured rollout-time mask via
        # ``mask=``, they pass ``apply_direction_gate=False`` so
        # the logits aren't gated twice (and so the trainer's
        # ``log_pi_new`` is computed against the SAME distribution
        # ``log_pi_old`` came from — without this, the in-forward
        # gate's recompute drifts during PPO weight updates and
        # produces approx_kl=inf; see findings.md).
        if apply_direction_gate is None:
            apply_direction_gate = self.direction_gate_enabled
        if apply_direction_gate:
            masked_logits = self._apply_direction_gate(
                masked_logits,
                direction_back_prob=direction_back_prob,
                direction_lay_prob=direction_lay_prob,
            )
        dist = Categorical(logits=masked_logits)

        # Stake Beta heads — softplus + 1 keeps alpha, beta > 1 (unimodal).
        # ``squeeze(-1)`` so shape is (batch,), matching the contract.
        stake_alpha = nn.functional.softplus(
            self.stake_alpha_head(lstm_last),
        ).squeeze(-1) + 1.0
        stake_beta = nn.functional.softplus(
            self.stake_beta_head(lstm_last),
        ).squeeze(-1) + 1.0

        value_per_runner = self.value_head(lstm_last)  # (batch, max_runners)

        return DiscretePolicyOutput(
            logits=logits,
            masked_logits=masked_logits,
            action_dist=dist,
            stake_alpha=stake_alpha,
            stake_beta=stake_beta,
            value_per_runner=value_per_runner,
            new_hidden_state=new_hidden,
            fill_prob_per_runner=fill_prob,
            mature_prob_per_runner=mature_prob,
            predicted_locked_pnl_per_runner=risk_mean,
            predicted_locked_log_var_per_runner=risk_log_var,
            direction_back_prob_per_runner=direction_back_prob,
            direction_lay_prob_per_runner=direction_lay_prob,
            direction_back_logits_per_runner=direction_back_logits,
            direction_lay_logits_per_runner=direction_lay_logits,
            fc_prob_per_runner=fc_prob,
            fc_prob_logits_per_runner=fc_prob_logit,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _apply_mask(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.shape != logits.shape:
            # Allow a single (n,) mask broadcast across the batch, which
            # is the common rollout-time shape (one mask per env step).
            if mask.dim() == 1 and mask.shape[0] == logits.shape[1]:
                mask = mask.unsqueeze(0).expand_as(logits)
            else:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} incompatible with "
                    f"logits shape {tuple(logits.shape)}",
                )
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # ``masked_fill`` with ~mask: True (legal) stays as logits,
        # False (illegal) becomes -inf. The Categorical's softmax then
        # routes zero probability to masked indices. Out-of-place form
        # (no trailing underscore) so the shared ``logits`` tensor
        # exposed on ``DiscretePolicyOutput`` is not mutated; ``masked_
        # _fill`` accepts a Python float scalar so no per-call ``-inf``
        # tensor allocation is needed (Phase 4 S07, 2026-05-03).
        return logits.masked_fill(~mask, float("-inf"))

    def set_effective_gate_threshold(self, value: float) -> None:
        """Phase-14 S06: trainer poke for warm-up annealing.

        The trainer computes
        ``effective = floor + frac × (gene_value - floor)`` per
        update (where ``floor = DIRECTION_GATE_THRESHOLD_MIN``
        and ``frac`` linearly ramps from 0 → 1 across the warm-up
        window) and writes the result here. The policy reads it
        in ``_apply_direction_gate``; absent this poke the gene
        value is used directly.
        """
        self._effective_gate_threshold = float(value)

    def _apply_direction_gate(
        self,
        logits: torch.Tensor,
        *,
        direction_back_prob: torch.Tensor,
        direction_lay_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Mask OPEN_BACK_i / OPEN_LAY_i logits where the per-runner
        direction confidence falls below ``direction_gate_threshold``.

        Phase-14 S03. The action layout is
        ``[NOOP, OPEN_BACK_0..R-1, OPEN_LAY_0..R-1, CLOSE_0..R-1]``
        so we slice the OPEN slots out of the flat logits tensor and
        write `-inf` where ``max(P_back_i, P_lay_i) < threshold``.

        NOOP (index 0) and CLOSE_i (last R indices) are NEVER masked
        — see ``hard_constraints.md §14, §15``. An agent at a strict
        threshold with no high-confidence runners simply emits NOOP;
        it can always still close existing positions.

        ``direction_back_prob`` / ``direction_lay_prob`` are
        ``(batch, R)`` sigmoid outputs from
        :attr:`direction_prob_head`. The mask uses
        ``max(P_back, P_lay)`` per runner (NOT per-side) so the agent
        is free to pick whichever side it prefers — the gate filters
        the OPPORTUNITY, not the side.

        Phase-14 S06: when the trainer has poked a warmup-annealed
        value via :meth:`set_effective_gate_threshold`, that value
        is used in place of the gene value. Absent the poke, the
        gene value is used directly (S05 / pre-S06 behaviour).
        """
        threshold = (
            self._effective_gate_threshold
            if self._effective_gate_threshold is not None
            else self.direction_gate_threshold
        )
        # Per-runner max confidence — gate-pass is a single bool per
        # (batch, slot).
        direction_max = torch.maximum(
            direction_back_prob, direction_lay_prob,
        )  # (batch, R)
        gate_pass = direction_max >= threshold  # (batch, R) bool
        R = self.max_runners
        # Locate OPEN slots in the flat layout.
        open_back_start = 1
        open_lay_start = 1 + R
        # Build a per-action gate-pass mask over the full logits.
        # Default True — only OPEN slots get the gate; other slots
        # (NOOP, CLOSE) stay legal regardless of threshold.
        batch = logits.shape[0]
        gate_mask = torch.ones(
            batch, logits.shape[-1], dtype=torch.bool,
            device=logits.device,
        )
        gate_mask[:, open_back_start: open_back_start + R] = gate_pass
        gate_mask[:, open_lay_start: open_lay_start + R] = gate_pass
        # AND with the existing legality mask (which is already
        # baked into ``logits`` via -inf at illegal positions). The
        # gate writes -inf at gate-blocked positions; positions that
        # were already -inf stay -inf.
        return logits.masked_fill(~gate_mask, float("-inf"))


def make_stake_distribution(
    stake_alpha: torch.Tensor,
    stake_beta: torch.Tensor,
) -> Beta:
    """Build a :class:`Beta` from the policy's stake parameters.

    Helper for the smoke driver and tests so the ``alpha, beta``
    contract is exercised in the same place every time.
    """
    return Beta(stake_alpha, stake_beta)
