"""Single policy factory — genome → policy (pbt-breeding Step 1b, HC#11).

ONE source of truth for building a v2 discrete-action policy from a
``CohortGenes``-shaped genome, used by BOTH the cohort worker
(``training_v2/cohort/worker.py``) AND the held-out re-eval tool
(``tools/reevaluate_cohort.py``). The re-eval previously rebuilt the
policy DIFFERENTLY from training (it forgot ``input_norm`` and silently
wrote zero rows); with two architectures now selectable per lineage, a
single constructor is mandatory, not optional — a transformer checkpoint
loaded into an LSTM (or vice versa) fails ``strict`` load and the agent
is silently dropped from the verdict.

The factory reads the STRUCTURAL genes off the genome by attribute
(duck-typed — no ``CohortGenes`` import, so ``agents_v2`` stays
independent of ``training_v2``):

* ``genes.architecture`` ∈ {``"lstm"``, ``"transformer"``}
* ``genes.hidden_size`` — LSTM hidden size / transformer ``d_model``
* ``genes.transformer_depth`` / ``transformer_heads`` /
  ``transformer_ctx_ticks`` — transformer-only sizing

Structural genes are frozen within a lineage (pbt HC#10), so the genome
the factory sees for an inheriting agent always matches the shapes of the
weights it will warm-start — the factory does not enforce that itself
(the breed step does), but the strict ``load_state_dict`` downstream is
the backstop.
"""

from __future__ import annotations

from agents_v2.discrete_policy import (
    BaseDiscretePolicy,
    DiscreteLSTMPolicy,
    DiscreteTransformerPolicy,
)


__all__ = ["build_policy", "policy_arch_name"]


def build_policy(
    genes,
    *,
    obs_dim: int,
    action_space,
    runner_dim: int | None = None,
    input_norm: bool = False,
    direction_gate_enabled: bool = False,
    direction_gate_threshold: float = 0.5,
    mature_prob_open_threshold: float = 0.0,
    enable_fc_prob_head: bool = False,
    frozen_direction_head_path=None,
) -> BaseDiscretePolicy:
    """Build the policy a genome describes.

    ``genes`` carries the STRUCTURAL choice (``architecture`` + sizing);
    the keyword args carry the RUNTIME context (obs/action shapes,
    ``runner_dim`` from ``env.active_runner_dim``, ``input_norm``, the
    open-gates, a frozen direction-head manifest). Both the worker and
    the re-eval tool pass the same runtime context so the trained and
    re-evaluated policies are bit-for-bit the same module.

    With ``genes.architecture == "lstm"`` (the default for every existing
    cohort and the gene-only GA) the call is byte-identical to the prior
    inline ``DiscreteLSTMPolicy(...)`` construction in the worker.
    """
    arch = str(getattr(genes, "architecture", "lstm"))
    common = dict(
        obs_dim=int(obs_dim),
        action_space=action_space,
        hidden_size=int(genes.hidden_size),
        direction_gate_enabled=bool(direction_gate_enabled),
        direction_gate_threshold=float(direction_gate_threshold),
        mature_prob_open_threshold=float(mature_prob_open_threshold),
        enable_fc_prob_head=bool(enable_fc_prob_head),
        runner_dim=runner_dim,
        frozen_direction_head_path=frozen_direction_head_path,
        input_norm=bool(input_norm),
    )
    if arch == "lstm":
        return DiscreteLSTMPolicy(**common)
    if arch == "transformer":
        return DiscreteTransformerPolicy(
            depth=int(getattr(genes, "transformer_depth", 2)),
            n_heads=int(getattr(genes, "transformer_heads", 4)),
            ctx_ticks=int(getattr(genes, "transformer_ctx_ticks", 32)),
            ffn_mult=int(getattr(genes, "transformer_ffn_mult", 2)),
            pos_encoding=str(getattr(genes, "transformer_pos_encoding", "learned")),
            **common,
        )
    raise ValueError(
        f"build_policy: unknown architecture {arch!r} "
        f"(expected 'lstm' or 'transformer')",
    )


def policy_arch_name(genes) -> str:
    """Registry ``architecture_name`` discriminator for a genome.

    Encodes the structural identity so the registry's weight-shape hash
    + the UI/scoreboard never confuse two architectures (and weights
    never cross-load). LSTM names are unchanged from the pre-pbt
    ``arch_name_for_genes`` (byte-identical discriminator for existing
    cohorts); transformers carry their depth/heads/ctx in the name.
    """
    arch = str(getattr(genes, "architecture", "lstm"))
    if arch == "transformer":
        name = (
            f"v2_discrete_ppo_transformer_d{int(genes.hidden_size)}"
            f"_L{int(genes.transformer_depth)}"
            f"_h{int(genes.transformer_heads)}"
            f"_ctx{int(genes.transformer_ctx_ticks)}"
        )
        # Suffix ONLY for non-default values so existing ffn=2 / learned
        # transformer champions keep their exact pre-gene hash (warm-load
        # intact); ffn=4 and rope DO change the weight shapes / module set,
        # so they MUST carry a distinct hash (the registry never cross-loads
        # incompatible shapes). The dataclass defaults are ffn_mult=2,
        # pos_encoding="learned".
        ffn_mult = int(getattr(genes, "transformer_ffn_mult", 2))
        if ffn_mult != 2:
            name += f"_ffn{ffn_mult}"
        pos_encoding = str(getattr(genes, "transformer_pos_encoding", "learned"))
        if pos_encoding != "learned":
            name += f"_pos{pos_encoding}"
        return name
    return f"v2_discrete_ppo_lstm_h{int(genes.hidden_size)}"
