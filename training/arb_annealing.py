"""Pure annealing helpers for arb-curriculum Session 03.

Generation-level interpolation of naked_loss_scale toward 1.0 over a
configured window. These are intentionally pure functions so they can be
tested in isolation and applied at any call site without side effects.

Vocabulary
----------
gene_value:
    The per-agent ``naked_loss_scale`` gene sampled by the GA. Represents
    the agent's preference for how much to discount naked losses.
effective_scale:
    The actual value passed to the env this generation. During the anneal
    window ``effective_scale > gene_value`` (losses less discounted than
    the gene alone would produce); once ``end_gen`` is reached
    ``effective_scale == gene_value == 1.0``.

Usage in the orchestrator::

    from training.arb_annealing import effective_naked_loss_scale
    schedule = training_plan.naked_loss_anneal  # dict or None
    for agent in agents:
        hp = dict(agent.hyperparameters)
        hp["naked_loss_scale"] = effective_naked_loss_scale(
            hp.get("naked_loss_scale", 1.0),
            current_gen=generation,
            schedule=schedule,
        )
        trainer = PPOTrainer(..., hyperparams=hp, ...)
"""

from __future__ import annotations


def anneal_factor(current_gen: int, start: int, end: int) -> float:
    """Interpolation progress in [0, 1].

    Returns 0.0 before ``start``, 1.0 from ``end`` onward, and a linear
    ramp between. Degenerate case ``end <= start`` always returns 1.0.
    """
    if end <= start:
        return 1.0
    if current_gen <= start:
        return 0.0
    if current_gen >= end:
        return 1.0
    return (current_gen - start) / (end - start)


def effective_naked_loss_scale(
    gene_value: float,
    current_gen: int,
    schedule: dict | None,
) -> float:
    """Return the generation-adjusted naked_loss_scale for one agent.

    Parameters
    ----------
    gene_value:
        The agent's raw ``naked_loss_scale`` gene in [0, 1].
    current_gen:
        The zero-indexed generation currently being trained.
    schedule:
        ``None`` → no annealing (return ``gene_value`` unchanged).
        Otherwise a dict with ``"start_gen"`` and ``"end_gen"`` keys.
        The scale is linearly interpolated from ``gene_value`` toward
        ``1.0`` over ``[start_gen, end_gen)``.

    Returns
    -------
    float
        The effective scale to write into the agent's HP dict before
        constructing the env. Always in [gene_value, 1.0].
    """
    if schedule is None:
        return gene_value
    p = anneal_factor(
        current_gen,
        int(schedule["start_gen"]),
        int(schedule["end_gen"]),
    )
    return gene_value + (1.0 - gene_value) * p
