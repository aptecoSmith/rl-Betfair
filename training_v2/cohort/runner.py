"""GA cohort orchestrator — Phase 3, Session 03 deliverable.

Drives ``n_agents`` workers across ``n_generations`` of training,
evaluating each agent on a held-out day, breeding the next generation
from the elite half of the previous, and writing a registry-shaped
scoreboard the v1 UI consumes unchanged (during the comparison window).

Concurrency: **sequential** for Session 03 (session prompt §3
"Concurrency for Session 03 = sequential"). v1's
``ThreadPoolExecutor`` worker pool is fragile at high N (OOM on a
shared GPU) and is the wrong hill for the first-run scaffolding. A
follow-on plan adds the worker pool if Session 04's wall time
demands it.

CLI:

    python -m training_v2.cohort.runner \\
        --n-agents 4 --generations 2 --days 7 \\
        --device cuda --seed 42 \\
        --output-dir registry/v2_dryrun_$(date +%s)

Output layout (mirrors v1):

    registry/{output-dir}/models.db
    registry/{output-dir}/weights/{model_id}.pt
    registry/{output-dir}/scoreboard.jsonl       (one row per agent)

The JSONL scoreboard duplicates the SQLite-stored data in a flat
shape the UI / findings.md aggregator can read without an SQL query;
v1 emits the same.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from registry.model_store import ModelStore
from training_v2.cohort.events import (
    WebSocketBroadcastServer,
    cohort_complete_event,
    cohort_started_event,
)
from training_v2.cohort.genes import (
    CohortGenes,
    assert_in_range,
    crossover,
    mutate,
    sample_genes,
)
from training_v2.cohort.batched_worker import train_cluster_batched
from training_v2.cohort.worker import AgentResult, train_one_agent
from training_v2.discrete_ppo.batched_rollout import cluster_agents_by_arch
from training_v2.discrete_ppo.train import select_days


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"


logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────


def run_cohort(
    *,
    n_agents: int,
    n_generations: int,
    days: int,
    data_dir: Path,
    device: str,
    seed: int,
    output_dir: Path,
    mutation_rate: float = 0.1,
    train_one_agent_fn: Callable[..., AgentResult] = train_one_agent,
    event_emitter: Callable[[dict], None] | None = None,
    reward_overrides: dict | None = None,
    batched: bool = False,
) -> list[AgentResult]:
    """Run the cohort end-to-end. Returns one :class:`AgentResult` per agent.

    The list contains the LAST generation's agents in eval-reward
    descending order. Earlier-generation agents are persisted to the
    registry (one model row, one scoreboard row each) but not returned
    in the list — they're recoverable by querying the registry.

    Parameters
    ----------
    n_agents:
        Cohort size per generation. Phase 3 dry-run uses 4; Session 04
        scales to 12.
    n_generations:
        Number of generations to train. ``1`` = no breeding (initial
        cohort only). ``2`` = one breeding pass between generations.
    days:
        Number of recent training days to use. Last one is held out
        as the eval day; the remaining ``days-1`` are training days.
    train_one_agent_fn:
        Injection point for the lightweight integration test —
        defaults to the real :func:`train_one_agent`.
    event_emitter:
        Optional ``Callable[[dict], None]``. When supplied, called with
        v1-shape websocket events at run-start, per-agent points (via
        :func:`train_one_agent`), and run-complete (Session 04
        deliverable). Pass ``None`` for silent runs (unit tests,
        scripted use). The CLI's ``--emit-websocket`` flag wires
        :class:`training_v2.cohort.events.WebSocketBroadcastServer`
        in here.
    """
    if n_agents < 2:
        raise ValueError(f"n_agents must be >= 2 for breeding, got {n_agents}")
    if n_generations < 1:
        raise ValueError(f"n_generations must be >= 1, got {n_generations}")
    if days < 2:
        raise ValueError(
            f"days must be >= 2 (training + eval), got {days}",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    bet_logs_dir = output_dir / "bet_logs"
    db_path = output_dir / "models.db"
    scoreboard_path = output_dir / "scoreboard.jsonl"

    model_store = ModelStore(
        db_path=db_path,
        weights_dir=weights_dir,
        bet_logs_dir=bet_logs_dir,
    )

    # ── Day selection ────────────────────────────────────────────────
    training_days, eval_day = select_days(
        data_dir=data_dir, n_days=int(days), day_shuffle_seed=int(seed),
    )
    logger.info(
        "Cohort: %d agents × %d generations on %d training days "
        "(eval=%s); device=%s output_dir=%s",
        n_agents, n_generations, len(training_days), eval_day, device, output_dir,
    )

    # ── Initial population (gen 0) ───────────────────────────────────
    rng = random.Random(int(seed))
    cohort: list[CohortGenes] = [sample_genes(rng) for _ in range(n_agents)]
    parent_ids: list[tuple[str | None, str | None]] = [
        (None, None) for _ in range(n_agents)
    ]

    last_results: list[AgentResult] = []
    cohort_t0 = time.perf_counter()
    run_id = str(uuid.uuid4())
    total_agents_trained = 0

    # ── Run-start event ──────────────────────────────────────────────
    if event_emitter is not None:
        try:
            event_emitter(cohort_started_event(
                run_id=run_id,
                n_generations=int(n_generations),
                n_agents=int(n_agents),
                train_days=list(training_days),
                eval_day=str(eval_day),
                seed=int(seed),
            ))
        except Exception:
            logger.exception("event_emitter raised on cohort_started; continuing")

    with scoreboard_path.open("w", encoding="utf-8") as sf:
        for generation in range(n_generations):
            gen_t0 = time.perf_counter()
            logger.info(
                "── Generation %d/%d ──", generation + 1, n_generations,
            )
            agent_ids_gen = [str(uuid.uuid4()) for _ in cohort]
            per_agent_seeds = [
                (int(seed) * 1_000_003 + generation * 10_000 + i) & 0x7FFFFFFF
                for i in range(len(cohort))
            ]
            for idx, genes in enumerate(cohort):
                assert_in_range(genes)
                logger.info(
                    "Generation %d agent %d/%d (id=%s) genes=%s",
                    generation + 1, idx + 1, n_agents,
                    agent_ids_gen[idx][:12], genes.to_dict(),
                )

            results: list[AgentResult] = [None] * len(cohort)  # type: ignore[list-item]

            if batched:
                # Cluster by architecture, run each cluster batched.
                # Cross-cluster scheduling is sequential (one cluster
                # consumes the GPU at a time — Session 02 prompt §2
                # "Cross-cluster scheduling. Sequential.").
                #
                # We dry-instantiate policies temporarily just to get
                # the cluster key from each agent's hidden_size; full
                # policy construction (under per-agent seed) happens
                # inside ``train_cluster_batched``.
                cluster_to_indices: dict[tuple, list[int]] = {}
                for i, g in enumerate(cohort):
                    key = (
                        "DiscreteLSTMPolicy",
                        int(g.hidden_size),
                    )
                    cluster_to_indices.setdefault(key, []).append(i)
                for cluster_key, idxs in cluster_to_indices.items():
                    logger.info(
                        "── Cluster %s: %d agents (batched) ──",
                        cluster_key, len(idxs),
                    )
                    cluster_results = train_cluster_batched(
                        agent_ids=[agent_ids_gen[i] for i in idxs],
                        genes_list=[cohort[i] for i in idxs],
                        days_to_train=list(training_days),
                        eval_day=eval_day,
                        data_dir=data_dir,
                        device=device,
                        seeds=[per_agent_seeds[i] for i in idxs],
                        model_store=model_store,
                        generation=generation,
                        parent_ids=[parent_ids[i] for i in idxs],
                        event_emitter=event_emitter,
                        agent_indices_in_cohort=[int(i) for i in idxs],
                        n_agents_in_cohort=int(n_agents),
                        reward_overrides=reward_overrides,
                    )
                    for k, i in enumerate(idxs):
                        results[i] = cluster_results[k]
                        total_agents_trained += 1
            else:
                for idx, genes in enumerate(cohort):
                    pa_id, pb_id = parent_ids[idx]
                    result = train_one_agent_fn(
                        agent_id=agent_ids_gen[idx],
                        genes=genes,
                        days_to_train=list(training_days),
                        eval_day=eval_day,
                        data_dir=data_dir,
                        device=device,
                        seed=per_agent_seeds[idx],
                        model_store=model_store,
                        generation=generation,
                        parent_a_id=pa_id,
                        parent_b_id=pb_id,
                        event_emitter=event_emitter,
                        agent_idx=int(idx),
                        n_agents=int(n_agents),
                        reward_overrides=reward_overrides,
                    )
                    results[idx] = result
                    total_agents_trained += 1

            # ── Scoreboard write (after all agents in this gen done) ─
            for idx, result in enumerate(results):
                row = _agent_result_to_scoreboard_row(
                    result=result,
                    generation=generation,
                    agent_idx=idx,
                    eval_day=eval_day,
                    training_days=list(training_days),
                )
                sf.write(json.dumps(row) + "\n")
                sf.flush()

            # Sort by eval-day total_reward (descending). Ties are
            # broken by day_pnl (descending) so a higher cash-P&L
            # agent ranks above a higher-shaped-reward agent at the
            # same total — useful when the eval rollouts produce
            # similar shaped contributions.
            results.sort(
                key=lambda r: (
                    -float(r.eval.total_reward),
                    -float(r.eval.day_pnl),
                ),
            )
            gen_wall = time.perf_counter() - gen_t0
            logger.info(
                "Generation %d complete in %.1fs. Top-3 by eval reward:",
                generation + 1, gen_wall,
            )
            for rank, r in enumerate(results[:3]):
                logger.info(
                    "  #%d agent=%s reward=%+.3f pnl=%+.2f bets=%d genes=%s",
                    rank + 1, r.agent_id[:12], r.eval.total_reward,
                    r.eval.day_pnl, r.eval.bet_count, r.genes.to_dict(),
                )

            last_results = results
            # ── Breed next generation if any left ─────────────────
            if generation < n_generations - 1:
                cohort, parent_ids = _breed_next_generation(
                    parents_ranked=results,
                    rng=rng,
                    n_agents=n_agents,
                    mutation_rate=mutation_rate,
                    model_store=model_store,
                    next_generation=generation + 1,
                )

    cohort_wall = time.perf_counter() - cohort_t0
    logger.info(
        "Cohort complete in %.1fs. Wrote %s + %s",
        cohort_wall, db_path, scoreboard_path,
    )

    # ── Run-complete event ──────────────────────────────────────────
    if event_emitter is not None:
        try:
            top_5 = [
                {
                    "model_id": r.model_id,
                    "composite_score": float(r.eval.total_reward),
                    "pnl": float(r.eval.day_pnl),
                    "win_rate": float(r.eval.bet_precision),
                    "architecture": r.architecture_name,
                }
                for r in last_results[:5]
            ]
            best_model = None
            if last_results:
                br = last_results[0]
                best_model = {
                    "model_id": br.model_id,
                    "composite_score": float(br.eval.total_reward),
                    "total_pnl": float(br.eval.day_pnl),
                    "win_rate": float(br.eval.bet_precision),
                    "architecture": br.architecture_name,
                }
            event_emitter(cohort_complete_event(
                run_id=run_id,
                status="completed",
                n_generations=int(n_generations),
                total_agents_trained=int(total_agents_trained),
                total_agents_evaluated=int(total_agents_trained),
                wall_time_seconds=float(cohort_wall),
                best_model=best_model,
                top_5=top_5,
            ))
        except Exception:
            logger.exception("event_emitter raised on cohort_complete; continuing")

    return last_results


# ── Breeding ─────────────────────────────────────────────────────────────


def _breed_next_generation(
    *,
    parents_ranked: list[AgentResult],
    rng: random.Random,
    n_agents: int,
    mutation_rate: float,
    model_store: ModelStore | None,
    next_generation: int,
) -> tuple[list[CohortGenes], list[tuple[str | None, str | None]]]:
    """Top-half elites carry over verbatim; bottom-half bred + mutated.

    Returns ``(next_cohort_genes, parent_ids)`` where ``parent_ids[i]``
    is ``(parent_a_id, parent_b_id)`` for child ``i``. Elites have
    ``(None, None)`` because the registry already has their parent
    chain on the previous generation's row.
    """
    n_elites = max(1, n_agents // 2)
    elites = parents_ranked[:n_elites]
    elite_genes = [e.genes for e in elites]

    next_cohort: list[CohortGenes] = list(elite_genes)
    next_parent_ids: list[tuple[str | None, str | None]] = [
        (None, None) for _ in elite_genes
    ]

    n_children = n_agents - len(next_cohort)
    for _ in range(n_children):
        if len(elites) >= 2:
            a, b = rng.sample(elites, 2)
        else:
            a = b = elites[0]
        child = crossover(a.genes, b.genes, rng)
        child = mutate(child, rng, mutation_rate=mutation_rate)
        assert_in_range(child)
        next_cohort.append(child)
        next_parent_ids.append((a.model_id, b.model_id))

        if model_store is not None:
            model_store.record_genetic_event(_make_genetic_event(
                generation=next_generation,
                child_id=None,  # child model_id created in worker; tied via parents
                parent_a_id=a.model_id,
                parent_b_id=b.model_id,
                child_genes=child,
            ))

    return next_cohort, next_parent_ids


def _make_genetic_event(
    *,
    generation: int,
    child_id: str | None,
    parent_a_id: str,
    parent_b_id: str,
    child_genes: CohortGenes,
):
    """Build a v1-shape ``GeneticEventRecord`` for a crossover+mutate event."""
    from registry.model_store import GeneticEventRecord
    return GeneticEventRecord(
        event_id=str(uuid.uuid4()),
        generation=int(generation),
        event_type="crossover",
        child_model_id=child_id,
        parent_a_id=parent_a_id,
        parent_b_id=parent_b_id,
        hyperparameter=None,
        parent_a_value=None,
        parent_b_value=None,
        inherited_from=None,
        mutation_delta=None,
        final_value=json.dumps(child_genes.to_dict()),
        selection_reason=None,
        human_summary=(
            f"Bred from parents {parent_a_id[:12]} × {parent_b_id[:12]}; "
            f"child genes={child_genes.to_dict()}"
        ),
    )


# ── Scoreboard row builder ────────────────────────────────────────────────


def _agent_result_to_scoreboard_row(
    *,
    result: AgentResult,
    generation: int,
    agent_idx: int,
    eval_day: str,
    training_days: list[str],
) -> dict:
    """Flatten an :class:`AgentResult` into a v1-shape scoreboard row.

    The shape mirrors v1's ``scoreboard.jsonl`` rows: flat primitives,
    one row per agent, gene dict embedded under ``hyperparameters``.
    The UI reads both v1 and v2 rows during the comparison window.
    """
    return {
        "schema": "v2_cohort_scoreboard",
        "model_id": result.model_id,
        "agent_id": result.agent_id,
        "architecture_name": result.architecture_name,
        "generation": int(generation),
        "agent_idx": int(agent_idx),
        "hyperparameters": result.genes.to_dict(),
        "weights_path": result.weights_path,
        "run_id": result.run_id,
        "training_days": list(training_days),
        "eval_day": eval_day,
        # Train aggregates
        "train_n_days": result.train.n_days,
        "train_total_steps": result.train.total_steps,
        "train_total_reward": result.train.total_reward,
        "train_mean_reward": result.train.mean_reward,
        "train_mean_pnl": result.train.mean_pnl,
        "train_mean_value_loss": result.train.mean_value_loss,
        "train_mean_policy_loss": result.train.mean_policy_loss,
        "train_mean_approx_kl": result.train.mean_approx_kl,
        "train_wall_time_sec": result.train.wall_time_sec,
        "train_per_day": result.train.per_day_rows,
        # Eval (held-out day)
        "eval_total_reward": result.eval.total_reward,
        "eval_day_pnl": result.eval.day_pnl,
        "eval_n_steps": result.eval.n_steps,
        "eval_bet_count": result.eval.bet_count,
        "eval_winning_bets": result.eval.winning_bets,
        "eval_bet_precision": result.eval.bet_precision,
        "eval_pnl_per_bet": result.eval.pnl_per_bet,
        "eval_early_picks": result.eval.early_picks,
        "eval_profitable": result.eval.profitable,
        "eval_action_histogram": result.eval.action_histogram,
        "eval_arbs_completed": result.eval.arbs_completed,
        "eval_arbs_naked": result.eval.arbs_naked,
        "eval_arbs_closed": result.eval.arbs_closed,
        "eval_arbs_force_closed": result.eval.arbs_force_closed,
        "eval_arbs_stop_closed": result.eval.arbs_stop_closed,
        "eval_arbs_target_pnl_refused": result.eval.arbs_target_pnl_refused,
        "eval_pairs_opened": result.eval.pairs_opened,
        "eval_locked_pnl": result.eval.locked_pnl,
        "eval_naked_pnl": result.eval.naked_pnl,
        "eval_closed_pnl": result.eval.closed_pnl,
        "eval_force_closed_pnl": result.eval.force_closed_pnl,
        "eval_stop_closed_pnl": result.eval.stop_closed_pnl,
        "eval_wall_time_sec": result.eval.wall_time_sec,
        # Composite — same as v1: a single scalar the UI sorts by.
        "composite_score": result.eval.total_reward,
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def _parse_reward_overrides(items: list[str]) -> dict:
    """Parse a list of ``key=value`` strings into a dict.

    Values are parsed as bool (``true``/``false``/``1``/``0``), then
    float, then fall back to string. The env's
    ``_REWARD_OVERRIDE_KEYS`` whitelist is the authoritative typecheck
    — passing an unknown key produces a one-time debug log inside
    ``BetfairEnv.__init__`` and is otherwise ignored.
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"--reward-overrides expects key=value, got {item!r}"
            )
        key, _, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        lo = raw.lower()
        if lo in ("true", "1"):
            out[key] = True
        elif lo in ("false", "0"):
            out[key] = False
        else:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "v2 GA cohort runner (Phase 3 Session 03). Trains N agents "
            "across the locked Phase 3 gene schema, breeds elites into "
            "the next generation, and writes a v1-shape scoreboard."
        ),
    )
    p.add_argument(
        "--n-agents", type=int, default=4,
        help="Cohort size per generation. Default 4 (Phase 3 dry-run).",
    )
    p.add_argument(
        "--generations", type=int, default=2,
        help="Number of generations to train. Default 2.",
    )
    p.add_argument(
        "--days", type=int, default=7,
        help=(
            "Number of recent days to use. Last is held out as eval. "
            "Default 7."
        ),
    )
    p.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="Directory containing YYYY-MM-DD.parquet day files.",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device (cpu, cuda, cuda:N). Default cpu.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Cohort seed (drives gene sampling + day shuffle).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help=(
            "Directory for the cohort's models.db, weights/, "
            "bet_logs/, and scoreboard.jsonl outputs."
        ),
    )
    p.add_argument(
        "--mutation-rate", type=float, default=0.1,
        help="Per-gene mutation probability for breeding. Default 0.1.",
    )
    p.add_argument(
        "--emit-websocket", action="store_true",
        help=(
            "Start a websocket broadcast server on localhost:8002 and "
            "emit v1-shape cohort events to all connected clients. "
            "Mutually exclusive with a running v1 ``training.worker`` "
            "(port collision). The api / frontend connection chain "
            "works unchanged because the api connects as a CLIENT to "
            "ws://localhost:8002 (api/main.py::_worker_connection)."
        ),
    )
    p.add_argument(
        "--ws-host", default="localhost",
        help="Bind host for --emit-websocket. Default localhost.",
    )
    p.add_argument(
        "--reward-overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help=(
            "Plan-level reward override (key=value). Repeatable. "
            "Values parse as bool ('true'/'false'/'1'/'0'), float, "
            "or string in that order. Whitelisted keys live in "
            "BetfairEnv._REWARD_OVERRIDE_KEYS. Example: "
            "--reward-overrides target_pnl_pair_sizing_enabled=true"
        ),
    )
    p.add_argument(
        "--ws-port", type=int, default=8002,
        help=(
            "Bind port for --emit-websocket. Default 8002 (matches the v1 "
            "training_worker default in config.yaml so no api change is "
            "needed)."
        ),
    )
    p.add_argument(
        "--batched", action="store_true",
        help=(
            "Use the batched cohort path (throughput-fix Session 02). "
            "Clusters agents by architecture (hidden_size) and shares "
            "one BatchedRolloutCollector per cluster per training day. "
            "Default OFF; the sequential per-agent path stays the "
            "default until at least one cohort run validates the "
            "batched path."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    server: WebSocketBroadcastServer | None = None
    emitter: Callable[[dict], None] | None = None
    if args.emit_websocket:
        server = WebSocketBroadcastServer(host=args.ws_host, port=args.ws_port)
        server.start()
        emitter = server

    reward_overrides = _parse_reward_overrides(args.reward_overrides)
    if reward_overrides:
        logger.info("reward_overrides: %s", reward_overrides)

    try:
        run_cohort(
            n_agents=args.n_agents,
            n_generations=args.generations,
            days=args.days,
            data_dir=Path(args.data_dir),
            device=args.device,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            mutation_rate=args.mutation_rate,
            event_emitter=emitter,
            reward_overrides=reward_overrides or None,
            batched=bool(args.batched),
        )
    finally:
        if server is not None:
            # Give clients a beat to receive the final cohort_complete
            # event before closing the listen socket.
            time.sleep(0.5)
            server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
