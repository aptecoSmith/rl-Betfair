"""Pydantic models for all API request/response types."""

from __future__ import annotations

from pydantic import BaseModel


# ── Scoreboard & Model Records ──────────────────────────────────────


class DayMetric(BaseModel):
    date: str
    day_pnl: float
    bet_count: int
    winning_bets: int
    bet_precision: float
    pnl_per_bet: float
    early_picks: int
    profitable: bool
    starting_budget: float = 100.0
    # Forced-arbitrage (scalping) metrics — Issue 05. Zero for
    # directional models; non-zero rows flag a scalping model.
    arbs_completed: int = 0
    arbs_naked: int = 0
    locked_pnl: float = 0.0
    naked_pnl: float = 0.0


class ScoreboardEntry(BaseModel):
    model_id: str
    generation: int
    architecture_name: str
    status: str
    composite_score: float | None
    win_rate: float
    sharpe: float
    mean_daily_pnl: float
    bet_precision: float
    efficiency: float
    test_days: int
    profitable_days: int
    early_picks: int = 0
    mean_daily_return_pct: float | None = None
    recorded_budget: float | None = None
    market_type_filter: str | None = None
    garaged: bool = False
    garaged_at: str | None = None
    created_at: str | None = None
    last_evaluated_at: str | None = None
    # Forced-arbitrage (scalping) fields — Issue 05. ``is_scalping``
    # is set from the model's hyperparameters (``scalping_mode`` gene)
    # and gates which scoreboard tab the row appears on. The numeric
    # fields are summed across the latest evaluation run's days; zero
    # for directional models.
    is_scalping: bool = False
    total_bets: int = 0
    arbs_completed: int = 0
    arbs_naked: int = 0
    locked_pnl: float = 0.0
    naked_pnl: float = 0.0
    # Scalping-active-management §06 — mean absolute calibration error
    # on the latest eval run. Diagnostic only; does NOT feed the
    # composite score (hard_constraints §14). ``None`` for directional
    # runs, pre-Session-02 runs, and scalping runs without enough
    # eval-day data (< 2 buckets with >= 20 pairs).
    mean_absolute_calibration_error: float | None = None


class ScoreboardResponse(BaseModel):
    models: list[ScoreboardEntry]


class GarageToggleRequest(BaseModel):
    garaged: bool


class GarageToggleResponse(BaseModel):
    model_id: str
    garaged: bool


class ReliabilityBucket(BaseModel):
    """One bucket of the fill-probability reliability diagram.

    Scalping-active-management §05. Fill-probability predictions at
    placement time are binned into four fixed buckets and compared
    against the observed completion rate of the pairs that landed in
    each bucket. A perfectly-calibrated head's ``observed_rate`` lines
    up with ``predicted_midpoint``.
    """

    bucket_label: str  # "<25%", "25-50%", "50-75%", ">75%"
    predicted_midpoint: float  # 0.125 / 0.375 / 0.625 / 0.875
    observed_rate: float  # count_completed / (completed + naked) in bucket
    count: int  # total pairs in the bucket (completed + naked)
    abs_calibration_error: float  # |predicted_midpoint - observed_rate|


class RiskScatterPoint(BaseModel):
    """One point on the risk-vs-realised scatter.

    Scalping-active-management §05. One record per completed pair. The
    ``stddev_bucket`` is derived from the 25th / 75th percentiles of
    this evaluation run's predicted stddev values, so it is
    self-scaling per-run (single-value runs collapse every point to
    ``"med"``).
    """

    predicted_pnl: float
    realised_pnl: float
    stddev_bucket: str  # "low" | "med" | "high"


class CalibrationStats(BaseModel):
    """Calibration diagnostics for the scalping aux heads on the
    latest evaluation run.

    ``None`` at the ``ModelDetail.calibration`` level means the run has
    zero scalping pairs (directional model or pre-Session-02 run). When
    non-null but ``insufficient_data`` is true, the head hasn't seen
    enough eval-day pairs for meaningful bucket statistics — fewer than
    two buckets cleared the ``count >= 20`` threshold.
    """

    reliability_buckets: list[ReliabilityBucket]
    mace: float | None  # None → insufficient data
    scatter: list[RiskScatterPoint]
    insufficient_data: bool


class ModelDetail(BaseModel):
    model_id: str
    generation: int
    parent_a_id: str | None
    parent_b_id: str | None
    architecture_name: str
    architecture_description: str
    hyperparameters: dict
    status: str
    created_at: str
    last_evaluated_at: str | None
    composite_score: float | None
    garaged: bool = False
    metrics_history: list[DayMetric]
    # Scalping-active-management §05 — diagnostic-only calibration card.
    # ``None`` for directional / pre-Session-02 runs. Does NOT feed
    # composite_score ranking (hard_constraints §14).
    calibration: CalibrationStats | None = None


# ── Lineage ─────────────────────────────────────────────────────────


class LineageNode(BaseModel):
    model_id: str
    generation: int
    parent_a_id: str | None
    parent_b_id: str | None
    architecture_name: str
    hyperparameters: dict
    composite_score: float | None


class LineageResponse(BaseModel):
    nodes: list[LineageNode]


# ── Genetic Events ──────────────────────────────────────────────────


class GeneticEvent(BaseModel):
    event_id: str
    generation: int
    event_type: str
    child_model_id: str | None = None
    parent_a_id: str | None = None
    parent_b_id: str | None = None
    hyperparameter: str | None = None
    parent_a_value: str | None = None
    parent_b_value: str | None = None
    inherited_from: str | None = None
    mutation_delta: float | None = None
    final_value: str | None = None
    selection_reason: str | None = None
    human_summary: str | None = None


class GeneticsResponse(BaseModel):
    events: list[GeneticEvent]


# ── Training Status ─────────────────────────────────────────────────


class ProgressSnapshot(BaseModel):
    label: str
    completed: int
    total: int
    pct: float
    item_eta_human: str
    process_eta_human: str


class TrainingStatus(BaseModel):
    running: bool
    phase: str | None = None
    generation: int | None = None
    process: ProgressSnapshot | None = None
    item: ProgressSnapshot | None = None
    overall: ProgressSnapshot | None = None
    detail: str | None = None
    last_agent_score: float | None = None
    worker_connected: bool = False
    unevaluated_count: int | None = None
    eval_rate_s: float | None = None
    plan_id: str | None = None


# ── WebSocket Events ────────────────────────────────────────────────


class WSEvent(BaseModel):
    event: str
    timestamp: float
    phase: str | None = None
    process: dict | None = None
    item: dict | None = None
    detail: str | None = None
    summary: dict | None = None


# ── System Metrics ─────────────────────────────────────────────────


class GpuMetrics(BaseModel):
    name: str
    utilisation_pct: int
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: int | None = None



class SystemMetrics(BaseModel):
    cpu_pct: float
    ram_used_mb: int
    ram_total_mb: int
    ram_pct: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    disk_used_gb: float
    disk_total_gb: float
    gpu: GpuMetrics | None = None


# ── Replay ──────────────────────────────────────────────────────────


class BetEvent(BaseModel):
    tick_timestamp: str
    seconds_to_off: float
    runner_id: int
    runner_name: str
    action: str
    price: float
    stake: float
    matched_size: float
    outcome: str
    pnl: float


class RaceSummary(BaseModel):
    race_id: str
    market_name: str
    venue: str
    market_start_time: str
    n_runners: int
    bet_count: int
    race_pnl: float


class ReplayDayResponse(BaseModel):
    model_id: str
    date: str
    races: list[RaceSummary]


class TickRunner(BaseModel):
    selection_id: int
    status: str
    last_traded_price: float
    total_matched: float
    available_to_back: list[dict]
    available_to_lay: list[dict]


class ReplayTick(BaseModel):
    timestamp: str
    sequence_number: int
    in_play: bool
    traded_volume: float
    runners: list[TickRunner]
    bets: list[BetEvent]


class ReplayRaceResponse(BaseModel):
    model_id: str
    date: str
    race_id: str
    venue: str
    market_start_time: str
    winner_selection_id: int | None
    ticks: list[ReplayTick]
    all_bets: list[BetEvent]
    race_pnl: float
    runner_names: dict[str, str] = {}  # selection_id → horse name


class ExplorerBet(BaseModel):
    date: str
    race_id: str
    venue: str
    race_time: str
    tick_timestamp: str
    seconds_to_off: float
    runner_id: int
    runner_name: str
    action: str
    price: float
    stake: float
    matched_size: float
    outcome: str
    pnl: float
    # EW metadata
    is_each_way: bool = False
    each_way_divisor: float | None = None
    number_of_places: int | None = None
    settlement_type: str = "standard"
    effective_place_odds: float | None = None
    # Scalping: pair_id links the back/lay legs of a hedged pair.
    # None for non-scalping (naked) bets.
    pair_id: str | None = None
    # Scalping aux-head predictions at placement time (Sessions 02 + 03).
    # None for pre-Session-02 bets or bets that didn't produce a prediction
    # (directional mode, stub tests). Surfaced in the Bet Explorer as a
    # confidence chip + risk tag; hidden while values are still ≈ 0.5 before
    # the aux heads activate (activation_playbook.md Step E).
    fill_prob_at_placement: float | None = None
    predicted_locked_pnl_at_placement: float | None = None
    predicted_locked_stddev_at_placement: float | None = None


class BetExplorerResponse(BaseModel):
    model_id: str
    total_bets: int
    total_pnl: float
    bet_precision: float
    pnl_per_bet: float
    starting_budget: float | None = None
    bets: list[ExplorerBet]


# ── Admin ──────────────────────────────────────────────────────────


class ExtractedDay(BaseModel):
    date: str
    tick_count: int
    race_count: int
    file_size_bytes: int


class ExtractedDaysResponse(BaseModel):
    days: list[ExtractedDay]


class BackupDay(BaseModel):
    date: str


class BackupDaysResponse(BaseModel):
    days: list[BackupDay]


class MysqlDatesResponse(BaseModel):
    dates: list[str]
    available: bool


class AdminAgentEntry(BaseModel):
    model_id: str
    generation: int
    architecture_name: str
    status: str
    composite_score: float | None
    created_at: str
    garaged: bool = False


class AdminAgentsResponse(BaseModel):
    agents: list[AdminAgentEntry]


class ImportDayRequest(BaseModel):
    date: str


class ImportRangeRequest(BaseModel):
    start_date: str
    end_date: str
    force: bool = False


class ResetRequest(BaseModel):
    confirm: str
    clear_garage: bool = False


class AdminDeleteResponse(BaseModel):
    deleted: bool
    detail: str


class ImportDayResponse(BaseModel):
    success: bool
    date: str
    detail: str


class ImportRangeResponse(BaseModel):
    job_id: str
    dates_queued: int
    detail: str


class ResetResponse(BaseModel):
    reset: bool
    detail: str


# ── Streamrecorder Restore ─────────────────────────────────────────


class StreamrecorderBackup(BaseModel):
    date: str
    timestamp: str  # e.g. "223000"
    cold_file: str
    hot_file: str
    cold_size_bytes: int
    hot_size_bytes: int
    already_extracted: bool


class StreamrecorderBackupsResponse(BaseModel):
    backups: list[StreamrecorderBackup]
    backup_dir: str


class RestoreRequest(BaseModel):
    dates: list[str]


class RestoreResponse(BaseModel):
    job_id: str
    dates_queued: int
    detail: str


# ── Training Control ──────────────────────────────────────────────────


class StartTrainingRequest(BaseModel):
    plan_id: str | None = None  # optional training plan to launch
    n_generations: int = 3
    n_epochs: int = 3
    population_size: int | None = None  # override config default if set
    seed: int | None = None
    reevaluate_garaged: bool = False
    reevaluate_min_score: float | None = None
    train_dates: list[str] | None = None  # explicit YYYY-MM-DD dates; None = auto-split
    test_dates: list[str] | None = None
    # Per-run architecture selection — None = use config defaults
    architectures: list[str] | None = None
    # Per-run constraint overrides — None = use admin defaults from config
    max_back_price: float | None = None
    max_lay_price: float | None = None
    min_seconds_before_off: int | None = None
    # Per-run budget override — None = use config/plan default
    starting_budget: float | None = None
    # Per-run market type filter restriction — None = all choices
    market_type_filters: list[str] | None = None
    # Per-run cap on simultaneous gene mutations per child. None = use
    # config default (which may itself be null = legacy coin-flip).
    max_mutations_per_child: int | None = None
    # Per-run breeding-pool scope: run_only | include_garaged | full_registry.
    # None = use config default.
    breeding_pool: str | None = None
    # Per-run stud models (Issue 13). List of model IDs that are forced to
    # be parents in every generation regardless of selection. Max 5.
    # None or empty list = no studs (current behaviour).
    stud_model_ids: list[str] | None = None
    # Per-run mutation rate override (Issue 09). None = use config default.
    mutation_rate: float | None = None
    # Adaptive breeding (Issue 09). None on any field = use config default.
    bad_generation_threshold: float | None = None
    bad_generation_policy: str | None = None  # persist|boost_mutation|inject_top
    adaptive_mutation: bool | None = None
    adaptive_mutation_increment: float | None = None
    adaptive_mutation_cap: float | None = None
    # Forced-arbitrage (scalping) toggle — Issue 05. When true, every
    # aggressive fill auto-generates a passive counter-order N ticks
    # away on the opposite side. None = use config default.
    scalping_mode: bool | None = None
    # Smoke-test gate — Session 04 of naked-clip-and-stability. Default
    # ON: a 2-agent × 3-episode probe runs before the full population
    # to catch policy-loss / entropy / arbs-closed regressions before
    # burning hours of compute. See agents/smoke_test.py and
    # plans/naked-clip-and-stability/hard_constraints.md §15.
    smoke_test_first: bool = True


class SmokeAssertionPayload(BaseModel):
    """One assertion row for the UI failure modal."""
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str


class SmokeTestResultPayload(BaseModel):
    """Latest probe outcome — served by ``GET /training/smoke-test``.

    ``None`` at the endpoint level (empty response) means no probe has
    run since the worker started. ``passed=true`` means the full
    population was subsequently launched; ``passed=false`` means the
    launch was blocked and the UI should surface the failure modal.
    """
    passed: bool
    assertions: list[SmokeAssertionPayload]
    probe_model_ids: list[str]


class StartTrainingResponseExtra(BaseModel):
    """Fields added to ``StartTrainingResponse`` when the launch path
    runs a smoke-test probe.

    Kept as a separate model so legacy test fixtures that ignore the
    field keep validating; the main response inherits it below.
    """
    smoke_test_result: SmokeTestResultPayload | None = None


class ResumeTrainingRequest(BaseModel):
    plan_id: str


class ResumeTrainingResponse(BaseModel):
    run_id: str
    session: int
    start_generation: int
    n_generations: int


class ArchitectureInfo(BaseModel):
    name: str
    description: str


class GeneticsInfo(BaseModel):
    population_size: int
    n_elite: int
    selection_top_pct: float
    mutation_rate: float


class HyperparamSchemaEntry(BaseModel):
    """One gene in the hyperparameter search-range schema.

    Returned by ``GET /api/training/hyperparameter-schema`` so the UI can
    render the right widget for each gene without hardcoding the list.
    """
    name: str
    type: str  # "float" | "float_log" | "int" | "int_choice" | "str_choice"
    min: float | None = None
    max: float | None = None
    choices: list | None = None
    source_file: str  # e.g. "config.yaml#hyperparameters.search_ranges.<name>"


class StartTrainingResponse(BaseModel):
    run_id: str
    train_days: list[str]
    test_days: list[str]
    n_generations: int
    n_epochs: int
    # Session 04 (naked-clip-and-stability) — when the launch ran a
    # smoke-test probe, the result is attached here. ``None`` when the
    # operator ticked the checkbox OFF, or on legacy clients that
    # pre-date the gate. When ``smoke_test_result.passed == false``,
    # the full population was NOT launched; the frontend opens a
    # failure modal with per-assertion detail.
    smoke_test_result: SmokeTestResultPayload | None = None


class StopTrainingResponse(BaseModel):
    detail: str


class FinishTrainingResponse(BaseModel):
    detail: str


class EvaluateRequest(BaseModel):
    """Body for POST /evaluate — manual standalone evaluation request."""
    model_ids: list[str]
    test_dates: list[str] | None = None  # None = all available processed days


class EvaluateResponse(BaseModel):
    accepted: bool
    job_id: str
    model_count: int
    day_count: int


class EvaluateStatus(BaseModel):
    """Lightweight status poll for the evaluation page."""
    running: bool
    phase: str | None = None
    detail: str | None = None
    process: ProgressSnapshot | None = None
    item: ProgressSnapshot | None = None
    manual_evaluation: bool = False


class BettingConstraints(BaseModel):
    max_back_price: float | None = None
    max_lay_price: float | None = None
    min_seconds_before_off: int = 0
    reevaluate_garaged_default: bool = True


# ── Log Paths ────────────────────────────────────────────────────────


class LogSubdir(BaseModel):
    name: str
    file_count: int
    total_size_bytes: int


class LogPathsResponse(BaseModel):
    logs_root: str
    subdirs: list[LogSubdir]
