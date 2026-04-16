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


class ScoreboardResponse(BaseModel):
    models: list[ScoreboardEntry]


class GarageToggleRequest(BaseModel):
    garaged: bool


class GarageToggleResponse(BaseModel):
    model_id: str
    garaged: bool


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
