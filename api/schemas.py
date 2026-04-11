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
    detail: str | None = None
    last_agent_score: float | None = None
    worker_connected: bool = False


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


class BettingConstraints(BaseModel):
    max_back_price: float | None = None
    max_lay_price: float | None = None
    min_seconds_before_off: int = 0
    reevaluate_garaged_default: bool = True
