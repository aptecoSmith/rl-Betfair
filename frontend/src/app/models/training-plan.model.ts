/**
 * Models for the Session 4 training-plan endpoints and the Session 8
 * hyperparameter-schema endpoint.
 *
 * Mirrors the JSON shapes returned by api/routers/training_plans.py and
 * api/routers/training.py — keep in sync if either router changes.
 */

export type GeneType =
  | 'float'
  | 'float_log'
  | 'int'
  | 'int_choice'
  | 'str_choice';

/** One entry from GET /api/training/hyperparameter-schema. */
export interface HyperparamSchemaEntry {
  name: string;
  type: GeneType;
  min: number | null;
  max: number | null;
  choices: (number | string)[] | null;
  source_file: string;
}

/** A range override on a single gene, mirrors HyperparamSpec on the backend. */
export interface HpRangeOverride {
  type: GeneType;
  min?: number | null;
  max?: number | null;
  choices?: (number | string)[] | null;
}

export interface GenerationOutcome {
  generation: number;
  recorded_at: string;
  best_fitness: number;
  mean_fitness: number;
  architectures_alive: string[];
  architectures_died: string[];
  n_agents: number;
  notes: string;
}

export interface TrainingPlan {
  plan_id: string;
  name: string;
  created_at: string;
  population_size: number;
  architectures: string[];
  arch_mix?: Record<string, number> | null;
  hp_ranges: Record<string, HpRangeOverride>;
  arch_lr_ranges?: Record<string, HpRangeOverride> | null;
  seed?: number | null;
  min_arch_samples: number;
  notes: string;
  outcomes: GenerationOutcome[];
  starting_budget?: number | null;
  exploration_strategy?: string;
  manual_seed_point?: Record<string, number | string> | null;
  n_generations?: number;
  n_epochs?: number;
  status?: 'draft' | 'running' | 'completed' | 'failed' | 'paused';
  current_generation?: number | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ValidationIssue {
  code: string;
  severity: 'warning' | 'error';
  message: string;
  field?: string | null;
}

export interface TrainingPlanListResponse {
  plans: TrainingPlan[];
  count: number;
}

export interface TrainingPlanDetailResponse {
  plan: TrainingPlan;
  validation: ValidationIssue[];
}

/** Shape of POST /api/training-plans payload. */
export interface TrainingPlanPayload {
  name: string;
  population_size: number;
  architectures: string[];
  hp_ranges: Record<string, HpRangeOverride>;
  arch_mix?: Record<string, number> | null;
  arch_lr_ranges?: Record<string, HpRangeOverride> | null;
  seed?: number | null;
  min_arch_samples?: number;
  notes?: string;
  starting_budget?: number | null;
  exploration_strategy?: string;
  manual_seed_point?: Record<string, number | string> | null;
  n_generations?: number;
  n_epochs?: number;
}

export interface GeneCoverageEntry {
  name: string;
  type: GeneType;
  bucket_edges: number[];
  bucket_counts: number[];
  is_well_covered: boolean;
  total_samples: number;
}

export interface CoverageReport {
  total_agents: number;
  arch_counts: Record<string, number>;
  arch_undercovered: string[];
  gene_coverage: GeneCoverageEntry[];
}

export interface CoverageResponse {
  report: CoverageReport;
  biased_genes: string[];
}
