import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, JsonPipe } from '@angular/common';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { ApiService } from '../services/api.service';
import {
  CoverageReport,
  HpRangeOverride,
  HyperparamSchemaEntry,
  TrainingPlan,
  TrainingPlanPayload,
  ValidationIssue,
} from '../models/training-plan.model';
import { GeneEditor } from '../components/gene-editor/gene-editor';
import { EarlyPickValidator } from '../components/early-pick-validator/early-pick-validator';
import { CoveragePanel } from '../components/coverage-panel/coverage-panel';

interface ArchInfo { name: string; description: string; }

/**
 * Training Plans page (Session 8).
 *
 * Three modes: list / detail / editor. The editor consumes the
 * hyperparameter schema endpoint to render one widget per gene without
 * any hardcoded list, so adding a gene to config.yaml automatically
 * shows up in the editor on the next page load.
 */
@Component({
  selector: 'app-training-plans',
  standalone: true,
  imports: [FormsModule, DecimalPipe, JsonPipe, RouterLink, GeneEditor, EarlyPickValidator, CoveragePanel],
  templateUrl: './training-plans.html',
  styleUrl: './training-plans.scss',
})
export class TrainingPlans implements OnInit {
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  // ── List / detail state ─────────────────────────────────────────
  readonly plans = signal<TrainingPlan[]>([]);
  readonly loadingPlans = signal(true);
  readonly listError = signal<string | null>(null);

  readonly selectedPlan = signal<TrainingPlan | null>(null);
  readonly selectedValidation = signal<ValidationIssue[]>([]);

  readonly mode = signal<'list' | 'detail' | 'editor'>('list');

  // ── Schema + architectures ──────────────────────────────────────
  readonly schema = signal<HyperparamSchemaEntry[]>([]);
  readonly schemaError = signal<string | null>(null);
  readonly architectures = signal<ArchInfo[]>([]);

  // Genes that get the special composite widget — excluded from the
  // generic gene-editor loop below.
  readonly EARLY_PICK_GENES = new Set(['early_pick_bonus_min', 'early_pick_bonus_max']);

  readonly genericGenes = computed(() =>
    this.schema().filter(g => !this.EARLY_PICK_GENES.has(g.name) && g.name !== 'architecture_name')
  );

  readonly minSpec = computed(() => this.schema().find(g => g.name === 'early_pick_bonus_min') ?? null);
  readonly maxSpec = computed(() => this.schema().find(g => g.name === 'early_pick_bonus_max') ?? null);

  // ── Editor state ────────────────────────────────────────────────
  readonly editorName = signal('');
  readonly editorPopulationSize = signal(50);
  readonly editorSelectedArchs = signal<Set<string>>(new Set());
  readonly editorSeed = signal<number | null>(null);
  readonly editorMinArchSamples = signal(5);
  readonly editorNotes = signal('');
  readonly editorHpRanges = signal<Record<string, HpRangeOverride>>({});
  readonly editorStartingBudget = signal<number | null>(null);
  readonly editorArchLrRanges = signal<Record<string, HpRangeOverride>>({});
  readonly editorNGenerations = signal(3);
  readonly editorNEpochs = signal(3);
  readonly editorExplorationStrategy = signal<string>('random');
  readonly editorBiasToggle = signal(false);
  readonly editorSaving = signal(false);
  readonly editorErrors = signal<ValidationIssue[]>([]);
  readonly editorTopError = signal<string | null>(null);

  readonly coverageWarning = computed(() => {
    const pop = this.editorPopulationSize();
    const min = this.editorMinArchSamples();
    const archs = this.editorSelectedArchs().size;
    if (archs === 0) return null;
    if (pop < min * archs) {
      return `population_size (${pop}) < min_arch_samples (${min}) × architectures (${archs}) = ${min * archs}. Some architectures will be undersampled.`;
    }
    return null;
  });

  // ── Coverage panel state ────────────────────────────────────────
  readonly coverageReport = signal<CoverageReport | null>(null);
  readonly coverageBiased = signal<string[]>([]);
  readonly coverageError = signal<string | null>(null);

  ngOnInit(): void {
    this.loadList();
    this.loadSchema();
    this.loadArchitectures();
    this.loadCoverage();

    // If a ?plan= query param is present, open that plan's detail view
    const planId = this.route.snapshot.queryParamMap.get('plan');
    if (planId) {
      this.openDetail(planId);
    }
  }

  // ── Loaders ─────────────────────────────────────────────────────
  loadList(): void {
    this.loadingPlans.set(true);
    this.listError.set(null);
    this.api.listTrainingPlans().subscribe({
      next: (resp) => {
        this.plans.set(resp.plans);
        this.loadingPlans.set(false);
      },
      error: (err) => {
        this.listError.set(err?.error?.detail ?? 'Failed to load training plans');
        this.loadingPlans.set(false);
      },
    });
  }

  loadSchema(): void {
    this.api.getHyperparameterSchema().subscribe({
      next: (entries) => this.schema.set(entries),
      error: (err) => this.schemaError.set(err?.error?.detail ?? 'Failed to load schema'),
    });
  }

  loadArchitectures(): void {
    this.api.getArchitectures().subscribe({
      next: (a) => this.architectures.set(a),
      error: () => {},
    });
  }

  loadCoverage(): void {
    this.api.getTrainingPlanCoverage().subscribe({
      next: (resp) => {
        this.coverageReport.set(resp.report);
        this.coverageBiased.set(resp.biased_genes);
      },
      error: (err) => this.coverageError.set(err?.error?.detail ?? 'Failed to load coverage'),
    });
  }

  // ── Mode switches ──────────────────────────────────────────────
  openList(): void { this.mode.set('list'); }

  openDetail(planId: string): void {
    this.api.getTrainingPlan(planId).subscribe({
      next: (resp) => {
        this.selectedPlan.set(resp.plan);
        this.selectedValidation.set(resp.validation);
        this.mode.set('detail');
      },
      error: (err) => this.listError.set(err?.error?.detail ?? 'Failed to load plan'),
    });
  }

  openEditor(): void {
    // Initialise hp_ranges from the schema's defaults so the form is
    // pre-populated with the canonical ranges.
    const ranges: Record<string, HpRangeOverride> = {};
    for (const g of this.schema()) {
      if (g.name === 'architecture_name') continue;
      if (g.type === 'float' || g.type === 'float_log' || g.type === 'int') {
        ranges[g.name] = { type: g.type, min: g.min, max: g.max };
      } else {
        ranges[g.name] = { type: g.type, choices: g.choices };
      }
    }
    this.editorHpRanges.set(ranges);
    this.editorArchLrRanges.set({});
    this.editorStartingBudget.set(null);
    this.editorNGenerations.set(3);
    this.editorNEpochs.set(3);
    this.editorName.set('');
    this.editorErrors.set([]);
    this.editorTopError.set(null);
    this.mode.set('editor');
  }

  // ── Editor mutations ────────────────────────────────────────────
  toggleArchSelection(name: string): void {
    const next = new Set(this.editorSelectedArchs());
    if (next.has(name)) next.delete(name);
    else next.add(name);
    this.editorSelectedArchs.set(next);
  }

  isArchSelected(name: string): boolean {
    return this.editorSelectedArchs().has(name);
  }

  selectedArchsArray(): string[] {
    return Array.from(this.editorSelectedArchs());
  }

  updateGene(name: string, value: HpRangeOverride): void {
    const next = { ...this.editorHpRanges(), [name]: value };
    this.editorHpRanges.set(next);
  }

  updateMinGene(value: HpRangeOverride): void {
    this.updateGene('early_pick_bonus_min', value);
  }
  updateMaxGene(value: HpRangeOverride): void {
    this.updateGene('early_pick_bonus_max', value);
  }

  geneValue(name: string): HpRangeOverride {
    return this.editorHpRanges()[name] ?? { type: 'float' };
  }

  // ── Arch LR override editor ─────────────────────────────────────
  addArchLrOverride(arch: string): void {
    const lrSpec = this.schema().find(g => g.name === 'learning_rate');
    if (!lrSpec) return;
    const next = {
      ...this.editorArchLrRanges(),
      [arch]: { type: 'float_log' as const, min: lrSpec.min, max: lrSpec.max },
    };
    this.editorArchLrRanges.set(next);
  }

  removeArchLrOverride(arch: string): void {
    const next = { ...this.editorArchLrRanges() };
    delete next[arch];
    this.editorArchLrRanges.set(next);
  }

  updateArchLrOverride(arch: string, value: HpRangeOverride): void {
    const next = { ...this.editorArchLrRanges(), [arch]: value };
    this.editorArchLrRanges.set(next);
  }

  archLrEntries(): { arch: string; value: HpRangeOverride }[] {
    const r = this.editorArchLrRanges();
    return Object.keys(r).map(arch => ({ arch, value: r[arch] }));
  }

  hasArchLrOverride(arch: string): boolean {
    return arch in this.editorArchLrRanges();
  }

  lrSpec(): HyperparamSchemaEntry | null {
    return this.schema().find(g => g.name === 'learning_rate') ?? null;
  }

  // ── Bias toward uncovered ───────────────────────────────────────
  applyBias(): void {
    // The backend's bias_sampler currently nudges by reducing the
    // upper bound of well-covered buckets. Here we ask the API for
    // the coverage report and tighten the user's editable hp_ranges
    // toward the bucket edges of any biased gene's first empty
    // bucket. This is a UI preview — the real bias is applied
    // server-side.
    this.api.getTrainingPlanCoverage().subscribe({
      next: (resp) => {
        this.coverageReport.set(resp.report);
        this.coverageBiased.set(resp.biased_genes);
        const next = { ...this.editorHpRanges() };
        for (const gene of resp.report.gene_coverage ?? []) {
          if (!resp.biased_genes.includes(gene.name)) continue;
          const counts = gene.bucket_counts;
          const edges = gene.bucket_edges;
          if (!counts || !edges || edges.length < 2) continue;
          // Find the first empty bucket and recommend tightening
          // toward [edge_i, edge_{i+1}].
          const firstEmpty = counts.findIndex(c => c === 0);
          if (firstEmpty < 0) continue;
          const lo = edges[firstEmpty];
          const hi = edges[firstEmpty + 1];
          if (lo == null || hi == null) continue;
          const current = next[gene.name];
          if (!current) continue;
          next[gene.name] = { ...current, min: lo, max: hi };
        }
        this.editorHpRanges.set(next);
        this.editorBiasToggle.set(true);
      },
      error: () => {},
    });
  }

  // ── Save ────────────────────────────────────────────────────────
  savePlan(): void {
    this.editorErrors.set([]);
    this.editorTopError.set(null);
    if (!this.editorName().trim()) {
      this.editorTopError.set('Plan name is required');
      return;
    }
    if (this.editorSelectedArchs().size === 0) {
      this.editorTopError.set('At least one architecture must be selected');
      return;
    }
    const payload: TrainingPlanPayload = {
      name: this.editorName(),
      population_size: this.editorPopulationSize(),
      architectures: this.selectedArchsArray(),
      hp_ranges: this.editorHpRanges(),
      arch_lr_ranges: Object.keys(this.editorArchLrRanges()).length
        ? this.editorArchLrRanges()
        : null,
      seed: this.editorSeed(),
      min_arch_samples: this.editorMinArchSamples(),
      notes: this.editorNotes(),
      starting_budget: this.editorStartingBudget(),
      exploration_strategy: this.editorExplorationStrategy(),
      n_generations: this.editorNGenerations(),
      n_epochs: this.editorNEpochs(),
    };
    this.editorSaving.set(true);
    this.api.createTrainingPlan(payload).subscribe({
      next: (resp) => {
        this.editorSaving.set(false);
        this.plans.update(prev => [resp.plan, ...prev]);
        this.selectedPlan.set(resp.plan);
        this.selectedValidation.set(resp.validation);
        this.mode.set('detail');
      },
      error: (err) => {
        this.editorSaving.set(false);
        const detail = err?.error?.detail;
        if (detail && typeof detail === 'object' && Array.isArray(detail.issues)) {
          this.editorErrors.set(detail.issues);
          this.editorTopError.set(detail.message ?? 'Plan failed validation');
        } else {
          this.editorTopError.set(detail ?? err?.message ?? 'Failed to save plan');
        }
      },
    });
  }

  // ── Delete a plan ──────────────────────────────────────────────
  readonly deleting = signal(false);

  deletePlan(): void {
    const plan = this.selectedPlan();
    if (!plan || !confirm(`Delete plan "${plan.name}"? This cannot be undone.`)) return;
    this.deleting.set(true);
    this.api.deleteTrainingPlan(plan.plan_id).subscribe({
      next: () => {
        this.deleting.set(false);
        this.plans.update(prev => prev.filter(p => p.plan_id !== plan.plan_id));
        this.selectedPlan.set(null);
        this.mode.set('list');
      },
      error: (err) => {
        this.deleting.set(false);
        this.launchError.set(err?.error?.detail ?? 'Failed to delete plan');
      },
    });
  }

  // ── Launch a plan ───────────────────────────────────────────────
  readonly launching = signal(false);
  readonly launchError = signal<string | null>(null);

  startPlan(): void {
    const plan = this.selectedPlan();
    if (!plan) return;
    this.launching.set(true);
    this.launchError.set(null);
    this.api.startTraining({
      plan_id: plan.plan_id,
      n_generations: plan.n_generations ?? 3,
      n_epochs: plan.n_epochs ?? 3,
      population_size: plan.population_size,
      architectures: plan.architectures,
      seed: plan.seed ?? null,
      starting_budget: plan.starting_budget ?? null,
    }).subscribe({
      next: () => {
        this.launching.set(false);
        this.router.navigate(['/training']);
      },
      error: (err) => {
        this.launching.set(false);
        this.launchError.set(err?.error?.detail ?? err?.message ?? 'Failed to start training');
      },
    });
  }

  // ── Helpers ─────────────────────────────────────────────────────
  /** Format ISO timestamp to short readable form, e.g. "7 Apr 2026, 01:34" */
  shortDate(iso: string): string {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
        + ', ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    } catch {
      return iso;
    }
  }

  errorForField(field: string): ValidationIssue | null {
    return this.editorErrors().find(i => i.field === field) ?? null;
  }

  validationErrors(): ValidationIssue[] {
    return this.editorErrors().filter(i => i.severity === 'error');
  }
  validationWarnings(): ValidationIssue[] {
    return this.editorErrors().filter(i => i.severity === 'warning');
  }
}
