import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { Router } from '@angular/router';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry } from '../models/scoreboard.model';
import { DecimalPipe, CurrencyPipe, PercentPipe } from '@angular/common';

/** Colour palette for generations — cycles if more generations than colours. */
const GENERATION_COLOURS = [
  '#2196F3', // blue
  '#4CAF50', // green
  '#FF9800', // orange
  '#9C27B0', // purple
  '#F44336', // red
  '#00BCD4', // cyan
  '#795548', // brown
  '#607D8B', // blue-grey
  '#E91E63', // pink
  '#CDDC39', // lime
];

@Component({
  selector: 'app-scoreboard',
  standalone: true,
  imports: [DecimalPipe, CurrencyPipe, PercentPipe],
  templateUrl: './scoreboard.html',
  styleUrl: './scoreboard.scss',
})
export class Scoreboard implements OnInit {
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private readonly selectionState = inject(SelectionStateService);

  readonly models = signal<ScoreboardEntry[]>([]);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);

  readonly rankedModels = computed(() =>
    this.models()
      .slice()
      .sort((a, b) => (b.composite_score ?? -Infinity) - (a.composite_score ?? -Infinity))
  );

  ngOnInit(): void {
    this.loadScoreboard();
  }

  loadScoreboard(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getScoreboard().subscribe({
      next: (res) => {
        this.models.set(res.models);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err.message || 'Failed to load scoreboard');
        this.loading.set(false);
      },
    });
  }

  generationColour(generation: number): string {
    return GENERATION_COLOURS[generation % GENERATION_COLOURS.length];
  }

  shortId(modelId: string): string {
    return modelId.substring(0, 8);
  }

  onRowClick(model: ScoreboardEntry): void {
    this.selectionState.selectedModelId.set(model.model_id);
    this.router.navigate(['/models', model.model_id]);
  }

  onToggleGarage(event: Event, model: ScoreboardEntry): void {
    event.stopPropagation();
    const newState = !model.garaged;
    this.api.toggleGarage(model.model_id, newState).subscribe({
      next: () => {
        this.models.update(models =>
          models.map(m =>
            m.model_id === model.model_id ? { ...m, garaged: newState } : m
          )
        );
      },
    });
  }
}
