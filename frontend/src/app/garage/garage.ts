import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { Router } from '@angular/router';
import { DecimalPipe, CurrencyPipe, PercentPipe, SlicePipe } from '@angular/common';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry } from '../models/scoreboard.model';

const GENERATION_COLOURS = [
  '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
  '#00BCD4', '#795548', '#607D8B', '#E91E63', '#CDDC39',
];

@Component({
  selector: 'app-garage',
  standalone: true,
  imports: [DecimalPipe, CurrencyPipe, PercentPipe, SlicePipe],
  templateUrl: './garage.html',
  styleUrl: './garage.scss',
})
export class Garage implements OnInit {
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
    this.loadGarage();
  }

  loadGarage(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getGarage().subscribe({
      next: (res) => {
        this.models.set(res.models);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err.message || 'Failed to load garage');
        this.loading.set(false);
      },
    });
  }

  onRemoveFromGarage(event: Event, model: ScoreboardEntry): void {
    event.stopPropagation();
    this.api.toggleGarage(model.model_id, false).subscribe({
      next: () => {
        this.models.update(models => models.filter(m => m.model_id !== model.model_id));
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
}
