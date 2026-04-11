import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class SelectionStateService {
  // ── Global (shared across pages) ──
  readonly selectedModelId = signal<string | null>(null);

  // ── Bet Explorer ──
  readonly betExplorerFilters = signal<{
    date: string;
    race: string;
    runner: string;
    action: string;
    outcome: string;
    betType: string;
  }>({ date: '', race: '', runner: '', action: '', outcome: '', betType: 'BOTH' });

  // ── Race Replay ──
  readonly replayDate = signal<string | null>(null);
  readonly replayRaceId = signal<string | null>(null);

  // ── Training Monitor ──
  readonly trainingFormValues = signal<{
    generations: number;
    epochs: number;
    populationSize: number | null;
  }>({ generations: 3, epochs: 3, populationSize: null });
}
