import { Component, OnInit, OnDestroy, inject, signal, computed, ElementRef, ViewChild } from '@angular/core';
import { DecimalPipe, CurrencyPipe, UpperCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../services/api.service';
import { SelectionStateService } from '../services/selection-state.service';
import { ScoreboardEntry } from '../models/scoreboard.model';
import {
  ReplayDayResponse,
  ReplayRaceResponse,
  RaceSummary,
  ReplayTick,
  BetEvent,
  TickRunner,
} from '../models/replay.model';

const RUNNER_COLOURS = [
  '#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
  '#00BCD4', '#F44336', '#CDDC39', '#795548', '#607D8B',
  '#3F51B5', '#009688', '#FF5722', '#8BC34A',
];

@Component({
  selector: 'app-race-replay',
  standalone: true,
  imports: [DecimalPipe, CurrencyPipe, UpperCasePipe, FormsModule],
  templateUrl: './race-replay.html',
  styleUrl: './race-replay.scss',
})
export class RaceReplay implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly selectionState = inject(SelectionStateService);

  @ViewChild('chartCanvas') chartCanvasRef!: ElementRef<HTMLCanvasElement>;

  // ── Model selection ──
  readonly models = signal<ScoreboardEntry[]>([]);
  readonly selectedModelId = signal<string | null>(null);

  // ── Date selection ──
  readonly dates = signal<string[]>([]);
  readonly selectedDate = signal<string | null>(null);

  // ── Race selection ──
  readonly races = signal<RaceSummary[]>([]);
  readonly selectedRaceId = signal<string | null>(null);

  // ── Race data ──
  readonly raceData = signal<ReplayRaceResponse | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  // ── Playback ──
  readonly currentTickIndex = signal(0);
  readonly playing = signal(false);
  readonly playbackSpeed = signal(1);
  private playbackTimer: ReturnType<typeof setInterval> | null = null;

  // ── Selected runner for order book ──
  readonly selectedRunnerId = signal<number | null>(null);

  // ── Derived data ──
  readonly ticks = computed(() => this.raceData()?.ticks ?? []);
  readonly allBets = computed(() => this.raceData()?.all_bets ?? []);
  readonly winnerSelectionId = computed(() => this.raceData()?.winner_selection_id ?? null);

  readonly runnerNames = computed(() => {
    const data = this.raceData();
    if (!data || data.ticks.length === 0) return new Map<number, string>();
    const map = new Map<number, string>();
    for (const bet of data.all_bets) {
      map.set(bet.runner_id, bet.runner_name);
    }
    // Also try to populate from tick runners
    for (const tick of data.ticks) {
      for (const r of tick.runners) {
        if (!map.has(r.selection_id)) {
          map.set(r.selection_id, `Runner ${r.selection_id}`);
        }
      }
    }
    return map;
  });

  readonly runnerIds = computed(() => {
    const t = this.ticks();
    if (t.length === 0) return [];
    const ids = new Set<number>();
    for (const tick of t) {
      for (const r of tick.runners) {
        ids.add(r.selection_id);
      }
    }
    return Array.from(ids).sort((a, b) => a - b);
  });

  readonly currentTick = computed(() => {
    const t = this.ticks();
    const idx = this.currentTickIndex();
    return t.length > 0 && idx < t.length ? t[idx] : null;
  });

  readonly currentOrderBook = computed(() => {
    const tick = this.currentTick();
    const runnerId = this.selectedRunnerId();
    if (!tick || runnerId == null) return null;
    return tick.runners.find(r => r.selection_id === runnerId) ?? null;
  });

  readonly timeToOff = computed(() => {
    const tick = this.currentTick();
    if (!tick) return null;
    const data = this.raceData();
    if (!data) return null;
    const startTime = new Date(data.market_start_time).getTime();
    const tickTime = new Date(tick.timestamp).getTime();
    return Math.round((startTime - tickTime) / 1000);
  });

  readonly summaryStats = computed(() => {
    const bets = this.allBets();
    const data = this.raceData();
    const totalBets = bets.length;
    const totalPnl = data?.race_pnl ?? 0;
    const earlyPicks = bets.filter(b => b.seconds_to_off >= 300 && b.pnl > 0).length;
    return { totalBets, totalPnl, earlyPicks };
  });

  // ── Chart data ──
  readonly chartData = computed(() => {
    const t = this.ticks();
    const data = this.raceData();
    if (t.length === 0 || !data) return [];

    const startTime = new Date(data.market_start_time).getTime();
    const ids = this.runnerIds();

    return ids.map((id, i) => ({
      runnerId: id,
      name: this.runnerNames().get(id) ?? `Runner ${id}`,
      colour: RUNNER_COLOURS[i % RUNNER_COLOURS.length],
      isWinner: id === this.winnerSelectionId(),
      points: t.map((tick, tickIdx) => {
        const runner = tick.runners.find(r => r.selection_id === id);
        const tickTime = new Date(tick.timestamp).getTime();
        const secondsToOff = (startTime - tickTime) / 1000;
        return {
          tickIndex: tickIdx,
          secondsToOff,
          ltp: runner?.last_traded_price ?? 0,
        };
      }).filter(p => p.ltp > 0),
    }));
  });

  readonly chartSvgPath = computed(() => {
    const data = this.chartData();
    const t = this.ticks();
    if (data.length === 0 || t.length === 0) return { paths: [], xRange: [0, 0], yRange: [0, 0] };

    const allPoints = data.flatMap(d => d.points);
    const xMin = Math.min(...allPoints.map(p => p.secondsToOff));
    const xMax = Math.max(...allPoints.map(p => p.secondsToOff));
    const yMin = Math.max(1, Math.min(...allPoints.map(p => p.ltp)) * 0.9);
    const yMax = Math.min(100, Math.max(...allPoints.map(p => p.ltp)) * 1.1);

    const w = 800;
    const h = 400;

    const scaleX = (v: number) => w - ((v - xMin) / (xMax - xMin || 1)) * w;
    const scaleY = (v: number) => h - ((v - yMin) / (yMax - yMin || 1)) * h;

    const paths = data.map(runner => {
      if (runner.points.length < 2) return { ...runner, d: '' };
      const d = runner.points
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.secondsToOff).toFixed(1)} ${scaleY(p.ltp).toFixed(1)}`)
        .join(' ');
      return { ...runner, d };
    });

    return { paths, xRange: [xMin, xMax], yRange: [yMin, yMax] };
  });

  readonly cursorX = computed(() => {
    const svg = this.chartSvgPath();
    const tick = this.currentTick();
    const data = this.raceData();
    if (!tick || !data || svg.xRange[0] === svg.xRange[1]) return null;

    const startTime = new Date(data.market_start_time).getTime();
    const tickTime = new Date(tick.timestamp).getTime();
    const secondsToOff = (startTime - tickTime) / 1000;

    const w = 800;
    const [xMin, xMax] = svg.xRange;
    return w - ((secondsToOff - xMin) / (xMax - xMin || 1)) * w;
  });

  ngOnInit(): void {
    this.loadModels();
    this.restoreState();
  }

  private restoreState(): void {
    const modelId = this.selectionState.selectedModelId();
    if (!modelId) return;

    this.selectedModelId.set(modelId);
    this.loadDates(modelId);

    const date = this.selectionState.replayDate();
    if (date) {
      this.selectedDate.set(date);
      this.loadRaces(modelId, date);

      const raceId = this.selectionState.replayRaceId();
      if (raceId) {
        this.selectedRaceId.set(raceId);
        this.loadRace(modelId, date, raceId);
      }
    }
  }

  ngOnDestroy(): void {
    this.stopPlayback();
  }

  loadModels(): void {
    this.api.getScoreboard().subscribe({
      next: (res) => this.models.set(res.models),
      error: () => this.error.set('Failed to load models'),
    });
  }

  onModelChange(modelId: string): void {
    this.selectedModelId.set(modelId);
    this.selectionState.selectedModelId.set(modelId);
    this.selectedDate.set(null);
    this.selectedRaceId.set(null);
    this.selectionState.replayDate.set(null);
    this.selectionState.replayRaceId.set(null);
    this.raceData.set(null);
    this.dates.set([]);
    this.races.set([]);
    this.stopPlayback();
    this.loadDates(modelId);
  }

  onDateChange(date: string): void {
    this.selectedDate.set(date);
    this.selectionState.replayDate.set(date);
    this.selectedRaceId.set(null);
    this.selectionState.replayRaceId.set(null);
    this.raceData.set(null);
    this.races.set([]);
    this.stopPlayback();
    const modelId = this.selectedModelId();
    if (modelId) this.loadRaces(modelId, date);
  }

  onRaceChange(raceId: string): void {
    this.selectedRaceId.set(raceId);
    this.selectionState.replayRaceId.set(raceId);
    this.raceData.set(null);
    this.stopPlayback();
    const modelId = this.selectedModelId();
    const date = this.selectedDate();
    if (modelId && date) this.loadRace(modelId, date, raceId);
  }

  private loadDates(modelId: string): void {
    // Get evaluation days from the model detail
    this.api.getModelDetail(modelId).subscribe({
      next: (res) => {
        const evalDates = (res.metrics_history ?? []).map((m: any) => m.date).sort();
        this.dates.set(evalDates);
      },
      error: () => this.error.set('Failed to load dates'),
    });
  }

  private loadRaces(modelId: string, date: string): void {
    this.loading.set(true);
    this.api.getReplayDay(modelId, date).subscribe({
      next: (res) => {
        this.races.set(res.races);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err.error?.detail ?? 'Failed to load races');
        this.loading.set(false);
      },
    });
  }

  private loadRace(modelId: string, date: string, raceId: string): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getReplayRace(modelId, date, raceId).subscribe({
      next: (res) => {
        this.raceData.set(res);
        this.currentTickIndex.set(0);
        this.loading.set(false);
        // Auto-select first runner
        if (res.ticks.length > 0 && res.ticks[0].runners.length > 0) {
          this.selectedRunnerId.set(res.ticks[0].runners[0].selection_id);
        }
      },
      error: (err) => {
        this.error.set(err.error?.detail ?? 'Failed to load race data');
        this.loading.set(false);
      },
    });
  }

  // ── Playback controls ──

  togglePlayback(): void {
    if (this.playing()) {
      this.stopPlayback();
    } else {
      this.startPlayback();
    }
  }

  startPlayback(): void {
    if (this.ticks().length === 0) return;
    this.playing.set(true);
    this.playbackTimer = setInterval(() => {
      const next = this.currentTickIndex() + 1;
      if (next >= this.ticks().length) {
        this.stopPlayback();
      } else {
        this.currentTickIndex.set(next);
      }
    }, 1000 / this.playbackSpeed());
  }

  stopPlayback(): void {
    this.playing.set(false);
    if (this.playbackTimer) {
      clearInterval(this.playbackTimer);
      this.playbackTimer = null;
    }
  }

  setSpeed(speed: number): void {
    this.playbackSpeed.set(speed);
    if (this.playing()) {
      this.stopPlayback();
      this.startPlayback();
    }
  }

  seekToTick(index: number): void {
    this.currentTickIndex.set(index);
  }

  onBetClick(bet: BetEvent): void {
    const t = this.ticks();
    const idx = t.findIndex(tick => tick.timestamp === bet.tick_timestamp);
    if (idx >= 0) {
      this.currentTickIndex.set(idx);
      this.selectedRunnerId.set(bet.runner_id);
    }
  }

  selectRunner(runnerId: number): void {
    this.selectedRunnerId.set(runnerId);
  }

  runnerColour(runnerId: number): string {
    const ids = this.runnerIds();
    const idx = ids.indexOf(runnerId);
    return RUNNER_COLOURS[idx >= 0 ? idx % RUNNER_COLOURS.length : 0];
  }

  shortId(id: string): string {
    return id.substring(0, 8);
  }

  formatSecondsToOff(seconds: number): string {
    const mins = Math.floor(Math.abs(seconds) / 60);
    const secs = Math.abs(seconds) % 60;
    const sign = seconds < 0 ? '+' : '-';
    return `${sign}${mins}:${secs.toString().padStart(2, '0')}`;
  }
}
