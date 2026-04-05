import {
  Component, OnInit, OnDestroy, inject, signal, computed,
  ElementRef, ViewChild, effect, AfterViewInit,
} from '@angular/core';
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
import uPlot from 'uplot';

const RUNNER_COLOURS = [
  '#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
  '#00BCD4', '#F44336', '#CDDC39', '#795548', '#607D8B',
  '#3F51B5', '#009688', '#FF5722', '#8BC34A',
];

export interface BetCard {
  index: number;
  bet: BetEvent;
  runningBalance: number;
  liability: number;
}

export interface ConclusionData {
  winnerName: string | null;
  winnerBacked: boolean;
  winnerBackPrice: number | null;
  totalBets: number;
  totalStake: number;
  totalPnl: number;
  wonCount: number;
  lostCount: number;
  earlyPicks: number;
  betResults: {
    action: string;
    runnerName: string;
    price: number;
    pnl: number;
    won: boolean;
  }[];
}

@Component({
  selector: 'app-race-replay',
  standalone: true,
  imports: [DecimalPipe, CurrencyPipe, UpperCasePipe, FormsModule],
  templateUrl: './race-replay.html',
  styleUrl: './race-replay.scss',
})
export class RaceReplay implements OnInit, OnDestroy, AfterViewInit {
  private readonly api = inject(ApiService);
  private readonly selectionState = inject(SelectionStateService);

  @ViewChild('chartEl') chartElRef!: ElementRef<HTMLDivElement>;

  private chart: uPlot | null = null;
  private resizeObserver: ResizeObserver | null = null;

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

  // ── Chart interaction ──
  readonly highlightedBetIndex = signal<number | null>(null);
  readonly visibleRunners = signal<Set<number>>(new Set());

  // ── Derived data ──
  readonly ticks = computed(() => this.raceData()?.ticks ?? []);
  readonly allBets = computed(() => this.raceData()?.all_bets ?? []);
  readonly winnerSelectionId = computed(() => this.raceData()?.winner_selection_id ?? null);

  readonly runnerNames = computed(() => {
    const data = this.raceData();
    if (!data || data.ticks.length === 0) return new Map<number, string>();
    const map = new Map<number, string>();
    // Primary: runner_names from backend (loaded from _runners.parquet)
    if (data.runner_names) {
      for (const [sid, name] of Object.entries(data.runner_names)) {
        map.set(Number(sid), name);
      }
    }
    // Secondary: bet records (in case runner_names missing)
    for (const bet of data.all_bets) {
      if (!map.has(bet.runner_id)) {
        map.set(bet.runner_id, bet.runner_name);
      }
    }
    // Fallback: selection ID for any remaining
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

  // ── uPlot data ──
  readonly uPlotData = computed(() => {
    const t = this.ticks();
    const data = this.raceData();
    if (t.length === 0 || !data) return null;

    const startTime = new Date(data.market_start_time).getTime();
    const ids = this.runnerIds();

    // X values: seconds to off (positive = before off)
    const xValues = t.map(tick => {
      const tickTime = new Date(tick.timestamp).getTime();
      return (startTime - tickTime) / 1000;
    });

    // Y values per runner
    const ySeriesArrays = ids.map(id =>
      t.map(tick => {
        const runner = tick.runners.find(r => r.selection_id === id);
        const ltp = runner?.last_traded_price ?? 0;
        return ltp > 0 ? ltp : null;
      }) as (number | null)[]
    );

    return {
      xValues,
      ySeriesArrays,
      runnerIds: ids,
    };
  });

  readonly chartData = computed(() => {
    const t = this.ticks();
    const data = this.raceData();
    if (t.length === 0 || !data) return [];

    const ids = this.runnerIds();

    // Sum stakes per runner from bets
    const stakeByRunner = new Map<number, number>();
    for (const bet of data.all_bets) {
      stakeByRunner.set(bet.runner_id, (stakeByRunner.get(bet.runner_id) ?? 0) + bet.stake);
    }

    return ids.map((id, i) => ({
      runnerId: id,
      name: this.runnerNames().get(id) ?? `Runner ${id}`,
      colour: RUNNER_COLOURS[i % RUNNER_COLOURS.length],
      isWinner: id === this.winnerSelectionId(),
      totalStake: stakeByRunner.get(id) ?? 0,
    }));
  });

  // ── Bet panel ──
  readonly visibleBets = computed((): BetCard[] => {
    const bets = this.allBets();
    const t = this.ticks();
    const idx = this.currentTickIndex();
    if (bets.length === 0 || t.length === 0) return [];

    const currentTimestamp = t[idx]?.timestamp;
    if (!currentTimestamp) return [];
    const currentTime = new Date(currentTimestamp).getTime();

    let balance = 100;
    const cards: BetCard[] = [];

    for (let i = 0; i < bets.length; i++) {
      const bet = bets[i];
      const betTime = new Date(bet.tick_timestamp).getTime();
      const liability = bet.action === 'lay' ? bet.stake * (bet.price - 1) : 0;

      if (bet.action === 'back') {
        balance -= bet.stake;
      } else {
        balance -= liability;
      }

      if (betTime <= currentTime) {
        cards.push({
          index: i,
          bet,
          runningBalance: Math.round(balance * 100) / 100,
          liability: Math.round(liability * 100) / 100,
        });
      }
    }

    return cards;
  });

  readonly runningBalances = computed((): number[] => {
    const bets = this.allBets();
    let balance = 100;
    return bets.map(bet => {
      if (bet.action === 'back') {
        balance -= bet.stake;
      } else {
        balance -= bet.stake * (bet.price - 1);
      }
      return Math.round(balance * 100) / 100;
    });
  });

  // ── Conclusion panel ──
  readonly conclusionData = computed((): ConclusionData | null => {
    const data = this.raceData();
    if (!data) return null;

    const bets = this.allBets();
    const winnerId = this.winnerSelectionId();
    const winnerName = winnerId ? (this.runnerNames().get(winnerId) ?? null) : null;

    const winnerBacks = bets.filter(b => b.runner_id === winnerId && b.action === 'back');
    const winnerBacked = winnerBacks.length > 0;
    const winnerBackPrice = winnerBacked ? winnerBacks[0].price : null;

    const totalStake = bets.reduce((sum, b) => sum + b.stake, 0);
    const totalPnl = bets.reduce((sum, b) => sum + b.pnl, 0);
    const wonCount = bets.filter(b => b.pnl > 0).length;
    const lostCount = bets.filter(b => b.pnl < 0).length;
    const earlyPicks = bets.filter(b => b.seconds_to_off >= 300 && b.pnl > 0).length;

    const betResults = bets.map(b => ({
      action: b.action,
      runnerName: b.runner_name,
      price: b.price,
      pnl: b.pnl,
      won: b.pnl > 0,
    }));

    return {
      winnerName,
      winnerBacked,
      winnerBackPrice,
      totalBets: bets.length,
      totalStake: Math.round(totalStake * 100) / 100,
      totalPnl: Math.round(totalPnl * 100) / 100,
      wonCount,
      lostCount,
      earlyPicks,
      betResults,
    };
  });

  // ── Bet marker positions for uPlot overlay ──
  readonly betMarkers = computed(() => {
    const bets = this.allBets();
    const t = this.ticks();
    const data = this.raceData();
    if (bets.length === 0 || t.length === 0 || !data) return [];

    const startTime = new Date(data.market_start_time).getTime();
    const ids = this.runnerIds();

    return bets.map((bet, i) => {
      const tickIdx = t.findIndex(tick => tick.timestamp === bet.tick_timestamp);
      if (tickIdx < 0) return null;
      const tickTime = new Date(t[tickIdx].timestamp).getTime();
      const secondsToOff = (startTime - tickTime) / 1000;
      const seriesIdx = ids.indexOf(bet.runner_id);
      return {
        betIndex: i,
        tickIndex: tickIdx,
        secondsToOff,
        price: bet.price,
        action: bet.action,
        seriesIndex: seriesIdx,
        runnerId: bet.runner_id,
      };
    }).filter(m => m !== null);
  });

  ngOnInit(): void {
    this.loadModels();
    this.restoreState();
  }

  ngAfterViewInit(): void {
    // Chart creation is triggered by buildChart() called after race data loads
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
    this.destroyChart();
  }

  private destroyChart(): void {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }

  buildChart(): void {
    this.destroyChart();

    const plotData = this.uPlotData();
    if (!plotData || !this.chartElRef) return;

    const container = this.chartElRef.nativeElement;
    const ids = plotData.runnerIds;
    const winnerId = this.winnerSelectionId();
    const visible = this.visibleRunners();

    const series: uPlot.Series[] = [
      { label: 'Time to Off' },
      ...ids.map((id, i) => ({
        label: this.runnerNames().get(id) ?? `Runner ${id}`,
        stroke: RUNNER_COLOURS[i % RUNNER_COLOURS.length],
        width: id === winnerId ? 3 : 1.5,
        show: visible.size === 0 || visible.has(id),
        spanGaps: true,
      })),
    ];

    const rect = container.getBoundingClientRect();
    const width = Math.max(rect.width, 300);
    const height = Math.max(rect.height - 10, 250);

    const self = this;

    const opts: uPlot.Options = {
      width,
      height,
      cursor: {
        show: true,
        x: true,
        y: false,
      },
      scales: {
        x: {
          time: false,
          dir: -1, // right-to-left (countdown)
        },
        y: {
          auto: true,
        },
      },
      axes: [
        {
          label: 'Seconds to Off',
          stroke: '#888',
          grid: { stroke: 'rgba(255,255,255,0.05)' },
          ticks: { stroke: 'rgba(255,255,255,0.1)' },
          values: (_u: uPlot, vals: number[]) => vals.map(v => {
            const mins = Math.floor(Math.abs(v) / 60);
            const secs = Math.abs(v) % 60;
            const sign = v < 0 ? '+' : '';
            return `${sign}${mins}:${secs.toFixed(0).padStart(2, '0')}`;
          }),
        },
        {
          label: 'LTP',
          stroke: '#888',
          grid: { stroke: 'rgba(255,255,255,0.05)' },
          ticks: { stroke: 'rgba(255,255,255,0.1)' },
        },
      ],
      series,
      hooks: {
        draw: [
          (u: uPlot) => {
            const ctx = u.ctx;
            const markers = self.betMarkers();
            const currentIdx = self.currentTickIndex();

            for (const marker of markers) {
              if (marker.tickIndex > currentIdx) continue;

              const cx = u.valToPos(marker.secondsToOff, 'x', true);
              const cy = u.valToPos(marker.price, 'y', true);

              if (cx == null || cy == null || isNaN(cx) || isNaN(cy)) continue;

              ctx.save();
              ctx.fillStyle = marker.action === 'back' ? '#4CAF50' : '#F44336';
              ctx.font = 'bold 16px sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText('\u2605', cx, cy);
              ctx.restore();
            }
          },
        ],
      },
    };

    // Build initial data sliced to current tick
    const slicedData = this.getSlicedData(plotData);

    this.chart = new uPlot(opts, slicedData, container);

    // Resize observer
    this.resizeObserver = new ResizeObserver(() => {
      if (this.chart && container) {
        const r = container.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
          this.chart.setSize({ width: r.width, height: Math.max(r.height - 10, 250) });
        }
      }
    });
    this.resizeObserver.observe(container);
  }

  updateChartData(): void {
    const plotData = this.uPlotData();
    if (!this.chart || !plotData) return;
    const slicedData = this.getSlicedData(plotData);
    this.chart.setData(slicedData);
  }

  private getSlicedData(plotData: { xValues: number[]; ySeriesArrays: (number | null)[][]; runnerIds: number[] }): uPlot.AlignedData {
    const idx = this.currentTickIndex() + 1;
    const visible = this.visibleRunners();
    const ids = plotData.runnerIds;

    const x = plotData.xValues.slice(0, idx);
    const ySeries = plotData.ySeriesArrays.map((arr, i) => {
      if (visible.size > 0 && !visible.has(ids[i])) {
        return new Array(idx).fill(null);
      }
      return arr.slice(0, idx);
    });

    return [x, ...ySeries] as uPlot.AlignedData;
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
    this.destroyChart();
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
    this.destroyChart();
    const modelId = this.selectedModelId();
    if (modelId) this.loadRaces(modelId, date);
  }

  onRaceChange(raceId: string): void {
    this.selectedRaceId.set(raceId);
    this.selectionState.replayRaceId.set(raceId);
    this.raceData.set(null);
    this.stopPlayback();
    this.destroyChart();
    const modelId = this.selectedModelId();
    const date = this.selectedDate();
    if (modelId && date) this.loadRace(modelId, date, raceId);
  }

  private loadDates(modelId: string): void {
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
        // Show all data by default so the chart renders full price lines
        this.currentTickIndex.set(Math.max(0, res.ticks.length - 1));
        this.highlightedBetIndex.set(null);
        // Initialise visible runners to all
        const ids = new Set<number>();
        for (const tick of res.ticks) {
          for (const r of tick.runners) {
            ids.add(r.selection_id);
          }
        }
        this.visibleRunners.set(ids);
        this.loading.set(false);

        // Build chart after a microtask so the template has rendered
        setTimeout(() => this.buildChart(), 0);
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
        this.updateChartData();
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
    this.updateChartData();
  }

  // ── Bet panel interactions ──

  onBetCardClick(card: BetCard): void {
    const t = this.ticks();
    const idx = t.findIndex(tick => tick.timestamp === card.bet.tick_timestamp);
    if (idx >= 0) {
      this.currentTickIndex.set(idx);
      this.highlightedBetIndex.set(card.index);
      this.updateChartData();
    }
  }

  onBetMarkerClick(betIndex: number): void {
    this.highlightedBetIndex.set(betIndex);
  }

  // ── Runner legend ──

  toggleRunner(runnerId: number): void {
    const current = new Set(this.visibleRunners());
    if (current.has(runnerId)) {
      current.delete(runnerId);
    } else {
      current.add(runnerId);
    }
    this.visibleRunners.set(current);

    // Update chart series visibility
    if (this.chart) {
      const ids = this.runnerIds();
      const seriesIdx = ids.indexOf(runnerId) + 1; // +1 because series[0] is x
      if (seriesIdx > 0) {
        this.chart.setSeries(seriesIdx, { show: current.has(runnerId) });
      }
    }

    this.updateChartData();
  }

  isRunnerVisible(runnerId: number): boolean {
    const v = this.visibleRunners();
    return v.size === 0 || v.has(runnerId);
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
    const abs = Math.round(Math.abs(seconds));
    const mins = Math.floor(abs / 60);
    const secs = abs % 60;
    const sign = seconds < 0 ? '+' : '-';
    return `${sign}${mins}:${secs.toString().padStart(2, '0')}`;
  }
}
