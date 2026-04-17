import { Component, computed, input } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import {
  CalibrationStats,
  ReliabilityBucket,
  RiskScatterPoint,
} from '../models/model-detail.model';

/**
 * MACE traffic-light thresholds. Session 05 rule: green < 0.1,
 * amber 0.1-0.2, red > 0.2. The 0.1 / 0.2 boundaries are inclusive
 * on the ``amber`` / ``red`` side respectively so a value of exactly
 * 0.1 renders amber and exactly 0.2 renders red (pinned by
 * ``test_mace_threshold_boundaries`` on the frontend side).
 */
export const MACE_GREEN_BELOW = 0.1;
export const MACE_AMBER_BELOW_OR_EQUAL = 0.2;

/**
 * Reliability-bar colour thresholds. Session 05 rule: green when the
 * bar is within 5 % of the diagonal, amber within 5-15 %, red above
 * 15 %. Same boundary convention as MACE.
 */
const RELIABILITY_GREEN_BELOW = 0.05;
const RELIABILITY_AMBER_BELOW_OR_EQUAL = 0.15;

/** Inline SVG canvas sizes. Kept local to the component so the two
 *  diagrams can use a consistent coordinate space. */
const RELIABILITY_SIZE = 200;
const SCATTER_SIZE = 220;
const SCATTER_PADDING = 20;

interface ReliabilityBarLayout {
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  colour: string;
  observed: number;
  count: number;
  predictedMidpoint: number;
}

interface ScatterPointLayout {
  cx: number;
  cy: number;
  colour: string;
  predicted: number;
  realised: number;
  stddevBucket: string;
}

interface ScatterLayout {
  points: ScatterPointLayout[];
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  diagonalX1: number;
  diagonalY1: number;
  diagonalX2: number;
  diagonalY2: number;
}

const BUCKET_COLOUR = {
  low: '#4caf50',
  med: '#ff9800',
  high: '#f44336',
} as const;

@Component({
  selector: 'app-calibration-card',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  templateUrl: './calibration-card.html',
  styleUrl: './calibration-card.scss',
})
export class CalibrationCard {
  /** ``null`` / ``undefined`` hides the whole card — directional runs
   *  and pre-Session-02 evaluations have no calibration payload. */
  readonly calibration = input<CalibrationStats | null | undefined>(null);

  readonly reliabilityCanvas = RELIABILITY_SIZE;
  readonly scatterCanvas = SCATTER_SIZE;

  /** MACE formatted to two decimals. Empty string if ``mace`` is
   *  null — the badge row is hidden in that case. */
  readonly maceDisplay = computed(() => {
    const cal = this.calibration();
    if (!cal || cal.mace == null) return '';
    return cal.mace.toFixed(2);
  });

  /** Traffic-light class for the MACE badge. ``green`` < 0.1; ``amber``
   *  [0.1, 0.2]; ``red`` > 0.2. Exact 0.1 → amber, exact 0.2 → red. */
  readonly maceClass = computed<'green' | 'amber' | 'red' | 'none'>(() => {
    const cal = this.calibration();
    if (!cal || cal.mace == null) return 'none';
    const m = cal.mace;
    if (m < MACE_GREEN_BELOW) return 'green';
    if (m <= MACE_AMBER_BELOW_OR_EQUAL) return 'amber';
    return 'red';
  });

  /** Pre-computed reliability-diagram bar layouts. Returns an empty
   *  array when there's no calibration payload. */
  readonly reliabilityBars = computed<ReliabilityBarLayout[]>(() => {
    const cal = this.calibration();
    if (!cal) return [];
    return cal.reliability_buckets.map((b, idx) =>
      this._layoutBar(b, idx, cal.reliability_buckets.length),
    );
  });

  readonly scatter = computed<ScatterLayout | null>(() => {
    const cal = this.calibration();
    if (!cal || cal.scatter.length === 0) return null;
    return this._layoutScatter(cal.scatter);
  });

  private _layoutBar(
    bucket: ReliabilityBucket,
    idx: number,
    total: number,
  ): ReliabilityBarLayout {
    const plot = RELIABILITY_SIZE - SCATTER_PADDING * 2;
    const barWidth = plot / total * 0.7;
    const gap = (plot / total) * 0.3;
    const x = SCATTER_PADDING + idx * (plot / total) + gap / 2;
    const height = bucket.observed_rate * plot;
    const y = SCATTER_PADDING + (plot - height);
    const err = bucket.abs_calibration_error;
    let colour = '#f44336';
    if (err < RELIABILITY_GREEN_BELOW) colour = '#4caf50';
    else if (err <= RELIABILITY_AMBER_BELOW_OR_EQUAL) colour = '#ff9800';
    return {
      label: bucket.bucket_label,
      x,
      y,
      width: barWidth,
      height,
      colour,
      observed: bucket.observed_rate,
      count: bucket.count,
      predictedMidpoint: bucket.predicted_midpoint,
    };
  }

  private _layoutScatter(points: RiskScatterPoint[]): ScatterLayout {
    const xs = points.map(p => p.predicted_pnl);
    const ys = points.map(p => p.realised_pnl);
    let lo = Math.min(...xs, ...ys);
    let hi = Math.max(...xs, ...ys);
    if (lo === hi) {
      // Avoid a zero-range axis when every point sits on the same
      // value — pad the range so the diagonal and points remain
      // visible rather than collapsing to the corner.
      lo -= 1;
      hi += 1;
    }
    const span = hi - lo;
    const plot = SCATTER_SIZE - SCATTER_PADDING * 2;
    const toX = (v: number) =>
      SCATTER_PADDING + ((v - lo) / span) * plot;
    const toY = (v: number) =>
      SCATTER_PADDING + plot - ((v - lo) / span) * plot;
    const layout: ScatterPointLayout[] = points.map(p => ({
      cx: toX(p.predicted_pnl),
      cy: toY(p.realised_pnl),
      colour: BUCKET_COLOUR[p.stddev_bucket],
      predicted: p.predicted_pnl,
      realised: p.realised_pnl,
      stddevBucket: p.stddev_bucket,
    }));
    return {
      points: layout,
      xMin: lo,
      xMax: hi,
      yMin: lo,
      yMax: hi,
      diagonalX1: toX(lo),
      diagonalY1: toY(lo),
      diagonalX2: toX(hi),
      diagonalY2: toY(hi),
    };
  }
}
