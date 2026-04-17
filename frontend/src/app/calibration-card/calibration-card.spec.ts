import { TestBed, ComponentFixture } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { describe, it, expect, beforeEach } from 'vitest';
import {
  CalibrationCard,
  MACE_GREEN_BELOW,
  MACE_AMBER_BELOW_OR_EQUAL,
} from './calibration-card';
import {
  CalibrationStats,
  ReliabilityBucket,
  RiskScatterPoint,
} from '../models/model-detail.model';

/** Test host so the calibration signal input can be driven from the
 *  spec. Mirrors the pattern other Angular-signal-input components use
 *  in this codebase. */
@Component({
  standalone: true,
  imports: [CalibrationCard],
  template: `<app-calibration-card [calibration]="cal()"></app-calibration-card>`,
})
class Host {
  cal = signal<CalibrationStats | null | undefined>(null);
}

function bucket(overrides: Partial<ReliabilityBucket> = {}): ReliabilityBucket {
  return {
    bucket_label: '<25%',
    predicted_midpoint: 0.125,
    observed_rate: 0.1,
    count: 50,
    abs_calibration_error: 0.025,
    ...overrides,
  };
}

function point(overrides: Partial<RiskScatterPoint> = {}): RiskScatterPoint {
  return {
    predicted_pnl: 1.0,
    realised_pnl: 1.0,
    stddev_bucket: 'med',
    ...overrides,
  };
}

function makeStats(overrides: Partial<CalibrationStats> = {}): CalibrationStats {
  return {
    reliability_buckets: [
      bucket({ bucket_label: '<25%', predicted_midpoint: 0.125 }),
      bucket({ bucket_label: '25-50%', predicted_midpoint: 0.375, observed_rate: 0.4, abs_calibration_error: 0.025 }),
      bucket({ bucket_label: '50-75%', predicted_midpoint: 0.625, observed_rate: 0.6, abs_calibration_error: 0.025 }),
      bucket({ bucket_label: '>75%', predicted_midpoint: 0.875, observed_rate: 0.88, abs_calibration_error: 0.005 }),
    ],
    mace: 0.02,
    scatter: [point(), point({ predicted_pnl: 2.0, realised_pnl: 1.8 })],
    insufficient_data: false,
    ...overrides,
  };
}

describe('CalibrationCard', () => {
  let fixture: ComponentFixture<Host>;
  let host: Host;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [Host],
    });
    fixture = TestBed.createComponent(Host);
    host = fixture.componentInstance;
  });

  it('hides the whole card when calibration is null', () => {
    host.cal.set(null);
    fixture.detectChanges();
    const card = fixture.nativeElement.querySelector('[data-testid="calibration-card"]');
    expect(card).toBeNull();
  });

  it('hides the whole card when calibration is undefined', () => {
    host.cal.set(undefined);
    fixture.detectChanges();
    const card = fixture.nativeElement.querySelector('[data-testid="calibration-card"]');
    expect(card).toBeNull();
  });

  it('renders the empty state when insufficient_data is true', () => {
    host.cal.set(makeStats({ insufficient_data: true, mace: null }));
    fixture.detectChanges();
    const empty = fixture.nativeElement.querySelector('[data-testid="calibration-empty"]');
    expect(empty).not.toBeNull();
    // Diagrams are suppressed in the empty state.
    const bars = fixture.nativeElement.querySelectorAll('[data-testid="reliability-bar"]');
    expect(bars.length).toBe(0);
  });

  it('renders four reliability bars when data is sufficient', () => {
    host.cal.set(makeStats());
    fixture.detectChanges();
    const bars = fixture.nativeElement.querySelectorAll('[data-testid="reliability-bar"]');
    expect(bars.length).toBe(4);
  });

  it('reliability diagram includes a dashed diagonal reference', () => {
    host.cal.set(makeStats());
    fixture.detectChanges();
    const diag = fixture.nativeElement.querySelector('[data-testid="reliability-diagonal"]');
    expect(diag).not.toBeNull();
    expect(diag.getAttribute('stroke-dasharray')).toBe('4 4');
  });

  it('MACE badge renders to two decimals', () => {
    host.cal.set(makeStats({ mace: 0.0732 }));
    fixture.detectChanges();
    const badge = fixture.nativeElement.querySelector('[data-testid="mace-badge"]');
    expect(badge).not.toBeNull();
    expect(badge.textContent.trim()).toContain('MACE: 0.07');
  });

  it('MACE badge traffic light boundaries: 0.09 → green, 0.1 → amber, 0.2 → amber, 0.2001 → red', () => {
    const checkClass = (mace: number, expected: string) => {
      host.cal.set(makeStats({ mace }));
      fixture.detectChanges();
      const badge = fixture.nativeElement.querySelector('[data-testid="mace-badge"]');
      expect(badge.getAttribute('data-mace')).toBe(expected);
    };
    // Confirm the named constants stay where the spec expects.
    expect(MACE_GREEN_BELOW).toBe(0.1);
    expect(MACE_AMBER_BELOW_OR_EQUAL).toBe(0.2);

    checkClass(0.09, 'green');
    checkClass(0.1, 'amber');     // exact 0.1 → amber
    checkClass(0.2, 'amber');     // exact 0.2 → amber (upper inclusive)
    checkClass(0.2001, 'red');
  });

  it('renders N scatter points when calibration.scatter.length === N', () => {
    const scatter = [
      point({ predicted_pnl: 0.5, realised_pnl: 0.4, stddev_bucket: 'low' }),
      point({ predicted_pnl: 1.0, realised_pnl: 1.1, stddev_bucket: 'med' }),
      point({ predicted_pnl: 2.0, realised_pnl: 2.3, stddev_bucket: 'high' }),
      point({ predicted_pnl: 3.0, realised_pnl: 2.8, stddev_bucket: 'high' }),
      point({ predicted_pnl: 0.1, realised_pnl: 0.0, stddev_bucket: 'low' }),
    ];
    host.cal.set(makeStats({ scatter }));
    fixture.detectChanges();
    const points = fixture.nativeElement.querySelectorAll('[data-testid="scatter-point"]');
    expect(points.length).toBe(scatter.length);
  });

  it('scatter plot renders an empty-state string when scatter is empty', () => {
    host.cal.set(makeStats({ scatter: [] }));
    fixture.detectChanges();
    const empty = fixture.nativeElement.querySelector('[data-testid="scatter-empty"]');
    expect(empty).not.toBeNull();
  });

  it('scatter points colour by stddev_bucket (low=green, med=amber, high=red)', () => {
    const scatter = [
      point({ stddev_bucket: 'low' }),
      point({ stddev_bucket: 'med' }),
      point({ stddev_bucket: 'high' }),
    ];
    host.cal.set(makeStats({ scatter }));
    fixture.detectChanges();
    const points = fixture.nativeElement.querySelectorAll('[data-testid="scatter-point"]');
    const fills = Array.from(points).map((p: any) => p.getAttribute('fill'));
    expect(fills).toEqual(['#4caf50', '#ff9800', '#f44336']);
  });
});
