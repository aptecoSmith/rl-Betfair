import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect } from 'vitest';
import { Component, signal } from '@angular/core';
import { CoveragePanel } from './coverage-panel';
import { CoverageReport } from '../../models/training-plan.model';

@Component({
  standalone: true,
  imports: [CoveragePanel],
  template: `<app-coverage-panel [report]="report()" [biasedGenes]="biased()" />`,
})
class HostComponent {
  readonly report = signal<CoverageReport>({
    total_agents: 42,
    arch_counts: { ppo_lstm_v1: 20, ppo_time_lstm_v1: 22 },
    arch_undercovered: ['ppo_time_lstm_v1'],
    gene_coverage: [
      {
        name: 'gamma',
        type: 'float',
        bucket_edges: [0.95, 0.96, 0.97, 0.98, 0.99],
        bucket_counts: [10, 5, 0, 7],
        is_well_covered: false,
        total_samples: 22,
      },
      {
        name: 'learning_rate',
        type: 'float_log',
        bucket_edges: [1e-5, 1e-4, 5e-4],
        bucket_counts: [11, 11],
        is_well_covered: true,
        total_samples: 22,
      },
    ],
  });
  readonly biased = signal<string[]>(['gamma']);
}

describe('CoveragePanel', () => {
  let fixture: ComponentFixture<HostComponent>;

  function setup() {
    TestBed.configureTestingModule({ imports: [HostComponent] });
    fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
  }

  it('renders total_agents', () => {
    setup();
    expect(fixture.nativeElement.textContent).toContain('42');
  });

  it('renders arch_counts entries', () => {
    setup();
    const text = fixture.nativeElement.textContent;
    expect(text).toContain('ppo_lstm_v1');
    expect(text).toContain('ppo_time_lstm_v1');
    expect(text).toContain('20');
    expect(text).toContain('22');
  });

  it('flags undercovered architectures', () => {
    setup();
    expect(fixture.nativeElement.textContent).toContain('undercovered');
  });

  it('renders one row per gene', () => {
    setup();
    expect(fixture.nativeElement.querySelector('[data-testid="gene-cov-gamma"]')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('[data-testid="gene-cov-learning_rate"]')).toBeTruthy();
  });

  it('renders one bar per bucket per gene', () => {
    setup();
    const gammaRow = fixture.nativeElement.querySelector('[data-testid="gene-cov-gamma"]');
    const bars = gammaRow.querySelectorAll('rect');
    expect(bars.length).toBe(4);
  });

  it('marks well-covered vs poorly-covered genes', () => {
    setup();
    const text = fixture.nativeElement.textContent;
    expect(text).toContain('well-covered');
    expect(text).toContain('poorly-covered');
  });

  it('renders biased gene chips', () => {
    setup();
    const text = fixture.nativeElement.textContent;
    expect(text).toContain('Biased genes');
    expect(text).toContain('gamma');
  });
});
