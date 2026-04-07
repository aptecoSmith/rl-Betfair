import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';
import { Component, signal } from '@angular/core';
import { EarlyPickValidator } from './early-pick-validator';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';

@Component({
  standalone: true,
  imports: [EarlyPickValidator],
  template: `
    <app-early-pick-validator
      [minSpec]="minSpec()"
      [maxSpec]="maxSpec()"
      [minValue]="minValue()"
      [maxValue]="maxValue()"
      (minValueChange)="minValue.set($event)"
      (maxValueChange)="maxValue.set($event)"
    />
  `,
})
class HostComponent {
  readonly minSpec = signal<HyperparamSchemaEntry>({
    name: 'early_pick_bonus_min', type: 'float', min: 1.0, max: 1.3, choices: null, source_file: '',
  });
  readonly maxSpec = signal<HyperparamSchemaEntry>({
    name: 'early_pick_bonus_max', type: 'float', min: 1.1, max: 1.8, choices: null, source_file: '',
  });
  readonly minValue = signal<HpRangeOverride>({ type: 'float', min: 1.0, max: 1.3 });
  readonly maxValue = signal<HpRangeOverride>({ type: 'float', min: 1.1, max: 1.8 });
}

describe('EarlyPickValidator', () => {
  let fixture: ComponentFixture<HostComponent>;
  let host: HostComponent;

  beforeEach(() => {
    TestBed.configureTestingModule({ imports: [HostComponent] });
    fixture = TestBed.createComponent(HostComponent);
    host = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('renders both child range-editors', () => {
    const editors = fixture.nativeElement.querySelectorAll('app-range-editor');
    expect(editors.length).toBe(2);
  });

  it('does not show the invariant error in the happy case', () => {
    expect(fixture.nativeElement.querySelector('[data-testid="early-pick-invariant-error"]')).toBeFalsy();
  });

  it('shows an inline error when max.max < min.min', () => {
    host.minValue.set({ type: 'float', min: 1.5, max: 1.6 });
    host.maxValue.set({ type: 'float', min: 1.1, max: 1.2 });
    fixture.detectChanges();
    const err = fixture.nativeElement.querySelector('[data-testid="early-pick-invariant-error"]');
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('< early_pick_bonus_min.min');
  });

  it('shows the cross-gene error when min.min > max.min', () => {
    host.minValue.set({ type: 'float', min: 1.5, max: 1.6 });
    host.maxValue.set({ type: 'float', min: 1.1, max: 2.0 });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="early-pick-invariant-error"]')).toBeTruthy();
  });
});
