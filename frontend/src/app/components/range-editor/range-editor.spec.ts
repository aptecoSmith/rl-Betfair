import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';
import { Component, signal } from '@angular/core';
import { RangeEditor } from './range-editor';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';

@Component({
  standalone: true,
  imports: [RangeEditor],
  template: `<app-range-editor [spec]="spec()" [value]="value()" (valueChange)="onChange($event)" />`,
})
class HostComponent {
  readonly spec = signal<HyperparamSchemaEntry>({
    name: 'gamma',
    type: 'float',
    min: 0.95,
    max: 0.999,
    choices: null,
    source_file: 'config.yaml#hyperparameters.search_ranges.gamma',
  });
  readonly value = signal<HpRangeOverride>({ type: 'float', min: 0.95, max: 0.999 });
  lastEmitted: HpRangeOverride | null = null;
  onChange(v: HpRangeOverride) { this.lastEmitted = v; this.value.set(v); }
}

describe('RangeEditor', () => {
  let fixture: ComponentFixture<HostComponent>;
  let host: HostComponent;

  beforeEach(() => {
    TestBed.configureTestingModule({ imports: [HostComponent] });
    fixture = TestBed.createComponent(HostComponent);
    host = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('renders the gene name as a label', () => {
    const el: HTMLElement = fixture.nativeElement;
    expect(el.textContent).toContain('gamma');
  });

  it('renders the gene type', () => {
    const el: HTMLElement = fixture.nativeElement;
    expect(el.textContent).toContain('float');
  });

  it('renders min/max inputs initialised to the value', () => {
    const minInput = fixture.nativeElement.querySelector('[data-testid="range-min-gamma"]') as HTMLInputElement;
    const maxInput = fixture.nativeElement.querySelector('[data-testid="range-max-gamma"]') as HTMLInputElement;
    expect(minInput.value).toBe('0.95');
    expect(maxInput.value).toBe('0.999');
  });

  it('emits a new value when min is edited', () => {
    const minInput = fixture.nativeElement.querySelector('[data-testid="range-min-gamma"]') as HTMLInputElement;
    minInput.value = '0.96';
    minInput.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(host.lastEmitted).toEqual({ type: 'float', min: 0.96, max: 0.999 });
  });

  it('shows a min>max error inline when bounds are inverted', () => {
    host.value.set({ type: 'float', min: 0.99, max: 0.95 });
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.inline-error')?.textContent).toContain('max < min');
  });

  it('shows out-of-range warning when min is below spec', () => {
    host.value.set({ type: 'float', min: 0.5, max: 0.999 });
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.inline-warning')?.textContent).toContain('outside spec');
  });

  it('shows the spec range hint', () => {
    const el: HTMLElement = fixture.nativeElement;
    expect(el.textContent).toContain('[0.95, 0.999]');
  });
});
