import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect } from 'vitest';
import { Component, signal } from '@angular/core';
import { GeneEditor } from './gene-editor';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';

@Component({
  standalone: true,
  imports: [GeneEditor],
  template: `<app-gene-editor [spec]="spec()" [value]="value()" (valueChange)="onChange($event)" />`,
})
class HostComponent {
  readonly spec = signal<HyperparamSchemaEntry>({
    name: 'gamma',
    type: 'float',
    min: 0.95,
    max: 0.999,
    choices: null,
    source_file: '',
  });
  readonly value = signal<HpRangeOverride>({ type: 'float', min: 0.95, max: 0.999 });
  emitted: HpRangeOverride | null = null;
  onChange(v: HpRangeOverride) { this.emitted = v; }
}

function setup() {
  TestBed.configureTestingModule({ imports: [HostComponent] });
  const fixture = TestBed.createComponent(HostComponent);
  fixture.detectChanges();
  return { fixture, host: fixture.componentInstance };
}

describe('GeneEditor (dispatcher)', () => {
  it('renders a range-editor for float genes', () => {
    const { fixture } = setup();
    expect(fixture.nativeElement.querySelector('app-range-editor')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('app-choice-editor')).toBeFalsy();
  });

  it('renders a range-editor for float_log genes', () => {
    const { fixture, host } = setup();
    host.spec.set({ name: 'learning_rate', type: 'float_log', min: 1e-5, max: 5e-4, choices: null, source_file: '' });
    host.value.set({ type: 'float_log', min: 1e-5, max: 5e-4 });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-range-editor')).toBeTruthy();
  });

  it('renders a range-editor for int genes', () => {
    const { fixture, host } = setup();
    host.spec.set({ name: 'mlp_layers', type: 'int', min: 1, max: 3, choices: null, source_file: '' });
    host.value.set({ type: 'int', min: 1, max: 3 });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-range-editor')).toBeTruthy();
  });

  it('renders a choice-editor for int_choice genes', () => {
    const { fixture, host } = setup();
    host.spec.set({ name: 'lstm_num_layers', type: 'int_choice', min: null, max: null, choices: [1, 2], source_file: '' });
    host.value.set({ type: 'int_choice', choices: [1, 2] });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-choice-editor')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('app-range-editor')).toBeFalsy();
  });

  it('renders a choice-editor for str_choice genes', () => {
    const { fixture, host } = setup();
    host.spec.set({ name: 'foo', type: 'str_choice', min: null, max: null, choices: ['a', 'b'], source_file: '' });
    host.value.set({ type: 'str_choice', choices: ['a', 'b'] });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-choice-editor')).toBeTruthy();
  });

  it('lstm_layer_norm uses toggle display', () => {
    const { fixture, host } = setup();
    host.spec.set({ name: 'lstm_layer_norm', type: 'int_choice', min: null, max: null, choices: [0, 1], source_file: '' });
    host.value.set({ type: 'int_choice', choices: [1] });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="toggle-lstm_layer_norm"]')).toBeTruthy();
  });

  it('architecture_name renders a no-op message (edited via multi-select)', () => {
    const { fixture, host } = setup();
    host.spec.set({
      name: 'architecture_name', type: 'str_choice',
      min: null, max: null, choices: ['ppo_lstm_v1'], source_file: '',
    });
    host.value.set({ type: 'str_choice', choices: ['ppo_lstm_v1'] });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[data-testid="arch-name-noop"]')).toBeTruthy();
  });
});
