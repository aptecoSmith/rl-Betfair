import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';
import { Component, signal } from '@angular/core';
import { ChoiceEditor } from './choice-editor';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';

@Component({
  standalone: true,
  imports: [ChoiceEditor],
  template: `<app-choice-editor [spec]="spec()" [value]="value()" [displayAs]="displayAs()" (valueChange)="onChange($event)" />`,
})
class HostComponent {
  readonly spec = signal<HyperparamSchemaEntry>({
    name: 'lstm_num_layers',
    type: 'int_choice',
    min: null,
    max: null,
    choices: [1, 2],
    source_file: 'config.yaml#hyperparameters.search_ranges.lstm_num_layers',
  });
  readonly value = signal<HpRangeOverride>({ type: 'int_choice', choices: [1, 2] });
  readonly displayAs = signal<'chips' | 'toggle'>('chips');
  lastEmitted: HpRangeOverride | null = null;
  onChange(v: HpRangeOverride) { this.lastEmitted = v; this.value.set(v); }
}

describe('ChoiceEditor', () => {
  let fixture: ComponentFixture<HostComponent>;
  let host: HostComponent;

  beforeEach(() => {
    TestBed.configureTestingModule({ imports: [HostComponent] });
    fixture = TestBed.createComponent(HostComponent);
    host = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('renders one chip per choice', () => {
    const chips = fixture.nativeElement.querySelectorAll('.chip');
    expect(chips.length).toBe(2);
  });

  it('marks selected chips active', () => {
    const chip = fixture.nativeElement.querySelector('[data-testid="chip-lstm_num_layers-1"]');
    expect(chip.classList.contains('active')).toBe(true);
  });

  it('toggling a chip removes it from the selection', () => {
    const chip = fixture.nativeElement.querySelector('[data-testid="chip-lstm_num_layers-2"]') as HTMLElement;
    chip.click();
    fixture.detectChanges();
    expect(host.lastEmitted?.choices).toEqual([1]);
  });

  it('shows an error when no choices are selected', () => {
    host.value.set({ type: 'int_choice', choices: [] });
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('.inline-error')?.textContent).toContain('at least one');
  });

  it('renders as a boolean toggle for lstm_layer_norm when displayAs=toggle', () => {
    host.spec.set({
      name: 'lstm_layer_norm',
      type: 'int_choice',
      min: null,
      max: null,
      choices: [0, 1],
      source_file: 'config.yaml#hyperparameters.search_ranges.lstm_layer_norm',
    });
    host.value.set({ type: 'int_choice', choices: [0] });
    host.displayAs.set('toggle');
    fixture.detectChanges();
    const checkbox = fixture.nativeElement.querySelector('[data-testid="toggle-lstm_layer_norm"]') as HTMLInputElement;
    expect(checkbox).toBeTruthy();
    expect(checkbox.checked).toBe(false);
  });

  it('toggle persists 0 and 1 (not boolean)', () => {
    host.spec.set({
      name: 'lstm_layer_norm',
      type: 'int_choice',
      min: null,
      max: null,
      choices: [0, 1],
      source_file: 'config.yaml#hyperparameters.search_ranges.lstm_layer_norm',
    });
    host.value.set({ type: 'int_choice', choices: [0] });
    host.displayAs.set('toggle');
    fixture.detectChanges();
    const checkbox = fixture.nativeElement.querySelector('[data-testid="toggle-lstm_layer_norm"]') as HTMLInputElement;
    checkbox.checked = true;
    checkbox.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(host.lastEmitted?.choices).toEqual([1]);
  });
});
