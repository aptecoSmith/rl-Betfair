import { Component, computed, input, output } from '@angular/core';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';
import { RangeEditor } from '../range-editor/range-editor';
import { ChoiceEditor } from '../choice-editor/choice-editor';

/**
 * Dispatcher: takes a HyperparamSchemaEntry and renders the right editor.
 *
 * This is the ONE place that maps gene type → widget. Pages and forms
 * never look at `spec.type` directly; they delegate to gene-editor.
 *
 * Special-cases `lstm_layer_norm` to render its 0/1 int_choice as a
 * boolean toggle (display only — the override still persists 0/1).
 */
@Component({
  selector: 'app-gene-editor',
  standalone: true,
  imports: [RangeEditor, ChoiceEditor],
  template: `
    @switch (kind()) {
      @case ('range') {
        <app-range-editor
          [spec]="spec()"
          [value]="value()"
          (valueChange)="valueChange.emit($event)"
        />
      }
      @case ('choice') {
        <app-choice-editor
          [spec]="spec()"
          [value]="value()"
          [displayAs]="displayAs()"
          (valueChange)="valueChange.emit($event)"
        />
      }
      @case ('arch-noop') {
        <div class="gene-noop" data-testid="arch-name-noop">
          <span class="gene-name">{{ spec().name }}</span>
          <span class="gene-noop-msg">edited via the Architectures multi-select above</span>
        </div>
      }
    }
  `,
  styles: [`
    .gene-noop {
      background: #1e1e2e;
      border: 1px dashed #333;
      border-radius: 6px;
      padding: 0.6rem 0.8rem;
      margin-bottom: 0.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .gene-name {
      font-family: monospace;
      color: #e0e0e0;
      font-weight: 600;
    }
    .gene-noop-msg {
      color: #888;
      font-size: 0.75rem;
    }
  `],
})
export class GeneEditor {
  readonly spec = input.required<HyperparamSchemaEntry>();
  readonly value = input.required<HpRangeOverride>();
  readonly valueChange = output<HpRangeOverride>();

  readonly kind = computed(() => {
    const s = this.spec();
    if (s.name === 'architecture_name') return 'arch-noop' as const;
    if (s.type === 'float' || s.type === 'float_log' || s.type === 'int') return 'range' as const;
    return 'choice' as const;
  });

  readonly displayAs = computed(() => {
    return this.spec().name === 'lstm_layer_norm' ? 'toggle' as const : 'chips' as const;
  });
}
