import { Component, computed, input, output } from '@angular/core';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';
import { RangeEditor } from '../range-editor/range-editor';

/**
 * Composite editor for the `early_pick_bonus_min` / `early_pick_bonus_max`
 * gene pair.
 *
 * Wraps two range-editors and enforces the cross-gene invariant
 * `max.max >= min.min` BEFORE save (Session 3 belt-and-braces). Server-side
 * validation in `population_manager._repair_reward_gene_pairs` is the
 * authoritative check; this widget just shows the user the problem inline.
 */
@Component({
  selector: 'app-early-pick-validator',
  standalone: true,
  imports: [RangeEditor],
  template: `
    <div class="early-pick-validator" [class.has-error]="invariantError()">
      <div class="title">Early-pick bonus interval</div>
      <app-range-editor
        [spec]="minSpec()"
        [value]="minValue()"
        (valueChange)="onMinChange($event)"
      />
      <app-range-editor
        [spec]="maxSpec()"
        [value]="maxValue()"
        (valueChange)="onMaxChange($event)"
      />
      @if (invariantError()) {
        <div class="invariant-error" data-testid="early-pick-invariant-error">
          {{ invariantError() }}
        </div>
      }
    </div>
  `,
  styles: [`
    .early-pick-validator {
      border: 1px solid #2a2a3a;
      border-radius: 6px;
      padding: 0.6rem 0.8rem;
      margin-bottom: 0.5rem;
      background: #181822;

      &.has-error {
        border-color: #e57373;
      }
    }
    .title {
      color: #aaa;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 0.4rem;
    }
    .invariant-error {
      color: #e57373;
      font-size: 0.75rem;
      margin-top: 0.3rem;
    }
  `],
})
export class EarlyPickValidator {
  readonly minSpec = input.required<HyperparamSchemaEntry>();
  readonly maxSpec = input.required<HyperparamSchemaEntry>();
  readonly minValue = input.required<HpRangeOverride>();
  readonly maxValue = input.required<HpRangeOverride>();
  readonly minValueChange = output<HpRangeOverride>();
  readonly maxValueChange = output<HpRangeOverride>();

  readonly invariantError = computed(() => {
    const lo = this.minValue();
    const hi = this.maxValue();
    // The interval [min.min, max.max] must satisfy max.max >= min.min.
    // We also enforce min.max >= min.min and hi.max >= hi.min — but those
    // are caught by each child range-editor. The cross-gene check is the
    // unique value-add of this composite.
    if (lo.min != null && hi.max != null && hi.max < lo.min) {
      return `early_pick_bonus_max.max (${hi.max}) < early_pick_bonus_min.min (${lo.min})`;
    }
    if (lo.min != null && hi.min != null && hi.min < lo.min) {
      return `early_pick_bonus_max.min (${hi.min}) < early_pick_bonus_min.min (${lo.min})`;
    }
    return null;
  });

  onMinChange(v: HpRangeOverride): void {
    this.minValueChange.emit(v);
  }
  onMaxChange(v: HpRangeOverride): void {
    this.maxValueChange.emit(v);
  }
}
