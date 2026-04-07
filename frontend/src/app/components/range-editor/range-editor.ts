import { Component, computed, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';

/**
 * Two-input min/max editor for a single numeric search-range gene.
 *
 * Type-aware: float/float_log/int. Step size is chosen for each type.
 * Surfaces inline errors:
 *  - min > max
 *  - either bound outside the schema's allowed range (warning, soft)
 *
 * The editor never modifies the schema's spec — it produces an
 * HpRangeOverride object that is sent to POST /training-plans.
 */
@Component({
  selector: 'app-range-editor',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './range-editor.html',
  styleUrl: './range-editor.scss',
})
export class RangeEditor {
  readonly spec = input.required<HyperparamSchemaEntry>();
  readonly value = input.required<HpRangeOverride>();
  readonly valueChange = output<HpRangeOverride>();

  readonly stepHint = computed(() => {
    const t = this.spec().type;
    if (t === 'int') return '1';
    if (t === 'float_log') return 'any';
    return '0.001';
  });

  readonly minError = computed(() => {
    const v = this.value();
    if (v.min == null || v.max == null) return null;
    if (v.max < v.min) return 'max < min';
    return null;
  });

  readonly outOfRangeWarning = computed(() => {
    const s = this.spec();
    const v = this.value();
    if (s.min == null || s.max == null) return null;
    if (v.min != null && (v.min < s.min || v.min > s.max)) return `min outside spec [${s.min}, ${s.max}]`;
    if (v.max != null && (v.max < s.min || v.max > s.max)) return `max outside spec [${s.min}, ${s.max}]`;
    return null;
  });

  onMinChange(raw: string | number): void {
    const n = raw === '' || raw === null ? null : Number(raw);
    this.valueChange.emit({ ...this.value(), min: n });
  }

  onMaxChange(raw: string | number): void {
    const n = raw === '' || raw === null ? null : Number(raw);
    this.valueChange.emit({ ...this.value(), max: n });
  }
}
