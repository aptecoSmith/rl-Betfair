import { Component, computed, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HpRangeOverride, HyperparamSchemaEntry } from '../../models/training-plan.model';

/**
 * Editor for `int_choice` / `str_choice` genes.
 *
 * Renders as a chip-group that lets the user toggle which choices are
 * allowed for a given training plan. The output is an HpRangeOverride
 * with a `choices` array that is a (non-empty) subset of the spec's
 * choices.
 *
 * Special case: when `displayAs === 'toggle'`, the choices [0, 1] are
 * rendered as a single true/false toggle (used for `lstm_layer_norm`,
 * which is stored as 0/1 int_choice but is semantically boolean).
 * The override still persists 0/1 — the conversion is purely visual.
 */
@Component({
  selector: 'app-choice-editor',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './choice-editor.html',
  styleUrl: './choice-editor.scss',
})
export class ChoiceEditor {
  readonly spec = input.required<HyperparamSchemaEntry>();
  readonly value = input.required<HpRangeOverride>();
  readonly valueChange = output<HpRangeOverride>();
  readonly displayAs = input<'chips' | 'toggle'>('chips');

  readonly availableChoices = computed(() => this.spec().choices ?? []);

  readonly selected = computed(() => {
    const v = this.value();
    return v.choices ?? this.availableChoices();
  });

  readonly isToggleMode = computed(() => {
    if (this.displayAs() !== 'toggle') return false;
    const choices = this.availableChoices();
    return choices.length === 2 && choices.includes(0 as any) && choices.includes(1 as any);
  });

  /** True if the toggle is "on" — i.e. only [1] is selected. */
  readonly toggleOn = computed(() => {
    const sel = this.selected();
    return sel.length === 1 && sel[0] === 1;
  });

  readonly emptyError = computed(() => {
    return this.selected().length === 0 ? 'at least one choice must be selected' : null;
  });

  isSelected(choice: number | string): boolean {
    return this.selected().includes(choice);
  }

  toggleChoice(choice: number | string): void {
    const sel = [...this.selected()];
    const idx = sel.indexOf(choice);
    if (idx >= 0) {
      sel.splice(idx, 1);
    } else {
      sel.push(choice);
    }
    // Re-order to match spec order so equality comparisons stay stable.
    const ordered = this.availableChoices().filter(c => sel.includes(c));
    this.valueChange.emit({ ...this.value(), choices: ordered });
  }

  onToggleChange(on: boolean): void {
    // Persist as 0/1 int_choice — see class-level docstring.
    this.valueChange.emit({ ...this.value(), choices: on ? [1] : [0] });
  }
}
