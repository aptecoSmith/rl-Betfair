import { Component, OnInit, inject, signal } from '@angular/core';
import { ApiService } from '../services/api.service';
import { HyperparamSchemaEntry } from '../models/training-plan.model';

/**
 * Schema Inspector page (Session 8).
 *
 * Read-only table of every gene currently in
 * `config.yaml` `hyperparameters.search_ranges`, with type, range/choices,
 * and `source_file` pointer. This is the "nothing dropped" visual check
 * that catches dead-gene bugs without grepping the codebase.
 */
@Component({
  selector: 'app-schema-inspector',
  standalone: true,
  templateUrl: './schema-inspector.html',
  styleUrl: './schema-inspector.scss',
})
export class SchemaInspector implements OnInit {
  private readonly api = inject(ApiService);

  readonly schema = signal<HyperparamSchemaEntry[]>([]);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);

  ngOnInit(): void {
    this.api.getHyperparameterSchema().subscribe({
      next: (entries) => {
        this.schema.set(entries);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.detail ?? err?.message ?? 'Failed to load schema');
        this.loading.set(false);
      },
    });
  }

  rangeOrChoices(entry: HyperparamSchemaEntry): string {
    if (entry.choices) return entry.choices.join(', ');
    if (entry.min != null && entry.max != null) return `[${entry.min}, ${entry.max}]`;
    return '';
  }
}
