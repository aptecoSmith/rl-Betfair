import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { inject } from '@angular/core';

@Component({
  selector: 'app-model-detail',
  standalone: true,
  template: `
    <div class="model-detail-container">
      <h1>Model Detail</h1>
      <p>Model ID: {{ modelId }}</p>
      <p><em>Full implementation in Session 3.5</em></p>
    </div>
  `,
  styles: [`.model-detail-container { max-width: 1200px; margin: 0 auto; padding: 24px; }`],
})
export class ModelDetail {
  private readonly route = inject(ActivatedRoute);
  readonly modelId = this.route.snapshot.paramMap.get('id') ?? '';
}
