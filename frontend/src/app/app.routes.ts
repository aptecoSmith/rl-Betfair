import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'scoreboard', pathMatch: 'full' },
  { path: 'scoreboard', loadComponent: () => import('./scoreboard/scoreboard').then(m => m.Scoreboard) },
  { path: 'models/:id', loadComponent: () => import('./model-detail/model-detail').then(m => m.ModelDetail) },
  { path: 'training', loadComponent: () => import('./training-monitor/training-monitor').then(m => m.TrainingMonitor) },
];
