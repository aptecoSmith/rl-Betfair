import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'scoreboard', pathMatch: 'full' },
  { path: 'scoreboard', loadComponent: () => import('./scoreboard/scoreboard').then(m => m.Scoreboard) },
  { path: 'models/:id', loadComponent: () => import('./model-detail/model-detail').then(m => m.ModelDetail) },
  { path: 'training', loadComponent: () => import('./training-monitor/training-monitor').then(m => m.TrainingMonitor) },
  { path: 'admin', loadComponent: () => import('./admin/admin').then(m => m.Admin) },
  { path: 'replay', loadComponent: () => import('./race-replay/race-replay').then(m => m.RaceReplay) },
  { path: 'bets', loadComponent: () => import('./bet-explorer/bet-explorer').then(m => m.BetExplorer) },
];
