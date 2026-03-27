import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { signal } from '@angular/core';
import { App } from './app';
import { routes } from './app.routes';
import { TrainingService } from './services/training.service';
import { SystemMetricsService } from './services/system-metrics.service';

describe('App', () => {
  beforeEach(async () => {
    const mockTraining = {
      status: signal({ running: false, phase: null, generation: null, process: null, item: null, detail: null, last_agent_score: null }),
      isRunning: signal(false),
      latestEvent: signal(null),
      rewardHistory: signal([]),
      lossHistory: signal([]),
      connect: () => {},
      clearHistory: () => {},
    };
    const mockSystemMetrics = {
      metrics: signal(null),
      error: signal(null),
      fetch: () => {},
    };

    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideRouter(routes),
        provideHttpClient(),
        { provide: TrainingService, useValue: mockTraining },
        { provide: SystemMetricsService, useValue: mockSystemMetrics },
      ],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should have a router outlet', () => {
    const fixture = TestBed.createComponent(App);
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('router-outlet')).toBeTruthy();
  });

  it('should have a header', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('app-header')).toBeTruthy();
  });
});
