import { TestBed } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';
import { SelectionStateService } from './selection-state.service';

describe('SelectionStateService', () => {
  let service: SelectionStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SelectionStateService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  // ── Default values ──

  it('should initialise selectedModelId as null', () => {
    expect(service.selectedModelId()).toBeNull();
  });

  it('should initialise betExplorerFilters with empty strings', () => {
    const filters = service.betExplorerFilters();
    expect(filters.date).toBe('');
    expect(filters.race).toBe('');
    expect(filters.runner).toBe('');
    expect(filters.action).toBe('');
    expect(filters.outcome).toBe('');
  });

  it('should initialise replayDate as null', () => {
    expect(service.replayDate()).toBeNull();
  });

  it('should initialise replayRaceId as null', () => {
    expect(service.replayRaceId()).toBeNull();
  });

  it('should initialise trainingFormValues with defaults', () => {
    const form = service.trainingFormValues();
    expect(form.generations).toBe(3);
    expect(form.epochs).toBe(3);
    expect(form.populationSize).toBeNull();
  });

  // ── Persistence (signals survive across reads) ──

  it('should persist selectedModelId across reads', () => {
    service.selectedModelId.set('model-abc');
    expect(service.selectedModelId()).toBe('model-abc');
  });

  it('should persist betExplorerFilters across reads', () => {
    service.betExplorerFilters.set({
      date: '2026-03-01',
      race: 'race-1',
      runner: 'Test Horse',
      action: 'back',
      outcome: 'won',
    });
    const filters = service.betExplorerFilters();
    expect(filters.date).toBe('2026-03-01');
    expect(filters.race).toBe('race-1');
    expect(filters.runner).toBe('Test Horse');
    expect(filters.action).toBe('back');
    expect(filters.outcome).toBe('won');
  });

  it('should persist replay selections across reads', () => {
    service.replayDate.set('2026-03-01');
    service.replayRaceId.set('race-1');
    expect(service.replayDate()).toBe('2026-03-01');
    expect(service.replayRaceId()).toBe('race-1');
  });

  it('should persist trainingFormValues across reads', () => {
    service.trainingFormValues.set({
      generations: 5,
      epochs: 10,
      populationSize: 100,
    });
    const form = service.trainingFormValues();
    expect(form.generations).toBe(5);
    expect(form.epochs).toBe(10);
    expect(form.populationSize).toBe(100);
  });

  // ── Global model ID is shared ──

  it('should share selectedModelId across independent reads', () => {
    service.selectedModelId.set('shared-model');
    // Simulates two different pages reading the same signal
    const readFromPageA = service.selectedModelId();
    const readFromPageB = service.selectedModelId();
    expect(readFromPageA).toBe('shared-model');
    expect(readFromPageB).toBe('shared-model');
  });

  // ── Singleton behaviour ──

  it('should be a singleton (providedIn root)', () => {
    const service2 = TestBed.inject(SelectionStateService);
    expect(service).toBe(service2);
  });

  // ── Independence of page-specific state ──

  it('should keep betExplorerFilters independent from replay state', () => {
    service.betExplorerFilters.set({
      date: '2026-03-01',
      race: 'race-1',
      runner: '',
      action: '',
      outcome: '',
    });
    service.replayDate.set('2026-03-15');
    service.replayRaceId.set('race-99');

    expect(service.betExplorerFilters().date).toBe('2026-03-01');
    expect(service.replayDate()).toBe('2026-03-15');
  });
});
