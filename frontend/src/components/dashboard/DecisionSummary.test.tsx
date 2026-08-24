import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DecisionSummary } from './DecisionSummary';
import { DecisionState } from '../../types/decision';

describe('DecisionSummary', () => {
    const mockDecision: DecisionState = {
        timestamp: "2023-01-01T12:00:00Z",
        decision_mode: "FULL",
        availability: { drowsiness: true, statefarm: true, road: true },
        events: {},
        normal: false,
        observed_event_count: 2,
        severity: "CRITICAL",
        action: "DROWSINESS_WARNING"
    };

    it('renders mode and severity correctly', () => {
        render(<DecisionSummary decision={mockDecision} />);
        
        expect(screen.getByText('FULL')).toBeDefined();
        expect(screen.getByText('CRITICAL')).toBeDefined();
        expect(screen.getByText('2 Active')).toBeDefined();
        expect(screen.getByText('DROWSINESS WARNING')).toBeDefined(); // Replaced underscores
    });

    it('renders DEGRADED mode correctly', () => {
        const degradedDecision = { ...mockDecision, decision_mode: "DEGRADED" as const, severity: "DEGRADED" as const };
        render(<DecisionSummary decision={degradedDecision} />);
        
        expect(screen.getAllByText('DEGRADED').length).toBe(2);
    });
});
