import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ErrorBoundary } from '../src/components/ErrorBoundary';

const FaultyChild: React.FC = () => {
  throw new Error('Simulation network crash test');
};

describe('ErrorBoundary Component', () => {
  it('catches render crashes and displays graceful recovery UI', () => {
    // Suppress expected console.error during boundary catch test
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <FaultyChild />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText(/Simulation network crash test/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Back to Home/i })).toBeInTheDocument();

    spy.mockRestore();
  });
});
