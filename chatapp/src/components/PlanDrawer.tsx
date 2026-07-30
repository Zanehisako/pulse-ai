import React from 'react';
import { ExecutionStep } from '../types/index';

export interface PlanDrawerProps {
  steps?: ExecutionStep[];
}

export function PlanDrawer({ steps }: PlanDrawerProps): React.JSX.Element | null {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="plan-drawer">
      <div className="plan-header">
        <i className="fa-solid fa-list-check"></i> Execution Workflow ({steps.length} steps)
      </div>
      {steps.map((step, idx) => (
        <div
          key={idx}
          className={`plan-step ${step.status === 'executing' ? 'plan-step-executing' : step.status === 'success' ? 'plan-step-success' : ''}`}
        >
          <span className="step-badge">STEP {idx + 1}</span>
          <div className="step-details">
            <div>
              <strong>{step.tool}</strong>
              {step.status && (
                <span
                  style={{
                    marginLeft: '8px',
                    fontSize: '11px',
                    fontWeight: 600,
                    color: step.status === 'success' ? 'var(--accent-emerald)' : step.status === 'executing' ? 'var(--accent-amber)' : 'var(--text-dim)'
                  }}
                >
                  [{step.status.toUpperCase()}]
                </span>
              )}
            </div>
            {step.command && <div className="step-command">{step.command}</div>}
          </div>
        </div>
      ))}
    </div>
  );
}
