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
      {steps.map((step, idx) => {
        const isFailed = step.status === 'failed' || Boolean(step.error);
        const isExecuting = step.status === 'executing';
        const isSuccess = step.status === 'success';

        return (
          <div
            key={idx}
            className={`plan-step ${
              isFailed
                ? 'plan-step-failed'
                : isExecuting
                ? 'plan-step-executing'
                : isSuccess
                ? 'plan-step-success'
                : ''
            }`}
            style={{
              borderColor: isFailed
                ? 'var(--accent-crimson)'
                : isExecuting
                ? 'var(--accent-amber)'
                : isSuccess
                ? 'var(--accent-emerald)'
                : 'var(--border-color)',
              background: isFailed
                ? 'rgba(239, 68, 68, 0.08)'
                : isExecuting
                ? 'rgba(245, 158, 11, 0.08)'
                : isSuccess
                ? 'rgba(16, 185, 129, 0.08)'
                : 'rgba(7, 9, 19, 0.6)'
            }}
          >
            <span
              className="step-badge"
              style={{
                background: isFailed ? 'var(--accent-crimson)' : 'var(--accent-cyan)',
                color: isFailed ? '#ffffff' : '#000000'
              }}
            >
              STEP {idx + 1}
            </span>
            <div className="step-details" style={{ width: '100%' }}>
              <div>
                <strong>{step.tool}</strong>
                {step.status && (
                  <span
                    style={{
                      marginLeft: '8px',
                      fontSize: '11px',
                      fontWeight: 600,
                      color: isFailed
                        ? 'var(--accent-crimson)'
                        : isSuccess
                        ? 'var(--accent-emerald)'
                        : isExecuting
                        ? 'var(--accent-amber)'
                        : 'var(--text-dim)'
                    }}
                  >
                    [{step.status.toUpperCase()}]
                  </span>
                )}
              </div>
              {step.command && <div className="step-command">{step.command}</div>}

              {/* Explicit Failure Error Display */}
              {step.error && (
                <div
                  style={{
                    color: 'var(--accent-crimson)',
                    background: 'rgba(239, 68, 68, 0.12)',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    marginTop: '8px',
                    fontFamily: 'var(--font-mono)',
                    lineHeight: '1.4'
                  }}
                >
                  <i className="fa-solid fa-triangle-exclamation" style={{ marginRight: '6px' }}></i>
                  <strong>Execution Failure:</strong> {step.error}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
