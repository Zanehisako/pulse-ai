import React from 'react';
import { ExecutionStep } from '../types/index';
import { uiConfig } from '../config/uiConfig';

export interface PlanDrawerProps {
  steps?: ExecutionStep[];
}

export function PlanDrawer({ steps }: PlanDrawerProps): React.JSX.Element | null {
  if (!steps || steps.length === 0) return null;

  return (
    <details className="plan-details">
      <summary className="plan-summary">
        <i className="fa-solid fa-list-check" aria-hidden="true"></i>
        <span>{uiConfig.labels.execution} ({steps.length})</span>
      </summary>
      <div className="plan-steps">
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
                  : 'plan-step-pending'
              }`}
            >
              <span className="step-badge">{idx + 1}</span>
              <div className="step-details">
                <div className="step-tool-row">
                  <strong>{step.tool}</strong>
                  {step.status && (
                    <span className={`step-status step-status-${step.status}`}>
                      [{step.status.toUpperCase()}]
                    </span>
                  )}
                </div>
                {step.command && <div className="step-command">{step.command}</div>}
                {step.error && (
                  <div className="step-error">
                    <i className="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>
                    <strong>{uiConfig.labels.execution}</strong> {step.error}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </details>
  );
}
