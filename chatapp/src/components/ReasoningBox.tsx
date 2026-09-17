import React from 'react';
import { uiConfig } from '../config/uiConfig';

export interface ReasoningBoxProps {
  reasoning?: string | null;
}

export function ReasoningBox({ reasoning }: ReasoningBoxProps): React.JSX.Element | null {
  if (!reasoning) return null;

  return (
    <details className="reasoning-details">
      <summary className="reasoning-summary">
        <i className="fa-solid fa-brain" aria-hidden="true"></i>
        <span>{uiConfig.labels.reasoning}</span>
      </summary>
      <div className="reasoning-body">{reasoning}</div>
    </details>
  );
}
