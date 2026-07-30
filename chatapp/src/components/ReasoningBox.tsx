import React, { useState } from 'react';

export interface ReasoningBoxProps {
  reasoning?: string | null;
}

export function ReasoningBox({ reasoning }: ReasoningBoxProps): React.JSX.Element | null {
  const [collapsed, setCollapsed] = useState<boolean>(false);

  if (!reasoning) return null;

  return (
    <div className="reasoning-box">
      <div
        className="reasoning-header"
        onClick={() => setCollapsed(prev => !prev)}
      >
        <span>
          <i className="fa-solid fa-brain"></i> Orchestrator Reasoning
        </span>
        <i className={`fa-solid ${collapsed ? 'fa-chevron-down' : 'fa-chevron-up'}`}></i>
      </div>
      {!collapsed && <div>{reasoning}</div>}
    </div>
  );
}
