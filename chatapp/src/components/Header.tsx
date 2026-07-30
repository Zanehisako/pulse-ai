import React from 'react';
import { OrchestratorStatus } from '../types/index';

export interface HeaderProps {
  status: OrchestratorStatus;
  onClear?: () => void;
}

export function Header({ status, onClear }: HeaderProps): React.JSX.Element {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-logo">
          <i className="fa-solid fa-droplet"></i>
        </div>
        <div>
          <div className="brand-title">PulseAI Orchestrator</div>
          <div className="brand-subtitle">Dynamic Tool-Calling Agent (Vite App)</div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div
          className="status-badge"
          style={{
            background: status.ws_connected ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
            borderColor: status.ws_connected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)',
            color: status.ws_connected ? 'var(--accent-emerald)' : 'var(--accent-amber)'
          }}
        >
          <div
            className="status-pulse"
            style={{ background: status.ws_connected ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}
          ></div>
          <span>WS: {status.ws_connected ? 'CONNECTED' : 'DISCONNECTED'} | MODEL: {status.active_model}</span>
        </div>

        {onClear && (
          <button
            onClick={onClear}
            style={{
              background: 'transparent',
              border: '1px solid var(--border-color)',
              color: 'var(--text-muted)',
              borderRadius: '8px',
              padding: '6px 12px',
              cursor: 'pointer',
              fontSize: '12px',
              fontFamily: 'var(--font-mono)'
            }}
            title="Clear Chat History"
          >
            <i className="fa-solid fa-trash-can"></i> Clear
          </button>
        )}
      </div>
    </header>
  );
}
