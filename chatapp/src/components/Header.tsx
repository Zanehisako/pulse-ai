import React from 'react';
import { OrchestratorStatus } from '../types/index';
import { uiConfig } from '../config/uiConfig';
import { Sidebar } from './Sidebar';

export interface HeaderProps {
  status: OrchestratorStatus;
  onClear: () => void;
  isStreaming: boolean;
  theme: string;
  onThemeChange: (preference: string) => void;
  onSelectPreset: (query: string) => void;
}

export function Header({ status, onClear, isStreaming, theme, onThemeChange, onSelectPreset }: HeaderProps): React.JSX.Element {
  const isOnline = status.ws_connected || status.llm_ready;

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-logo" aria-hidden="true">
          <i className="fa-solid fa-droplet"></i>
        </div>
        <div>
          <div className="brand-title">{uiConfig.brand}</div>
          <div className="brand-subtitle">{uiConfig.subtitle}</div>
        </div>
      </div>
      <div className="header-actions">
        <details className="header-shortcuts">
          <summary className="header-shortcuts-summary">{uiConfig.labels.shortcuts}</summary>
          <div className="header-shortcuts-menu">
            <Sidebar onSelectPreset={onSelectPreset} disabled={isStreaming} />
          </div>
        </details>
        <button type="button" className="header-btn" onClick={onClear} disabled={isStreaming}>
          {uiConfig.labels.newChat}
        </button>
        <label className="theme-select-wrap">
          <span>{uiConfig.labels.theme}</span>
          <select className="theme-select" value={theme} onChange={e => onThemeChange(e.target.value)}>
            {uiConfig.theme.options.map(option => (
              <option key={option.id} value={option.id}>{option.label}</option>
            ))}
          </select>
        </label>
        <div className={`status-badge ${isOnline ? 'status-online' : 'status-offline'}`} title={status.active_model} role="status">
          <span className="status-dot" aria-hidden="true"></span>
          <span>{isOnline ? uiConfig.labels.ready : uiConfig.labels.offline}</span>
        </div>
      </div>
    </header>
  );
}
