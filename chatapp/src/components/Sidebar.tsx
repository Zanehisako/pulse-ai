import React from 'react';
import { uiConfig } from '../config/uiConfig';

export interface SidebarProps {
  onSelectPreset: (query: string) => void;
  disabled?: boolean;
}

export function Sidebar({ onSelectPreset, disabled }: SidebarProps): React.JSX.Element {
  return (
    <div className="suggestion-grid" aria-label={uiConfig.labels.shortcuts}>
      {uiConfig.suggestions.map(suggestion => (
        <button
          key={suggestion.id}
          type="button"
          className="suggestion-card"
          onClick={() => onSelectPreset(suggestion.query)}
          disabled={disabled}
        >
          <span className="suggestion-icon" aria-hidden="true">
            <i className={suggestion.icon}></i>
          </span>
          <span className="suggestion-body">
            <span className="suggestion-title">{suggestion.title}</span>
            <span className="suggestion-desc">{suggestion.desc}</span>
          </span>
        </button>
      ))}
    </div>
  );
}
