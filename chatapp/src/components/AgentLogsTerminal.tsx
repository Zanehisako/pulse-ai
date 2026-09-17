import React, { useRef } from 'react';
import { LogEntry } from '../types/index';
import { uiConfig } from '../config/uiConfig';

export interface AgentLogsTerminalProps {
  logs: LogEntry[];
  onClearLogs?: () => void;
  isStreaming?: boolean;
}

export function AgentLogsTerminal({ logs, onClearLogs, isStreaming }: AgentLogsTerminalProps): React.JSX.Element {
  const terminalEndRef = useRef<HTMLDivElement | null>(null);

  return (
    <details
      className="activity-log"
      onToggle={(e: React.SyntheticEvent<HTMLDetailsElement>) => {
        if ((e.target as HTMLDetailsElement).open) {
          terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
      }}
    >
      <summary className="activity-log-summary">
        <span className="activity-log-title">
          <i className="fa-solid fa-terminal" aria-hidden="true"></i>
          {uiConfig.labels.activity}
        </span>
        <span className="activity-log-actions">
          {onClearLogs && (
            <button
              type="button"
              className="activity-log-clear"
              onClick={e => {
                e.stopPropagation();
                onClearLogs();
              }}
              disabled={isStreaming}
            >
              {uiConfig.labels.clearLogs}
            </button>
          )}
          <span className="activity-log-count">{logs.length}</span>
        </span>
      </summary>
      <div className="activity-log-body">
        {logs.length === 0 ? (
          <div className="activity-log-empty">{uiConfig.labels.emptyLogs}</div>
        ) : (
          logs.map((log, idx) => (
            <div key={idx} className={`activity-log-line log-level-${log.level.toLowerCase()}`}>
              <span className="activity-log-timestamp">[{log.timestamp}]</span>
              <span className="activity-log-level">[{log.level}]</span>
              <span className="activity-log-logger">{log.logger}:</span>
              <span className="activity-log-message">{log.message}</span>
            </div>
          ))
        )}
        <div ref={terminalEndRef} />
      </div>
    </details>
  );
}
