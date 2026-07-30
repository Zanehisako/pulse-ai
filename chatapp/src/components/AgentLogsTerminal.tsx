import React, { useState, useEffect, useRef } from 'react';
import { LogEntry } from '../types/index';

export interface AgentLogsTerminalProps {
  logs: LogEntry[];
  onClearLogs?: () => void;
}

export function AgentLogsTerminal({ logs, onClearLogs }: AgentLogsTerminalProps): React.JSX.Element {
  const [expanded, setExpanded] = useState<boolean>(false);
  const terminalEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (expanded) {
      terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, expanded]);

  return (
    <div
      style={{
        background: '#090d16',
        border: '1px solid var(--border-bright)',
        borderRadius: '10px',
        margin: '10px 0',
        overflow: 'hidden',
        fontFamily: 'var(--font-mono)'
      }}
    >
      <div
        onClick={() => setExpanded(prev => !prev)}
        style={{
          padding: '8px 14px',
          background: 'rgba(13, 17, 32, 0.95)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          fontSize: '12px',
          color: 'var(--accent-cyan)'
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <i className="fa-solid fa-terminal"></i> Live Agent Logs Terminal ({logs.length} entries)
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {onClearLogs && expanded && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClearLogs();
              }}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '11px'
              }}
            >
              Clear Logs
            </button>
          )}
          <i className={`fa-solid ${expanded ? 'fa-chevron-down' : 'fa-chevron-right'}`}></i>
        </div>
      </div>

      {expanded && (
        <div
          style={{
            maxHeight: '220px',
            overflowY: 'auto',
            padding: '12px',
            fontSize: '11px',
            lineHeight: '1.5',
            color: '#cbd5e1',
            background: '#05070d'
          }}
        >
          {logs.length === 0 ? (
            <div style={{ color: 'var(--text-dim)', fontStyle: 'italic' }}>
              No log output recorded yet. Run a query to view live agent execution logs.
            </div>
          ) : (
            logs.map((log, idx) => (
              <div key={idx} style={{ display: 'flex', gap: '8px', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-dim)', flexShrink: 0 }}>[{log.timestamp}]</span>
                <span
                  style={{
                    color:
                      log.level === 'ERROR'
                        ? 'var(--accent-crimson)'
                        : log.level === 'WARNING'
                        ? 'var(--accent-amber)'
                        : 'var(--accent-cyan)',
                    fontWeight: 600,
                    flexShrink: 0
                  }}
                >
                  [{log.level}]
                </span>
                <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{log.logger}:</span>
                <span style={{ wordBreak: 'break-all' }}>{log.message}</span>
              </div>
            ))
          )}
          <div ref={terminalEndRef} />
        </div>
      )}
    </div>
  );
}
