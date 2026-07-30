import React, { useState } from 'react';

export interface InputBarProps {
  onSend: (queryText: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}

export function InputBar({ onSend, onStop, isStreaming }: InputBarProps): React.JSX.Element {
  const [inputQuery, setInputQuery] = useState<string>('');

  const handleSubmit = (): void => {
    if (!inputQuery.trim() || isStreaming) return;
    onSend(inputQuery);
    setInputQuery('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-area">
      <div className="input-box-wrapper">
        <textarea
          className="chat-textarea"
          placeholder="Ask the PulseAI Orchestrator anything (e.g. Check donor eligibility, predict stockout risk)..."
          value={inputQuery}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setInputQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        {isStreaming ? (
          <button
            className="send-btn"
            style={{ background: 'var(--accent-amber)', color: '#000' }}
            onClick={onStop}
            title="Stop Streaming"
          >
            <i className="fa-solid fa-square"></i>
          </button>
        ) : (
          <button
            className="send-btn"
            disabled={!inputQuery.trim()}
            onClick={handleSubmit}
            title="Send Query"
          >
            <i className="fa-solid fa-paper-plane"></i>
          </button>
        )}
      </div>
    </div>
  );
}
