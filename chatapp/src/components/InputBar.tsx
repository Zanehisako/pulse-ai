import React, { useId, useRef, useState } from 'react';
import { uiConfig } from '../config/uiConfig';

export interface InputBarProps {
  onSend: (queryText: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}

export function InputBar({ onSend, onStop, isStreaming }: InputBarProps): React.JSX.Element {
  const [inputQuery, setInputQuery] = useState<string>('');
  const isComposing = useRef(false);
  const inputId = useId();
  const disclaimerId = useId();

  const handleSubmit = (): void => {
    if (!inputQuery.trim() || isStreaming || isComposing.current) return;
    onSend(inputQuery);
    setInputQuery('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.nativeEvent.isComposing || isComposing.current || e.keyCode === 229) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-area">
      <label className="input-label" htmlFor={inputId}>{uiConfig.labels.placeholder}</label>
      <div className="input-box-wrapper">
        <textarea
          id={inputId}
          className="chat-textarea"
          aria-describedby={disclaimerId}
          placeholder={uiConfig.labels.placeholder}
          value={inputQuery}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setInputQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => { isComposing.current = true; }}
          onCompositionEnd={() => { isComposing.current = false; }}
          rows={1}
        />
        {isStreaming ? (
          <button
            type="button"
            className="send-btn send-btn-stop"
            onClick={onStop}
            title={uiConfig.labels.stop}
            aria-label={uiConfig.labels.stop}
          >
            <i className="fa-solid fa-square" aria-hidden="true"></i>
          </button>
        ) : (
          <button
            type="button"
            className="send-btn"
            disabled={!inputQuery.trim()}
            onClick={handleSubmit}
            title={uiConfig.labels.send}
            aria-label={uiConfig.labels.send}
          >
            <i className="fa-solid fa-paper-plane" aria-hidden="true"></i>
          </button>
        )}
      </div>
      <p className="input-disclaimer" id={disclaimerId}>{uiConfig.labels.disclaimer}</p>
    </div>
  );
}
