import React, { useEffect, useMemo, useState } from 'react';
import { marked } from 'marked';
import { ReasoningBox } from './ReasoningBox';
import { PlanDrawer } from './PlanDrawer';
import { ChatMessage } from '../types/index';
import { uiConfig } from '../config/uiConfig';

export interface AssistantMessageProps {
  message: ChatMessage;
  isStreaming?: boolean;
}

export function AssistantMessage({ message, isStreaming }: AssistantMessageProps): React.JSX.Element {
  const renderedHtml = useMemo<string>(() => {
    try {
      return marked.parse(message.markdown || '') as string;
    } catch {
      return message.markdown || '';
    }
  }, [message.markdown]);

  const showPhase = Boolean(
    isStreaming &&
    message.phase &&
    message.phase !== 'completed' &&
    message.phase !== 'error' &&
    (!message.telemetry || message.telemetry.durationMs === 0)
  );
  const humanizedPhase = (message.phase || '').replace(/_/g, ' ');

  const [elapsedSec, setElapsedSec] = useState<number>(0);
  useEffect(() => {
    if (!isStreaming) return;
    setElapsedSec(0);
    const timer = setInterval(() => setElapsedSec(prev => prev + 1), 1000);
    return () => clearInterval(timer);
  }, [isStreaming, message.id]);

  return (
    <div className="message-row message-row-assistant">
      <div className="message-avatar assistant-avatar" aria-hidden="true">
        <i className="fa-solid fa-robot"></i>
      </div>

      <div className="message-content">
        <div className="message-header">
          <span className="message-sender">{uiConfig.labels.assistant}</span>
          <span>{message.timestamp}</span>
        </div>

        {showPhase && (
          <div className="agent-phase-indicator">
            <i className="fa-solid fa-circle-notch fa-spin" aria-hidden="true"></i>
            <span>{humanizedPhase}... · {elapsedSec}s</span>
          </div>
        )}
        <ReasoningBox reasoning={message.reasoning} />
        <PlanDrawer steps={message.steps} />

        <div className="markdown-body">
          <span dangerouslySetInnerHTML={{ __html: renderedHtml }} />
          {isStreaming && (!message.telemetry || message.telemetry.durationMs === 0) && (
            <span className="typing-cursor"></span>
          )}
        </div>

        {message.telemetry && message.telemetry.durationMs > 0 && (
          <div className="telemetry-tag">
            <span>
              <i className="fa-solid fa-clock" aria-hidden="true"></i> {message.telemetry.durationMs} ms
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
