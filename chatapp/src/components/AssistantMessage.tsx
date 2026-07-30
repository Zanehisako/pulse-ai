import React, { useMemo } from 'react';
import { marked } from 'marked';
import { ReasoningBox } from './ReasoningBox';
import { PlanDrawer } from './PlanDrawer';
import { ChatMessage } from '../types/index';

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

  return (
    <div className="message-row message-row-assistant">
      <div className="message-avatar assistant-avatar">
        <i className="fa-solid fa-robot"></i>
      </div>

      <div className="message-content">
        <div className="message-header">
          <span className="message-sender">PulseAI Agent</span>
          <span>{message.timestamp}</span>
        </div>

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
              <i className="fa-solid fa-clock"></i> Stream Time: {message.telemetry.durationMs} ms
            </span>
            <span>
              <i className="fa-solid fa-shield-halved"></i> Config-Driven Tool Execution
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
