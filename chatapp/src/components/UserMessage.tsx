import React from 'react';
import { ChatMessage } from '../types/index';
import { uiConfig } from '../config/uiConfig';

export interface UserMessageProps {
  message: ChatMessage;
}

export function UserMessage({ message }: UserMessageProps): React.JSX.Element {
  return (
    <div className="message-row message-row-user">
      <div className="message-avatar user-avatar" aria-hidden="true">
        <i className="fa-solid fa-user"></i>
      </div>
      <div className="message-content">
        <div className="message-header">
          <span className="message-sender">{uiConfig.labels.user}</span>
          <span>{message.timestamp}</span>
        </div>
        <div className="user-bubble">{message.text}</div>
      </div>
    </div>
  );
}
