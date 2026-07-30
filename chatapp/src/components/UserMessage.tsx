import React from 'react';
import { ChatMessage } from '../types/index';

export interface UserMessageProps {
  message: ChatMessage;
}

export function UserMessage({ message }: UserMessageProps): React.JSX.Element {
  return (
    <div className="message-row message-row-user">
      <div className="message-avatar user-avatar">
        <i className="fa-solid fa-user"></i>
      </div>
      <div className="message-content">
        <div className="message-header">
          <span>{message.timestamp}</span>
          <span className="message-sender">Operator</span>
        </div>
        <div className="user-bubble">{message.text}</div>
      </div>
    </div>
  );
}
