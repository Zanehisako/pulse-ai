import React, { useEffect, useRef } from 'react';
import { UserMessage } from './UserMessage';
import { AssistantMessage } from './AssistantMessage';
import { Sidebar } from './Sidebar';
import { ChatMessage } from '../types/index';
import { uiConfig } from '../config/uiConfig';

export interface ChatFeedProps {
  messages: ChatMessage[];
  isStreaming?: boolean;
  onSelectPreset: (query: string) => void;
}

export function ChatFeed({ messages, isStreaming, onSelectPreset }: ChatFeedProps): React.JSX.Element {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="chat-feed-empty">
        <div className="welcome-block">
          <h1 className="welcome-title">{uiConfig.welcomeTitle}</h1>
          <p className="welcome-description">{uiConfig.welcomeDescription}</p>
          <Sidebar onSelectPreset={onSelectPreset} disabled={isStreaming} />
        </div>
      </div>
    );
  }

  return (
    <div className="chat-messages">
      {messages.map((msg, index) =>
        msg.sender === 'user' ? (
          <UserMessage key={msg.id} message={msg} />
        ) : (
          <AssistantMessage
            key={msg.id}
            message={msg}
            isStreaming={isStreaming && index === messages.length - 1}
          />
        )
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
