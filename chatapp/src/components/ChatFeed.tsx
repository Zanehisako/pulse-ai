import React, { useEffect, useRef } from 'react';
import { UserMessage } from './UserMessage';
import { AssistantMessage } from './AssistantMessage';
import { ChatMessage } from '../types/index';

export interface ChatFeedProps {
  messages: ChatMessage[];
  isStreaming?: boolean;
}

export function ChatFeed({ messages, isStreaming }: ChatFeedProps): React.JSX.Element {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
