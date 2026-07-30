import React from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { AgentLogsTerminal } from './components/AgentLogsTerminal';
import { ChatFeed } from './components/ChatFeed';
import { InputBar } from './components/InputBar';
import { useOrchestratorStream } from './hooks/useOrchestratorStream';

export function App(): React.JSX.Element {
  const {
    messages,
    isStreaming,
    status,
    logs,
    sendQuery,
    stopStream,
    clearMessages,
    clearLogs
  } = useOrchestratorStream();

  return (
    <React.Fragment>
      <Header status={status} onClear={clearMessages} />
      <div className="app-container">
        <Sidebar onSelectPreset={(presetQuery: string) => sendQuery(presetQuery)} />
        <main className="main-chat-area">
          <AgentLogsTerminal logs={logs} onClearLogs={clearLogs} />
          <ChatFeed messages={messages} isStreaming={isStreaming} />
          <InputBar
            onSend={(query: string) => sendQuery(query)}
            onStop={stopStream}
            isStreaming={isStreaming}
          />
        </main>
      </div>
    </React.Fragment>
  );
}
