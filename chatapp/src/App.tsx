import React from 'react';
import { Header } from './components/Header';
import { AgentLogsTerminal } from './components/AgentLogsTerminal';
import { ChatFeed } from './components/ChatFeed';
import { InputBar } from './components/InputBar';
import { useOrchestratorStream } from './hooks/useOrchestratorStream';
import { useTheme } from './hooks/useTheme';

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
  const { preference, setPreference } = useTheme();

  return (
    <React.Fragment>
      <Header
        status={status}
        onClear={clearMessages}
        isStreaming={isStreaming}
        theme={preference}
        onThemeChange={setPreference}
        onSelectPreset={sendQuery}
      />
      <div className="app-container">
        <main className="main-chat-area">
          <AgentLogsTerminal logs={logs} onClearLogs={clearLogs} isStreaming={isStreaming} />
          <ChatFeed messages={messages} isStreaming={isStreaming} onSelectPreset={sendQuery} />
          <InputBar onSend={sendQuery} onStop={stopStream} isStreaming={isStreaming} />
        </main>
      </div>
    </React.Fragment>
  );
}
