/**
 * Custom React Hook: useOrchestratorStream (TypeScript)
 * Manages WebSocket streaming query execution, real-time agent log records, reasoning, tool execution, and auto-reconnection.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchOrchestratorStatus, streamNLPrediction } from '../services/api';
import { ChatMessage, OrchestratorStatus, StreamEvent, LogEntry } from '../types/index';

export interface UseOrchestratorStreamReturn {
  messages: ChatMessage[];
  isStreaming: boolean;
  status: OrchestratorStatus;
  logs: LogEntry[];
  sendQuery: (queryText: string) => Promise<void>;
  stopStream: () => void;
  clearMessages: () => void;
  clearLogs: () => void;
}

export function useOrchestratorStream(): UseOrchestratorStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      markdown: '### 🩸 PulseAI Orchestrator Online\nConnected via **Vite + React 18 + WebSocket Protocol (`ws://`)** to **Gemma 12B QAT GGUF** model via local `llama_cpp` engine.\nAsk me any natural-language query to inspect donor eligibility, hospital stockout predictions, or digital twin simulation parameters.',
      timestamp: new Date().toLocaleTimeString(),
      phase: null,
      reasoning: null,
      steps: [],
      telemetry: null
    }
  ]);

  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [status, setStatus] = useState<OrchestratorStatus>({
    llm_ready: true,
    active_model: 'gemma-4-12B-it-QAT-Q4_0.gguf',
    ws_connected: false
  });
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const connectWebSocket = useCallback(() => {
    fetchOrchestratorStatus()
      .then(resStatus => {
        setStatus(prev => ({ ...prev, ...resStatus }));
      })
      .catch(() => {});

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.port === '5173' ? 'localhost:8000' : window.location.host;
    const wsUrl = `${wsProtocol}//${wsHost}/ws/ml/orchestrator/`;

    try {
      const socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        setStatus(prev => ({ ...prev, ws_connected: true }));
      };

      socket.onclose = () => {
        setStatus(prev => ({ ...prev, ws_connected: false }));
        setTimeout(() => {
          if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, 3000);
      };

      socket.onerror = () => {
        setStatus(prev => ({ ...prev, ws_connected: false }));
      };
    } catch (e) {
      console.warn('WebSocket connection failed:', e);
    }
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  const sendQuery = useCallback(async (queryText: string): Promise<void> => {
    if (!queryText || !queryText.trim() || isStreaming) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: queryText.trim(),
      timestamp: new Date().toLocaleTimeString()
    };

    const assistantMsgId = (Date.now() + 1).toString();
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      sender: 'assistant',
      markdown: '',
      timestamp: new Date().toLocaleTimeString(),
      phase: 'received',
      reasoning: null,
      steps: [],
      telemetry: { startTime: performance.now(), durationMs: 0 }
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const handleEvent = (event: StreamEvent) => {
      if (event.type === 'log') {
        setLogs(prev => [...prev.slice(-300), {
          level: event.level || 'INFO',
          logger: event.logger || 'orchestrator',
          message: event.message || '',
          timestamp: event.timestamp || new Date().toLocaleTimeString()
        }]);
        return;
      }

      setMessages(prev => prev.map(m => {
        if (m.id !== assistantMsgId) return m;

        const updated: ChatMessage = { ...m };
        if (event.phase) updated.phase = event.phase;

        if (event.type === 'planner_context') {
          updated.phase = 'planning';
        } else if (event.type === 'plan') {
          updated.phase = 'planned';
          updated.reasoning = event.reasoning || null;
          if (event.steps && event.steps.length > 0) {
            updated.steps = event.steps.map((s, idx) => ({
              step: s.index || idx + 1,
              tool: s.tool,
              command: s.command,
              reasoning: s.reasoning,
              status: 'pending'
            }));
          } else {
            updated.steps = [];
          }
        } else if (event.type === 'tool_started') {
          updated.phase = 'executing_tool';
          const targetStepIndex = event.step || 1;
          let stepFound = false;

          updated.steps = (updated.steps || []).map(s => {
            if (s.step === targetStepIndex || s.tool === event.tool) {
              stepFound = true;
              return { ...s, status: 'executing', tool: event.tool || s.tool };
            }
            return s;
          });

          if (!stepFound && event.tool) {
            updated.steps = [...(updated.steps || []), {
              step: targetStepIndex,
              tool: event.tool,
              command: event.command,
              status: 'executing'
            }];
          }
        } else if (event.type === 'tool_finished') {
          const isSuccess = event.success !== false && !event.error;
          updated.phase = isSuccess ? 'tool_completed' : 'tool_failed';
          const targetStepIndex = event.step || 1;
          let stepFound = false;

          updated.steps = (updated.steps || []).map(s => {
            if (s.step === targetStepIndex || s.tool === event.tool) {
              stepFound = true;
              return {
                ...s,
                status: isSuccess ? ('success' as const) : ('failed' as const),
                output: event.output,
                error: event.error || (isSuccess ? undefined : 'Tool execution failed or was skipped.')
              };
            }
            return s;
          });

          if (!stepFound && event.tool) {
            updated.steps = [...(updated.steps || []), {
              step: targetStepIndex,
              tool: event.tool,
              command: event.command,
              status: isSuccess ? ('success' as const) : ('failed' as const),
              output: event.output,
              error: event.error || (isSuccess ? undefined : 'Tool execution failed.')
            }];
          }
        } else if (event.type === 'token') {
          updated.phase = 'generating_response';
          const newContent = event.text !== undefined ? event.text : ((updated.markdown || '') + (event.token || ''));
          updated.markdown = newContent;
        } else if (event.type === 'final' || event.type === 'result') {
          updated.phase = 'completed';
          const summary = event.markdown || (event.result ? event.result.summary : null);
          if (summary) updated.markdown = summary;
          // Finalize all remaining executing/pending steps as finished
          updated.steps = (updated.steps || []).map(s =>
            s.status === 'executing' ? { ...s, status: 'success' } : s
          );
          if (updated.telemetry) {
            updated.telemetry = {
              ...updated.telemetry,
              durationMs: Math.round(performance.now() - updated.telemetry.startTime)
            };
          }
        }
        return updated;
      }));
    };

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const onWsMessage = (msgEvent: MessageEvent) => {
        try {
          const event: StreamEvent = JSON.parse(msgEvent.data);
          handleEvent(event);
          if (event.type === 'final' || event.type === 'error' || event.phase === 'completed') {
            wsRef.current?.removeEventListener('message', onWsMessage);
            setIsStreaming(false);
          }
        } catch (e) {}
      };
      wsRef.current.addEventListener('message', onWsMessage);
      wsRef.current.send(JSON.stringify({
        action: 'predict',
        query: queryText.trim(),
        features: { current_inventory: 45.0, daily_consumption_rate: 12.0 }
      }));
    } else {
      abortControllerRef.current = new AbortController();
      await streamNLPrediction({
        query: queryText.trim(),
        features: { current_inventory: 45.0, daily_consumption_rate: 12.0 },
        signal: abortControllerRef.current.signal,
        onEvent: handleEvent,
        onError: (err: Error) => {
          setMessages(prev => prev.map(m =>
            m.id === assistantMsgId ? { ...m, phase: 'error', markdown: '❌ Error: ' + err.message } : m
          ));
        },
        onComplete: () => {
          setIsStreaming(false);
        }
      });
    }
  }, [isStreaming]);

  const stopStream = useCallback((): void => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsStreaming(false);
  }, []);

  const clearMessages = useCallback((): void => {
    setMessages([]);
  }, []);

  const clearLogs = useCallback((): void => {
    setLogs([]);
  }, []);

  return {
    messages,
    isStreaming,
    status,
    logs,
    sendQuery,
    stopStream,
    clearMessages,
    clearLogs
  };
}
