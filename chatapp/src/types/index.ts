/**
 * Shared TypeScript Types & Interfaces for PulseAI Orchestrator ChatApp
 */

export type MessageSender = 'user' | 'assistant';

export type StepStatus = 'pending' | 'executing' | 'success' | 'failed';

export interface ExecutionStep {
  step: number;
  tool: string;
  arguments?: Record<string, unknown>;
  command?: string;
  reasoning?: string;
  status?: StepStatus;
  output?: unknown;
}

export interface MessageTelemetry {
  startTime: number;
  durationMs: number;
}

export interface LogEntry {
  level: string;
  logger: string;
  message: string;
  timestamp: string;
}

export interface ChatMessage {
  id: string;
  sender: MessageSender;
  text?: string;
  markdown?: string;
  timestamp: string;
  phase?: string | null;
  reasoning?: string | null;
  steps?: ExecutionStep[];
  telemetry?: MessageTelemetry | null;
}

export interface OrchestratorStatus {
  llm_ready: boolean;
  active_model: string;
  selected_model_path?: string;
  ws_connected?: boolean;
}

export interface StreamEvent {
  type: 'progress' | 'plan' | 'planner_context' | 'tool_started' | 'tool_finished' | 'token' | 'final' | 'result' | 'log' | 'connection' | 'error';
  phase?: string;
  reasoning?: string;
  steps?: Array<{
    index?: number;
    step?: number;
    tool: string;
    command?: string;
    reasoning?: string;
  }>;
  step?: number;
  tool?: string;
  text?: string;
  token?: string;
  output?: unknown;
  markdown?: string;
  level?: string;
  logger?: string;
  message?: string;
  timestamp?: string;
  result?: {
    summary?: string;
  };
}

export interface StreamPredictionParams {
  query: string;
  features?: Record<string, unknown>;
  onEvent?: (event: StreamEvent) => void;
  onError?: (err: Error) => void;
  onComplete?: () => void;
  signal?: AbortSignal;
}

export interface PresetQuery {
  id: string;
  icon: string;
  title: string;
  desc: string;
  query: string;
}
