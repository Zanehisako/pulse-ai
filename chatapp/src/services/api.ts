/**
 * PulseAI Orchestrator API Service (TypeScript)
 * Encapsulates typed backend communication, model status checks, and EventStream parsing.
 */

import { OrchestratorStatus, StreamEvent, StreamPredictionParams } from '../types/index';

export async function fetchOrchestratorStatus(): Promise<OrchestratorStatus> {
  const response = await fetch('/api/ml/orchestrator/status/');
  if (!response.ok) {
    throw new Error(`Status HTTP ${response.status}`);
  }
  const data = await response.json();
  return {
    llm_ready: Boolean(data.llm_ready),
    active_model: data.selected_model_path
      ? data.selected_model_path.split('/').pop() || 'gemma-4-12B-it-QAT-Q4_0.gguf'
      : 'gemma-4-12B-it-QAT-Q4_0.gguf',
    selected_model_path: data.selected_model_path
  };
}

export async function streamNLPrediction({
  query,
  features = {},
  onEvent,
  onError,
  onComplete,
  signal
}: StreamPredictionParams): Promise<void> {
  try {
    const response = await fetch('/api/ml/predict/nl/?stream=true', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({ query, features }),
      signal
    });

    if (!response.ok) {
      throw new Error(`Prediction HTTP ${response.status}`);
    }

    if (!response.body) {
      throw new Error('ReadableStream not supported on response body.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.replace('data: ', '').trim();
          if (!jsonStr) continue;
          try {
            const event: StreamEvent = JSON.parse(jsonStr);
            if (onEvent) onEvent(event);
          } catch (e) {
            console.warn('Failed to parse SSE payload:', e);
          }
        }
      }
    }
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.name !== 'AbortError' && onError) {
        onError(err);
      }
    } else if (onError) {
      onError(new Error(String(err)));
    }
  } finally {
    if (onComplete) onComplete();
  }
}
