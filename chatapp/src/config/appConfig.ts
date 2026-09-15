/// <reference types="vite/client" />
/**
 * PulseAI Frontend Configuration
 * Fully config-driven: reads from environment variables with sensible defaults.
 * Allows seamless hosting on Cloudflare Pages, local dev proxy, or custom backend domains.
 */

export interface AppConfig {
  apiBaseUrl: string;
  wsBaseUrl: string;
  defaultModel: string;
  endpoints: {
    status: string;
    streamPredict: string;
    wsOrchestrator: string;
  };
  resolveApiUrl: (path?: string) => string;
  resolveWsUrl: (path?: string) => string;
}

export function buildAppConfig(envSource?: Record<string, string | undefined>): AppConfig {
  const get = (key: string, def = ''): string => {
    if (envSource && envSource[key] !== undefined) {
      return envSource[key] || '';
    }
    if (
      typeof import.meta !== 'undefined' &&
      import.meta.env &&
      (import.meta.env as Record<string, string | undefined>)[key] !== undefined
    ) {
      return String((import.meta.env as Record<string, string | undefined>)[key] || '');
    }
    if (
      typeof process !== 'undefined' &&
      process.env &&
      process.env[key] !== undefined
    ) {
      return String(process.env[key] || '');
    }
    return def;
  };

  const rawApiBaseUrl = (get('VITE_API_BASE_URL') || get('VITE_API_URL') || '')
    .trim()
    .replace(/\/+$/, '');
  const rawWsBaseUrl = (get('VITE_WS_BASE_URL') || '').trim().replace(/\/+$/, '');
  const defaultModel = (
    get('VITE_DEFAULT_MODEL') || 'Qwen/Qwen2.5-7B-Instruct'
  ).trim();

  const endpoints = {
    status: get('VITE_API_STATUS_PATH') || '/api/ml/orchestrator/status/',
    streamPredict:
      get('VITE_API_STREAM_PATH') || '/api/ml/predict/nl/?stream=true',
    wsOrchestrator: get('VITE_WS_PATH') || '/ws/ml/orchestrator/',
  };

  const resolveApiUrl = (path: string = endpoints.status): string => {
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    if (!rawApiBaseUrl) {
      return normalizedPath;
    }
    return `${rawApiBaseUrl}${normalizedPath}`;
  };

  const resolveWsUrl = (path: string = endpoints.wsOrchestrator): string => {
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;

    if (rawWsBaseUrl) {
      return `${rawWsBaseUrl}${normalizedPath}`;
    }

    if (rawApiBaseUrl) {
      const derived = rawApiBaseUrl
        .replace(/^http:\/\//i, 'ws://')
        .replace(/^https:\/\//i, 'wss://');
      return `${derived}${normalizedPath}`;
    }

    if (typeof window !== 'undefined' && window.location) {
      const hostname = window.location.hostname;
      if (hostname === 'localhost' || hostname === '127.0.0.1') {
        const wsPort = window.location.port === '5173' ? '8000' : window.location.port || '8000';
        return `ws://localhost:${wsPort}${normalizedPath}`;
      }
      // On Cloudflare Pages or remote production without explicit VITE_WS_BASE_URL, do not attempt WS
      return '';
    }

    return '';
  };

  return {
    apiBaseUrl: rawApiBaseUrl,
    wsBaseUrl: rawWsBaseUrl,
    defaultModel,
    endpoints,
    resolveApiUrl,
    resolveWsUrl,
  };
}

export const appConfig = buildAppConfig();
