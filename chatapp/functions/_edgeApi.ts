/**
 * PulseAI Edge API Handler (Cloudflare Pages Functions)
 * Proxies frontend chat requests to:
 * 1. The full Django DynamicXLAMOrchestrator (when DJANGO_BACKEND_URL is configured),
 *    executing real tools, ML models, and clinical alert pipelines.
 * 2. Fallback direct Modal AI GPU backend (when DJANGO_BACKEND_URL is not configured).
 */

import rawConfig from './config.ts';

export interface EdgeConfig {
  djangoUrl: string;
  djangoStatusPath: string;
  djangoPredictPath: string;
  djangoTimeoutMs: number;
  djangoAwaitHeartbeatSeconds: number;
  apiUrl: string;
  model: string;
  apiKey: string;
  chatPath: string;
  healthPath: string;
  timeoutMs: number;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  corsOrigins: string[];
}

export function resolveEdgeConfig(env?: Record<string, string | undefined>): EdgeConfig {
  const get = (key: string, fallback: string): string => {
    if (env && env[key] !== undefined && env[key] !== '') {
      return String(env[key]);
    }
    return fallback;
  };

  const rawDjangoUrl = get(
    'DJANGO_BACKEND_URL',
    rawConfig.django_backend?.default_url || ''
  ).trim().replace(/\/+$/, '');

  const djangoTimeoutS = parseFloat(
    get('DJANGO_BACKEND_TIMEOUT_S', String(rawConfig.django_backend?.timeout_seconds || 120))
  ) || 120.0;

  const djangoAwaitHeartbeatS = parseFloat(
    get('DJANGO_BACKEND_AWAIT_HEARTBEAT_S', String(rawConfig.django_backend?.await_heartbeat_seconds ?? 5))
  ) || 5.0;

  const rawUrl = get(
    'MODAL_LLM_API_URL',
    rawConfig.remote_llm.default_api_url
  ).trim().replace(/\/+$/, '');

  const timeoutS = parseFloat(
    get('MODAL_LLM_TIMEOUT_S', String(rawConfig.remote_llm.timeout_seconds))
  ) || 110.0;

  return {
    djangoUrl: rawDjangoUrl,
    djangoStatusPath: rawConfig.django_backend?.status_path || '/api/ml/orchestrator/status/',
    djangoPredictPath: rawConfig.django_backend?.predict_stream_path || '/api/ml/predict/nl/?stream=true',
    djangoTimeoutMs: djangoTimeoutS * 1000,
    djangoAwaitHeartbeatSeconds: djangoAwaitHeartbeatS,
    apiUrl: rawUrl,
    model: get('MODAL_LLM_MODEL', rawConfig.remote_llm.default_model).trim(),
    apiKey: get('MODAL_LLM_API_KEY', '').trim(),
    chatPath: rawConfig.remote_llm.chat_completions_path || '/v1/chat/completions',
    healthPath: rawConfig.remote_llm.health_path || '/health',
    timeoutMs: timeoutS * 1000,
    temperature: rawConfig.remote_llm.temperature || 0.1,
    maxTokens: rawConfig.remote_llm.max_tokens || 1024,
    systemPrompt: get('MODAL_SYSTEM_PROMPT', rawConfig.remote_llm.system_prompt).trim(),
    corsOrigins: rawConfig.cors.allow_origins || ['*'],
  };
}

export function parseVLLMStreamChunk(line: string): string | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) {
    return null;
  }
  const payloadStr = trimmed.slice(5).trim();
  if (!payloadStr || payloadStr === '[DONE]') {
    return null;
  }
  try {
    const parsed = JSON.parse(payloadStr);
    const choices = parsed.choices;
    if (Array.isArray(choices) && choices.length > 0) {
      const delta = choices[0].delta;
      if (delta && typeof delta.content === 'string') {
        return delta.content;
      }
    }
  } catch {
    // Ignore JSON parse errors for incomplete/corrupted SSE chunks
  }
  return null;
}

export function buildPromptWithFeatures(
  query: string,
  features?: Record<string, unknown>
): string {
  const parts: string[] = [query.trim()];
  if (features && Object.keys(features).length > 0) {
    parts.push('\n\n### Provided Context & Patient/Donor Features:');
    for (const [key, value] of Object.entries(features)) {
      if (value !== undefined && value !== null && value !== '') {
        parts.push(`- **${key}**: ${value}`);
      }
    }
  }
  return parts.join('\n');
}

export function buildCorsHeaders(config: EdgeConfig): HeadersInit {
  return {
    'Access-Control-Allow-Origin': config.corsOrigins.join(', '),
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept, X-Requested-With',
  };
}

export async function forwardToDjangoPredict(
  request: Request,
  query: string,
  features: Record<string, unknown>,
  isStreaming: boolean,
  config: EdgeConfig,
  corsHeaders: HeadersInit
): Promise<Response> {
  const basePath = isStreaming
    ? config.djangoPredictPath
    : config.djangoPredictPath.replace('?stream=true', '');
  const targetUrl = `${config.djangoUrl}${basePath}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.djangoTimeoutMs);

  if (!isStreaming) {
    try {
      const res = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({ query, features }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const resData = await res.text();
      return new Response(resData, {
        status: res.status,
        headers: {
          'Content-Type': res.headers.get('Content-Type') || 'application/json',
          'X-PulseAI-Backend': 'django-orchestrator',
          ...corsHeaders,
        },
      });
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      const msg = err instanceof Error ? err.message : String(err);
      return new Response(
        JSON.stringify({
          success: false,
          error: `Failed to reach Django orchestrator at ${config.djangoUrl}: ${msg}`,
          planner_mode: 'django_orchestrator_error',
        }),
        {
          status: 502,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        }
      );
    }
  }

  // Streaming: return a live stream immediately so the browser receives bytes
  // while the Django backend boots or warms up (serverless cold starts can
  // take a minute with zero upstream bytes). Emit a preamble plus periodic
  // heartbeats until Django's first byte arrives, then pipe its SSE through.
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();
  const streamStartedAt = Date.now();
  const heartbeatMs = Math.max(
    1000,
    (config.djangoAwaitHeartbeatSeconds || 5) * 1000
  );

  const writeSSE = async (data: Record<string, unknown>) => {
    await writer.write(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
  };

  (async () => {
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    try {
      await writeSSE({ type: 'progress', phase: 'received' });
      heartbeat = setInterval(() => {
        writeSSE({
          type: 'progress',
          phase: 'awaiting_orchestrator',
          heartbeat: true,
          elapsed_seconds: Math.round((Date.now() - streamStartedAt) / 100) / 10,
          message: 'Django orchestrator is starting; live updates stream as they arrive.',
        }).catch(() => {
          // Stream might be closed
        });
      }, heartbeatMs);

      const res = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ query, features }),
        signal: controller.signal,
      });

      // Upstream headers arrived: stop preamble heartbeats, pipe SSE through.
      clearInterval(heartbeat);
      heartbeat = undefined;

      if (!res.ok || !res.body) {
        const detail = await res.text().catch(() => '');
        await writeSSE({
          type: 'tool_finished',
          step: 1,
          tool: 'django_orchestrator',
          success: false,
          error: `Django orchestrator HTTP ${res.status}: ${detail.slice(0, 300)}`,
        });
        await writeSSE({
          type: 'final',
          phase: 'completed',
          markdown: `### ⚠️ Django Orchestrator Error\n\nBackend at \`${config.djangoUrl}\` returned HTTP ${res.status}.\n\`${detail.slice(0, 300)}\``,
          result: {
            summary: `Django orchestrator HTTP ${res.status}`,
          },
        });
        return;
      }

      const reader = res.body.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          await writer.write(value);
        }
      } finally {
        reader.releaseLock();
      }
    } catch (err: unknown) {
      if (heartbeat) clearInterval(heartbeat);
      heartbeat = undefined;
      const msg = err instanceof Error ? err.message : String(err);

      try {
        await writeSSE({
          type: 'tool_finished',
          step: 1,
          tool: 'django_orchestrator',
          success: false,
          error: `Django backend unreachable at ${config.djangoUrl}: ${msg}`,
        });
        await writeSSE({
          type: 'final',
          phase: 'completed',
          markdown: `### ⚠️ Django Orchestrator Connection Notice\n\nCould not reach Django backend at \`${config.djangoUrl}\`:\n\`${msg}\`\n\n*Please ensure your Django backend or Cloudflare Tunnel is running.*`,
          result: {
            summary: `Django backend unreachable: ${msg}`,
          },
        });
      } catch {
        // Client disconnected mid-stream; nothing left to write.
      }
    } finally {
      if (heartbeat) clearInterval(heartbeat);
      clearTimeout(timeoutId);
      try {
        await writer.close();
      } catch {
        // Already closed
      }
    }
  })();

  return new Response(readable, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-PulseAI-Backend': 'django-orchestrator',
      ...corsHeaders,
    },
  });
}

export async function handleStatusRequest(
  config: EdgeConfig,
  corsHeaders: HeadersInit
): Promise<Response> {
  if (config.djangoUrl) {
    try {
      const targetUrl = `${config.djangoUrl}${config.djangoStatusPath}`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);
      const djangoRes = await fetch(targetUrl, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (djangoRes.ok) {
        const djangoData = (await djangoRes.json()) as Record<string, unknown>;
        return new Response(
          JSON.stringify(
            {
              ...djangoData,
              orchestration_mode: 'django_orchestrator',
              django_backend_url: config.djangoUrl,
            },
            null,
            2
          ),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          }
        );
      }
    } catch {
      // Fallback if Django probe times out or is unreachable
    }
  }

  const body = {
    llm_ready: true,
    active_model: config.model,
    selected_model_path: config.model,
    planner_mode: config.djangoUrl ? 'django_orchestrator_unreachable' : 'modal_serverless_edge',
    provider: config.djangoUrl ? 'django' : 'modal_vllm',
    backend: 'cloudflare_pages_functions',
    modal_endpoint: config.apiUrl,
    django_backend_configured: Boolean(config.djangoUrl),
    timestamp: new Date().toISOString(),
  };

  return new Response(JSON.stringify(body, null, 2), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders,
    },
  });
}

export async function handlePredictRequest(
  request: Request,
  config: EdgeConfig,
  corsHeaders: HeadersInit
): Promise<Response> {
  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed. Use POST.' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }

  let query = '';
  let features: Record<string, unknown> = {};

  try {
    const reqBody = (await request.json()) as {
      query?: string;
      features?: Record<string, unknown>;
    };
    query = (reqBody.query || '').trim();
    features = reqBody.features || {};
  } catch {
    return new Response(JSON.stringify({ error: 'Invalid JSON request body.' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }

  if (!query) {
    query = 'Provide blood donor eligibility guidelines and platform overview.';
  }

  const urlObj = new URL(request.url);
  const acceptHeader = request.headers.get('Accept') || '';
  const isStreaming =
    urlObj.searchParams.get('stream') === 'true' ||
    acceptHeader.includes('text/event-stream');

  // If a Django backend URL is configured, forward query directly to the Django Orchestrator!
  if (config.djangoUrl) {
    return forwardToDjangoPredict(
      request,
      query,
      features,
      isStreaming,
      config,
      corsHeaders
    );
  }

  // Fallback: direct Modal AI serverless edge execution
  const fullPrompt = buildPromptWithFeatures(query, features);
  const targetUrl = `${config.apiUrl}${config.chatPath}`;

  const requestHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: isStreaming ? 'text/event-stream' : 'application/json',
  };
  if (config.apiKey) {
    requestHeaders['Authorization'] = `Bearer ${config.apiKey}`;
  }

  const modalPayload = {
    model: config.model,
    messages: [
      { role: 'system', content: config.systemPrompt },
      { role: 'user', content: fullPrompt },
    ],
    temperature: config.temperature,
    max_tokens: config.maxTokens,
    stream: isStreaming,
  };

  if (!isStreaming) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), config.timeoutMs);

      const modalRes = await fetch(targetUrl, {
        method: 'POST',
        headers: requestHeaders,
        body: JSON.stringify(modalPayload),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!modalRes.ok) {
        const errText = await modalRes.text();
        throw new Error(`Modal GPU HTTP ${modalRes.status}: ${errText}`);
      }

      const resData = (await modalRes.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      const answer = resData.choices?.[0]?.message?.content || '';

      return new Response(
        JSON.stringify(
          {
            success: true,
            query,
            planner_mode: 'modal_serverless_edge',
            answer,
            markdown: answer,
          },
          null,
          2
        ),
        {
          status: 200,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        }
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return new Response(
        JSON.stringify(
          {
            success: false,
            error: msg,
            planner_mode: 'modal_serverless_edge',
          },
          null,
          2
        ),
        {
          status: 502,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        }
      );
    }
  }

  // Streaming execution (SSE)
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();

  const writeSSE = async (data: Record<string, unknown>) => {
    await writer.write(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
  };

  (async () => {
    try {
      await writeSSE({ type: 'progress', phase: 'received' });
      await writeSSE({ type: 'progress', phase: 'planning' });

      await writeSSE({
        type: 'plan',
        phase: 'planned',
        reasoning: `Orchestrating clinical decision query via Modal AI GPU backend (${config.model}).`,
        steps: [
          {
            index: 1,
            step: 1,
            tool: 'modal_vllm_inference',
            command: `vllm-serve --model ${config.model}`,
            reasoning: 'Synthesizing clinical donor eligibility guidelines on NVIDIA A10G',
          },
        ],
      });

      await writeSSE({
        type: 'tool_started',
        step: 1,
        tool: 'modal_vllm_inference',
        command: `vllm-serve --model ${config.model}`,
        phase: 'executing_tool',
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), config.timeoutMs);

      const heartbeatInterval = setInterval(async () => {
        try {
          await writeSSE({
            type: 'progress',
            phase: 'waiting_for_gpu',
            message: 'Serverless GPU on Modal AI is warming up...',
          });
        } catch {
          // Stream might be closed
        }
      }, 15000);

      try {
        const modalRes = await fetch(targetUrl, {
          method: 'POST',
          headers: requestHeaders,
          body: JSON.stringify(modalPayload),
          signal: controller.signal,
        });

        if (!modalRes.ok) {
          clearTimeout(timeoutId);
          const errText = await modalRes.text();
          throw new Error(`Modal GPU HTTP ${modalRes.status}: ${errText}`);
        }

        if (!modalRes.body) {
          clearTimeout(timeoutId);
          throw new Error('Modal response did not contain a readable stream body.');
        }

        const reader = modalRes.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let accumulatedText = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const content = parseVLLMStreamChunk(line);
            if (content) {
              accumulatedText += content;
              await writeSSE({
                type: 'token',
                phase: 'generating_response',
                token: content,
                text: accumulatedText,
              });
            }
          }
        }

        clearTimeout(timeoutId);

        if (buffer.trim()) {
          const content = parseVLLMStreamChunk(buffer);
          if (content) {
            accumulatedText += content;
            await writeSSE({
              type: 'token',
              phase: 'generating_response',
              token: content,
              text: accumulatedText,
            });
          }
        }

        await writeSSE({
          type: 'tool_finished',
          step: 1,
          tool: 'modal_vllm_inference',
          phase: 'tool_completed',
          success: true,
          output: {
            model: config.model,
            tokens_generated: accumulatedText.length,
          },
        });

        await writeSSE({
          type: 'final',
          phase: 'completed',
          markdown: accumulatedText,
          result: {
            summary: accumulatedText,
          },
        });
      } finally {
        clearInterval(heartbeatInterval);
        clearTimeout(timeoutId);
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : String(err);
      const isAbort =
        errMsg.toLowerCase().includes('abort') ||
        errMsg.toLowerCase().includes('timeout') ||
        errMsg.toLowerCase().includes('timed out');

      const fallbackMarkdown = isAbort
        ? `### ⚠️ Modal GPU Serverless Cold-Start\n\nThe serverless GPU on Modal AI was cold-starting (provisioning an NVIDIA A10G container and loading weights). The container is now **waking up or warm**.\n\n**Please resend your query now** — subsequent responses stream within ~2-5 seconds!`
        : `### ⚠️ Decision Support Notice\n\nUnable to reach Modal GPU inference backend:\n\`${errMsg}\`\n\n*Please verify your Modal deployment or consult local blood bank protocols.*`;

      await writeSSE({
        type: 'tool_finished',
        step: 1,
        tool: 'modal_vllm_inference',
        phase: 'tool_failed',
        success: false,
        error: isAbort ? 'Modal GPU cold-start timeout. Container is now warm.' : errMsg,
      });

      await writeSSE({
        type: 'final',
        phase: 'completed',
        markdown: fallbackMarkdown,
        result: { summary: fallbackMarkdown },
      });
    } finally {
      await writer.close();
    }
  })();

  return new Response(readable, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      ...corsHeaders,
    },
  });
}

export async function handleEdgeApiRequest(
  request: Request,
  env?: Record<string, string | undefined>
): Promise<Response> {
  const config = resolveEdgeConfig(env);
  const corsHeaders = buildCorsHeaders(config);

  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: corsHeaders,
    });
  }

  const url = new URL(request.url);
  const pathname = url.pathname.replace(/\/+$/, '');

  if (pathname.endsWith('/health') || pathname === '/api/health') {
    return new Response(
      JSON.stringify({ status: 'ok', timestamp: new Date().toISOString() }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      }
    );
  }

  if (
    pathname.endsWith('/orchestrator/status') ||
    pathname === '/api/ml/orchestrator/status'
  ) {
    return handleStatusRequest(config, corsHeaders);
  }

  if (pathname.endsWith('/predict/nl') || pathname === '/api/ml/predict/nl') {
    return handlePredictRequest(request, config, corsHeaders);
  }

  return new Response(
    JSON.stringify({ error: 'Endpoint not found', pathname }),
    {
      status: 404,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    }
  );
}
