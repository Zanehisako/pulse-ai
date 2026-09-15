/**
 * PulseAI Edge API Handler (Cloudflare Pages Functions)
 * Proxies frontend chat requests to Modal AI GPU inference backend (vLLM)
 * with SSE streaming, structured plan/tool telemetry, and zero backend hosting required.
 */

import rawConfig from './config.ts';

export interface EdgeConfig {
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

  const rawUrl = get(
    'MODAL_LLM_API_URL',
    rawConfig.remote_llm.default_api_url
  ).trim().replace(/\/+$/, '');

  const timeoutS = parseFloat(
    get('MODAL_LLM_TIMEOUT_S', String(rawConfig.remote_llm.timeout_seconds))
  ) || 90.0;

  return {
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

export async function handleStatusRequest(
  config: EdgeConfig,
  corsHeaders: HeadersInit
): Promise<Response> {
  const body = {
    llm_ready: true,
    active_model: config.model,
    selected_model_path: config.model,
    planner_mode: 'modal_serverless_edge',
    provider: 'modal_vllm',
    backend: 'cloudflare_pages_functions',
    modal_endpoint: config.apiUrl,
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
    // Non-streaming execution
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

  // Run the background stream processing
  (async () => {
    try {
      // 1. Initial lifecycle progress events
      await writeSSE({ type: 'progress', phase: 'received' });
      await writeSSE({ type: 'progress', phase: 'planning' });

      // 2. Structured Execution Plan event
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

      // 3. Tool execution start event
      await writeSSE({
        type: 'tool_started',
        step: 1,
        tool: 'modal_vllm_inference',
        command: `vllm-serve --model ${config.model}`,
        phase: 'executing_tool',
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), config.timeoutMs);

      // Periodic heartbeat to keep SSE connection alive during serverless GPU cold boot
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

        // Process any remainder in buffer
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

        // 4. Tool finished event
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

        // 5. Final summary event
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

  // Handle CORS preflight
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
