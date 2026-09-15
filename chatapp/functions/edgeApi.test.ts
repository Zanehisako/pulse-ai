/**
 * Unit tests for Cloudflare Pages Edge API Handlers
 * Tests config resolution, SSE parsing, status endpoint, streaming and non-streaming prediction,
 * CORS preflight, and graceful fallback behavior.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  resolveEdgeConfig,
  parseVLLMStreamChunk,
  buildPromptWithFeatures,
  buildCorsHeaders,
  handleEdgeApiRequest,
} from './_edgeApi.ts';

test('resolveEdgeConfig returns defaults when no env is provided', () => {
  const config = resolveEdgeConfig({});
  assert.equal(config.model, 'Qwen/Qwen2.5-7B-Instruct');
  assert.equal(config.apiUrl, 'https://yassinetakiko--pulseai-vllm-backend-serve.modal.run');
  assert.equal(config.chatPath, '/v1/chat/completions');
  assert.equal(config.timeoutMs, 90000);
  assert.equal(config.temperature, 0.1);
});

test('resolveEdgeConfig respects environment variable overrides', () => {
  const env = {
    MODAL_LLM_API_URL: 'https://custom-modal-endpoint.modal.run/',
    MODAL_LLM_MODEL: 'custom-model-7b',
    MODAL_LLM_API_KEY: 'secret-token-123',
    MODAL_LLM_TIMEOUT_S: '45.0',
    MODAL_SYSTEM_PROMPT: 'Custom Clinical Assistant',
  };
  const config = resolveEdgeConfig(env);
  assert.equal(config.apiUrl, 'https://custom-modal-endpoint.modal.run');
  assert.equal(config.model, 'custom-model-7b');
  assert.equal(config.apiKey, 'secret-token-123');
  assert.equal(config.timeoutMs, 45000);
  assert.equal(config.systemPrompt, 'Custom Clinical Assistant');
});

test('parseVLLMStreamChunk extracts content tokens from valid SSE lines', () => {
  const chunk = 'data: {"choices":[{"delta":{"content":"Eligible for donation."}}]}';
  const token = parseVLLMStreamChunk(chunk);
  assert.equal(token, 'Eligible for donation.');
});

test('parseVLLMStreamChunk returns null for [DONE] or invalid data', () => {
  assert.equal(parseVLLMStreamChunk('data: [DONE]'), null);
  assert.equal(parseVLLMStreamChunk(': ping'), null);
  assert.equal(parseVLLMStreamChunk(''), null);
  assert.equal(parseVLLMStreamChunk('data: {invalid-json}'), null);
});

test('buildPromptWithFeatures formats patient features cleanly', () => {
  const prompt = buildPromptWithFeatures('Assess donor eligibility', {
    age: 29,
    weight: 70,
    blood_type: 'O+',
  });
  assert.match(prompt, /Assess donor eligibility/);
  assert.match(prompt, /- \*\*age\*\*: 29/);
  assert.match(prompt, /- \*\*weight\*\*: 70/);
  assert.match(prompt, /- \*\*blood_type\*\*: O\+/);
});

test('buildCorsHeaders contains required CORS directives', () => {
  const config = resolveEdgeConfig({});
  const headers = buildCorsHeaders(config) as Record<string, string>;
  assert.equal(headers['Access-Control-Allow-Origin'], '*');
  assert.match(headers['Access-Control-Allow-Methods'], /POST/);
  assert.match(headers['Access-Control-Allow-Headers'], /Content-Type/);
});

test('handleEdgeApiRequest handles OPTIONS preflight', async () => {
  const req = new Request('https://pulse-ai-chatapp.pages.dev/api/ml/predict/nl', {
    method: 'OPTIONS',
  });
  const res = await handleEdgeApiRequest(req);
  assert.equal(res.status, 204);
  assert.equal(res.headers.get('Access-Control-Allow-Origin'), '*');
});

test('handleEdgeApiRequest handles /api/health', async () => {
  const req = new Request('https://pulse-ai-chatapp.pages.dev/api/health', {
    method: 'GET',
  });
  const res = await handleEdgeApiRequest(req);
  assert.equal(res.status, 200);
  const data = (await res.json()) as { status: string };
  assert.equal(data.status, 'ok');
});

test('handleEdgeApiRequest handles /api/ml/orchestrator/status/', async () => {
  const req = new Request(
    'https://pulse-ai-chatapp.pages.dev/api/ml/orchestrator/status/',
    { method: 'GET' }
  );
  const res = await handleEdgeApiRequest(req, {
    MODAL_LLM_MODEL: 'Qwen/Qwen2.5-7B-Instruct',
  });
  assert.equal(res.status, 200);
  const data = (await res.json()) as {
    llm_ready: boolean;
    active_model: string;
    planner_mode: string;
  };
  assert.equal(data.llm_ready, true);
  assert.equal(data.active_model, 'Qwen/Qwen2.5-7B-Instruct');
  assert.equal(data.planner_mode, 'modal_serverless_edge');
});

test('handleEdgeApiRequest rejects GET on /api/ml/predict/nl with 405', async () => {
  const req = new Request('https://pulse-ai-chatapp.pages.dev/api/ml/predict/nl', {
    method: 'GET',
  });
  const res = await handleEdgeApiRequest(req);
  assert.equal(res.status, 405);
});

test('handleEdgeApiRequest non-streaming prediction calls Modal AI backend', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL) => {
    const urlStr = String(input);
    if (urlStr.includes('/chat/completions')) {
      return new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: 'Donor meets all standard criteria (age 28, weight 75kg).',
              },
            },
          ],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }
    return originalFetch(input);
  };

  try {
    const req = new Request('https://pulse-ai-chatapp.pages.dev/api/ml/predict/nl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'Check eligibility for donor aged 28, weight 75kg',
      }),
    });
    const res = await handleEdgeApiRequest(req);
    assert.equal(res.status, 200);
    const data = (await res.json()) as { success: boolean; answer: string };
    assert.equal(data.success, true);
    assert.match(data.answer, /Donor meets all standard criteria/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('handleEdgeApiRequest streaming prediction emits SSE events and tokens', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL) => {
    const urlStr = String(input);
    if (urlStr.includes('/chat/completions')) {
      const ssePayload = [
        'data: {"choices":[{"delta":{"content":"Donor "}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"is eligible."}}]}\n\n',
        'data: [DONE]\n\n',
      ].join('');

      return new Response(ssePayload, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }
    return originalFetch(input);
  };

  try {
    const req = new Request(
      'https://pulse-ai-chatapp.pages.dev/api/ml/predict/nl/?stream=true',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({
          query: 'Evaluate donor 101',
        }),
      }
    );
    const res = await handleEdgeApiRequest(req);
    assert.equal(res.status, 200);
    assert.equal(res.headers.get('Content-Type'), 'text/event-stream; charset=utf-8');

    const text = await res.text();
    assert.match(text, /"type":"progress"/);
    assert.match(text, /"type":"plan"/);
    assert.match(text, /"type":"tool_started"/);
    assert.match(text, /"type":"token"/);
    assert.match(text, /"type":"tool_finished"/);
    assert.match(text, /"type":"final"/);
    assert.match(text, /Donor is eligible\./);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('handleEdgeApiRequest handles backend connection failure gracefully with fallback notice', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error('Connection refused to Modal GPU backend');
  };

  try {
    const req = new Request(
      'https://pulse-ai-chatapp.pages.dev/api/ml/predict/nl/?stream=true',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'Check donor status' }),
      }
    );
    const res = await handleEdgeApiRequest(req);
    assert.equal(res.status, 200);
    const text = await res.text();
    assert.match(text, /"type":"tool_finished"/);
    assert.match(text, /"success":false/);
    assert.match(text, /"type":"final"/);
    assert.match(text, /Decision Support Notice/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
