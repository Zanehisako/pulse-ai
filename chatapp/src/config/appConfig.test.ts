import test from 'node:test';
import assert from 'node:assert/strict';
import { buildAppConfig } from './appConfig.ts';

test('buildAppConfig provides default relative URLs when no env vars are set', () => {
  const config = buildAppConfig({});
  assert.equal(config.apiBaseUrl, '');
  assert.equal(config.wsBaseUrl, '');
  assert.equal(config.defaultModel, 'Qwen/Qwen2.5-7B-Instruct');
  assert.equal(config.resolveApiUrl('/api/ml/orchestrator/status/'), '/api/ml/orchestrator/status/');
  assert.equal(config.resolveWsUrl('/ws/ml/orchestrator/'), '');
});

test('buildAppConfig respects VITE_API_BASE_URL and derives ws URL', () => {
  const config = buildAppConfig({
    VITE_API_BASE_URL: 'https://api.pulse-ai.org',
    VITE_DEFAULT_MODEL: 'Custom-Donor-Model'
  });
  assert.equal(config.apiBaseUrl, 'https://api.pulse-ai.org');
  assert.equal(config.defaultModel, 'Custom-Donor-Model');
  assert.equal(config.resolveApiUrl('/api/test'), 'https://api.pulse-ai.org/api/test');
  assert.equal(config.resolveWsUrl('/ws/test'), 'wss://api.pulse-ai.org/ws/test');
});

test('buildAppConfig respects dedicated VITE_WS_BASE_URL if set', () => {
  const config = buildAppConfig({
    VITE_API_BASE_URL: 'https://api.pulse-ai.org',
    VITE_WS_BASE_URL: 'wss://ws.pulse-ai.org'
  });
  assert.equal(config.resolveWsUrl('/ws/stream'), 'wss://ws.pulse-ai.org/ws/stream');
});

test('buildAppConfig normalizes trailing and leading slashes', () => {
  const config = buildAppConfig({
    VITE_API_BASE_URL: 'https://api.pulse-ai.org///'
  });
  assert.equal(config.resolveApiUrl('api/without-slash'), 'https://api.pulse-ai.org/api/without-slash');
});
