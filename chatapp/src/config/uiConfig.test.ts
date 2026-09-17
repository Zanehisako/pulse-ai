import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePreference, readPreference, applyTheme } from '../hooks/useTheme.ts';
import { uiConfig } from './uiConfig.ts';

test('theme config declares light, dark, and system options', () => {
  const ids = uiConfig.theme.options.map(option => option.id);
  assert.deepStrictEqual(new Set(ids), new Set(['light', 'dark', 'system']));
  assert.ok(uiConfig.theme.palettes.light);
  assert.ok(uiConfig.theme.palettes.dark);
});

test('light and dark palettes expose the same token set', () => {
  const light = Object.keys(uiConfig.theme.palettes.light).sort();
  const dark = Object.keys(uiConfig.theme.palettes.dark).sort();
  assert.deepStrictEqual(light, dark);
});

test('normalizePreference falls back to the configured default for unknown values', () => {
  assert.strictEqual(normalizePreference('dark'), 'dark');
  assert.strictEqual(normalizePreference('nonsense'), uiConfig.theme.defaultPreference);
  assert.strictEqual(normalizePreference(null), uiConfig.theme.defaultPreference);
});

test('applyTheme applies palette tokens and dataset attribute', () => {
  globalThis.window = {
    matchMedia: () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} }),
    localStorage: { getItem: () => null, setItem: () => {} },
    addEventListener: () => {},
    removeEventListener: () => {}
  } as unknown as typeof window;
  const documentElement = {
    dataset: {} as Record<string, string>,
    style: { setProperty: () => {} } as unknown as CSSStyleDeclaration
  };
  globalThis.document = { documentElement } as unknown as Document;
  applyTheme('light');
  assert.strictEqual(documentElement.dataset.theme, 'light');
  applyTheme('dark');
  assert.strictEqual(documentElement.dataset.theme, 'dark');
});

test('readPreference returns the default when storage is unavailable', () => {
  globalThis.window = undefined as unknown as typeof window;
  assert.strictEqual(readPreference(), uiConfig.theme.defaultPreference);
});
