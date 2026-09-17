import { useEffect, useState } from 'react';
import { uiConfig } from '../config/uiConfig.ts';

const config = uiConfig.theme;

export function normalizePreference(value: string | null): string {
  return config.options.some(option => option.id === value) ? value! : config.defaultPreference;
}

export function readPreference(): string {
  try {
    return normalizePreference(window.localStorage.getItem(config.storageKey));
  } catch {
    return config.defaultPreference;
  }
}

export function applyTheme(preference: string): void {
  const dark = window.matchMedia(config.mediaQuery).matches;
  const resolved = preference === config.systemPreference ? (dark ? config.dark : config.light) : preference;
  const palette = config.palettes[resolved as keyof typeof config.palettes];
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
  Object.entries(palette).forEach(([token, value]) => {
    document.documentElement.style.setProperty(`--${token}`, value);
  });
}

export function useTheme() {
  const [preference, updatePreference] = useState(readPreference);

  useEffect(() => {
    const media = window.matchMedia(config.mediaQuery);
    const update = () => applyTheme(preference);
    update();
    media.addEventListener('change', update);
    const sync = (event: StorageEvent) => {
      if (event.key === config.storageKey || event.key === null) {
        updatePreference(normalizePreference(event.newValue));
      }
    };
    window.addEventListener('storage', sync);
    return () => {
      media.removeEventListener('change', update);
      window.removeEventListener('storage', sync);
    };
  }, [preference]);

  const setPreference = (value: string) => {
    const next = normalizePreference(value);
    updatePreference(next);
    applyTheme(next);
    try {
      window.localStorage.setItem(config.storageKey, next);
    } catch {}
  };

  return { preference, setPreference };
}
