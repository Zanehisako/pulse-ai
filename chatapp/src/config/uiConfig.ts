const configuredUi = {
  brand: 'PulseAI',
  subtitle: 'Blood donation assistant',
  welcomeTitle: 'How can we help today?',
  welcomeDescription: 'Explore donor eligibility, blood supply, and hospital inventory with your AI assistant.',
  labels: {
    shortcuts: 'Shortcuts',
    newChat: 'New chat',
    theme: 'Appearance',
    activity: 'Agent activity',
    clearLogs: 'Clear logs',
    emptyLogs: 'Activity will appear here when you send a message.',
    reasoning: 'Planning notes',
    execution: 'Execution details',
    send: 'Send message',
    stop: 'Stop response',
    placeholder: 'Ask about donors, inventory, or blood supply…',
    disclaimer: 'AI decision support only. Verify recommendations with a qualified healthcare professional.',
    ready: 'Available',
    offline: 'Disconnected',
    assistant: 'PulseAI',
    user: 'You'
  },
  theme: {
    storageKey: 'pulseai.appearance',
    defaultPreference: 'system',
    systemPreference: 'system',
    mediaQuery: '(prefers-color-scheme: dark)',
    light: 'light',
    dark: 'dark',
    palettes: {
      light: {
        'bg-dark': '#fafafa', 'bg-surface': '#ffffff', 'bg-card': '#f4f4f5',
        'bg-card-hover': '#ededee', 'bg-input': '#ffffff', 'border-color': '#e4e4e7',
        'border-bright': '#b4b4bd', 'text-main': '#242429', 'text-muted': '#62626d',
        'text-dim': '#71717a', 'accent-crimson': '#a83c50', 'accent-blood': '#8f3042',
        'accent-soft': '#f8ecef', 'accent-emerald': '#24724f', 'accent-amber': '#906011',
        'on-accent': '#ffffff'
      },
      dark: {
        'bg-dark': '#19191c', 'bg-surface': '#202024', 'bg-card': '#28282d',
        'bg-card-hover': '#313137', 'bg-input': '#242428', 'border-color': '#38383f',
        'border-bright': '#666671', 'text-main': '#ededf0', 'text-muted': '#b0b0ba',
        'text-dim': '#9999a5', 'accent-crimson': '#df8b9c', 'accent-blood': '#ed9cad',
        'accent-soft': '#392830', 'accent-emerald': '#86caa4', 'accent-amber': '#e2bb77',
        'on-accent': '#25181c'
      }
    },
    options: [
      { id: 'light', label: 'Light' },
      { id: 'dark', label: 'Dark' },
      { id: 'system', label: 'System' }
    ]
  },
  suggestions: [
    {
      id: 'eligibility',
      icon: 'fa-solid fa-droplet',
      title: 'Donor eligibility',
      desc: 'Understand donation criteria',
      query: 'What information is needed to check donor eligibility?'
    },
    {
      id: 'stockout',
      icon: 'fa-solid fa-chart-line',
      title: 'Blood supply risk',
      desc: 'Explore inventory and demand',
      query: 'What information is needed to predict blood supply stockout risk for a hospital?'
    },
    {
      id: 'inventory',
      icon: 'fa-solid fa-hospital',
      title: 'Hospital inventory',
      desc: 'Check current blood stock levels',
      query: 'Help me check hospital blood inventory. What details do you need?'
    }
  ]
};
export { configuredUi as uiConfig };
