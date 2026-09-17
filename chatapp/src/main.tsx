import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './styles/main.css';
import { applyTheme, readPreference } from './hooks/useTheme';

applyTheme(readPreference());

const rootElement = document.getElementById('root');
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
