# PulseAI Orchestrator — Streaming ChatApp Frontend

This directory (`chatapp/`) contains the standalone, dedicated frontend web application for conversing with the PulseAI Orchestrator agent in real time.

---

## 🏛️ Application Structure

```
chatapp/
├── public/
│   ├── _headers          # Cloudflare Pages security headers
│   └── _redirects        # Cloudflare Pages SPA client-side routing fallback
├── src/
│   ├── config/
│   │   ├── appConfig.ts  # Config-driven API / WebSocket resolver
│   │   └── index.ts
│   ├── components/       # React UI components
│   ├── hooks/            # Streaming & WebSocket hooks
│   ├── services/         # Typed API clients
│   ├── styles/
│   │   └── main.css      # Vanilla CSS Design System (Dark Glassmorphism)
│   └── types/            # TypeScript interfaces
├── .env.example          # Environment variables template
├── wrangler.toml         # Cloudflare Pages configuration
├── index.html            # Main React 18 single-page application document
├── package.json          # Scripts & dependencies
├── tsconfig.json         # TypeScript configuration
├── vite.config.ts        # Vite build & local dev proxy setup
└── README.md             # Documentation
```

---

## ☁️ Cloudflare Pages Hosting

This frontend is configured for deployment on **Cloudflare Pages**. See the full guide at [`docs/cloudflare-pages-deployment.md`](file:///Users/a1234/Documents/pulse-ai/docs/cloudflare-pages-deployment.md).

### Quick Setup:
1. Connect this git repository to Cloudflare Pages in the Cloudflare Dashboard.
2. Set **Root directory**: `chatapp`
3. Set **Framework preset**: `Vite` (Build command: `npm run build`, Output directory: `dist`)
4. Environment Variables (optional, defaults are pre-configured):
   - `MODAL_LLM_API_URL`: `https://yassinetakiko--pulseai-vllm-backend-serve.modal.run`
   - `MODAL_LLM_MODEL`: `Qwen/Qwen2.5-7B-Instruct`
   - `VITE_API_BASE_URL`: *(Leave empty to use built-in Cloudflare Pages Functions edge proxy)*
5. Deploy!

### Useful Scripts:
```bash
# Run locally with dev server & proxy
npm run dev

# Run config resolver tests
npm test

# Build production assets for Cloudflare Pages
npm run build

# Preview build with Cloudflare Wrangler locally
npm run pages:preview

# Deploy directly to Cloudflare Pages via CLI
npm run pages:deploy
```

---

## 🚀 Key Features

1. **Real-time SSE Streaming**: Connects directly to `POST /api/ml/predict/nl/?stream=true` using HTTP Server-Sent Events (`text/event-stream`).
2. **Markdown Stream Renderer**: Renders live tokens as formatted Markdown tables, code blocks (`tokyo-night-dark` syntax highlighting), and callouts using `marked.js`.
3. **Orchestrator Reasoning & Plan Drawer**:
   - **Reasoning Box**: Shows the LLM's step-by-step reasoning.
   - **Execution Plan Drawer**: Multi-step tool workflow visualization (Step #, Tool Name, Input Arguments, CLI Command, Execution Status).
   - **Telemetry Tag**: Stream duration (ms) and tool execution statistics.

---

## 🌐 Access Endpoints

When the PulseAI Django backend is running (`./run.sh start` or `python backendMulti/manage.py runserver`):
- **Local ChatApp Interface**: `http://localhost:5173/` (Vite dev) or `http://localhost:8000/chat/` (Django static)
- **Orchestrator Status API**: `/api/ml/orchestrator/status/`
- **Prediction Stream API**: `/api/ml/predict/nl/?stream=true`
- **WebSocket Endpoint**: `/ws/ml/orchestrator/`
