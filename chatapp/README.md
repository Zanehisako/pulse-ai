# PulseAI Orchestrator — Streaming ChatApp Frontend

This directory (`chatapp/`) contains the standalone, dedicated frontend web application for conversing with the PulseAI Orchestrator agent in real time.

---

## 🏛️ Application Structure

```
chatapp/
├── index.html            # Main React 18 single-page application document
├── src/
│   └── styles/
│       └── main.css      # Vanilla CSS Design System (Dark Glassmorphism, Micro-animations)
└── README.md             # Documentation
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
- **ChatApp Interface**: `http://localhost:8000/chat/`
- **Orchestrator Status API**: `http://localhost:8000/api/ml/orchestrator/status/`
- **Prediction Stream API**: `http://localhost:8000/api/ml/predict/nl/?stream=true`
