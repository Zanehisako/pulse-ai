# Architecture & Deep Dive: PulseAI Platform

This document provides a technical deep-dive into the architectural components of the **PulseAI Platform**.

---

## 1. High-Level Design Principles

1. **Config-Driven Tool Architecture**:
   - The platform strictly adheres to zero hardcoding rules defined in `AGENTS.md`.
   - AI tools, ML models, schemas, prediction routes, and execution rules are configured via JSON/YAML manifests.
2. **Decoupled Engine Interfaces**:
   - The simulation engine (`simulation_studio`) operates as an independent FastAPI service.
   - The AI agent orchestrator (`backendMulti`) consumes simulation telemetry and feature stores to execute autonomous decisions.
3. **Enterprise MLOps Lineage**:
   - Feature engineering definitions are centralized in **Feast Feature Store** (`ml-backend/feature_repo`).
   - Training runs and model lifecycle artifacts are tracked in **MLflow** with a PostgreSQL backend store.

---

## 2. Component Breakdown

### A. Config-Driven AI Agent Engine (`backendMulti`)
The Django backend dynamically registers AI capabilities at runtime:
- **Registry & Loader**: Scans tool configuration paths at startup (`manage.py import_ml_config`).
- **Adapter Layer**: Adapts tool interfaces (classification, regression, forecasting, RL policy calls) into standard LLM tool schema representations.
- **Security & Authorization**: Every request is validated against Keycloak JWT Bearer tokens and per-tool permission specs.

### B. MLOps Core (`ml-backend`)
- **Feast Feature Store**: Defines entities (`donor`, `hospital`, `supply_center`) and feature views (`blood_stock_features`, `donation_history_features`).
- **Model Catalog**: Managed via `ml-backend/config/sota_model_catalog.json`. Models (CatBoost, LightGBM, XGBoost, GGUF/xLAM) are versioned and cataloged with metric metadata.
- **MLflow Tracking Server**: Listens on port `8889` backed by PostgreSQL database storage.

### C. Digital Twin Simulation Engine (`simulation_studio`)
FastAPI application (Port `8010`) simulating regional blood bank supply networks:
- **PPO & SAC RL Controllers**: Trained reinforcement learning agents for inventory replenishment and emergency re-allocation.
- **Ablation & Sensitivity Analysis**: Integrated scripts (`run_horizon_ablation.py`, `run_sensitivity_sweep.py`) for evaluating policy resilience under demand spikes.
- **Digital Twin Telemetry**: Real-time WebSocket broadcasting of state changes, stockout predictions, and expiration warnings.

### D. Infrastructure & Security Services (`infrastructure`)
- **PostgreSQL**: Multi-schema database hosting application data and MLflow run state.
- **Keycloak**: OpenID Connect provider (port `8080`) issuing and validating JWT access tokens.
- **pgAdmin**: Web GUI database manager for operators.
- **SonarQube**: Static code quality scanner enforcing security and coverage bounds.

---

## 3. Tool Discovery & Execution Flow

```
[Agent / User Prompt]
        │
        ▼
[Django Orchestrator]
        │
        ├──► Discovers Tools in Config (`prediction_runtime.json`)
        ├──► Validates Permissions & Schemas
        │
        ├──► [Option A: Query Feast Feature Store]
        ├──► [Option B: Call ML Model via Adapter]
        └──► [Option C: Invoke Digital Twin Forecast Endpoint]
        │
        ▼
[Structured Response / Decision Support Output]
```
