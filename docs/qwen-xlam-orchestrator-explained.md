# Qwen / xLAM Orchestrator in `backendMulti`

This document explains what the orchestrator is, how it works in production, how the notebooks in `ml-backend\notebooks\` relate to it, how it is trained, and where to modify it.

## Short answer

- **It is not a classic RAG pipeline.**
- **It is not a full autonomous agent loop by default.**
- **It is mainly a config-driven tool-and-model orchestrator with an optional local LLM planner.**
- In production, the planner is usually a **local GGUF model** loaded through `llama_cpp`, originally centered on **xLAM**, but the runtime can also **auto-detect Qwen GGUFs** and use them if present.
- If the local planner LLM is unavailable, the system falls back to **heuristic routing** plus configured external tools.

So the closest description is:

> **A config-driven ML tool orchestrator that can behave like a lightweight AI agent/planner when a local GGUF LLM is loaded.**

---

## Main production files

### Backend runtime

- `backendMulti\ml\orchestrator\service.py`
- `backendMulti\ml\api\views.py`
- `backendMulti\ml\core\startup.py`
- `backendMulti\ml\core\feature_extraction.py`
- `backendMulti\ml\core\feature_store.py`
- `backendMulti\ml\core\external_tools.py`
- `backendMulti\ml\config\config.json`
- `backendMulti\ml\config\orchestrator_models.json`
- `backendMulti\ml\config\db_reference_tables.json`
- `backendMulti\ml\config\feature_store_dynamic.json`
- `backendMulti\ml\models.py`

### Related notebooks

- `ml-backend\notebooks\qwen-orchestrator.ipynb`
- `ml-backend\notebooks\xlma-orchestrator.ipynb`
- `ml-backend\notebooks\SLM_PIOS.ipynb`

---

## What happens at runtime

## 1. Startup

At startup, `backendMulti\ml\core\startup.py`:

1. Loads ML model configs from the database into `ModelRegistry`.
2. Instantiates `DynamicXLAMOrchestrator`.
3. Syncs orchestrator model variants from DB.
4. Starts warmup for the planner LLM.

Key point:

- **ML predictive models** and **orchestrator variants** are database-backed.
- The orchestrator itself remains **generic** and reads config/catalog state dynamically.

---

## 2. Request entrypoint

Natural language requests go through:

- `POST /api/ml/predict/nl/`
- implemented in `backendMulti\ml\api\views.py` via `PredictNLView`

That view:

1. Sanitizes incoming features.
2. Optionally infers a forced model if the query explicitly names one.
3. Calls:

```python
orchestrator.run(
    query=query,
    provided_features=features,
    forced_model_ids=forced_model_ids,
    top_k=top_k,
)
```

If the client requests `text/event-stream`, the same run is exposed as a streaming planning/execution feed.

---

## 3. The orchestrator is config-driven

`DynamicXLAMOrchestrator` loads:

- external tool config from `backendMulti\ml\config\config.json`
- model variant catalog from DB and/or `backendMulti\ml\config\orchestrator_models.json`
- DB reference schemas from `backendMulti\ml\config\db_reference_tables.json`
- NL extraction + feature/domain mappings from `backendMulti\ml\config\feature_store_dynamic.json`

Important: external tools are validated at startup by `_load_external_tool_definitions()`.

Required fields include:

- `id`
- `name`
- `type`
- `enabled`
- `description`
- `input_schema`
- `output_schema`
- `adapter`
- `entrypoint`
- `permissions`
- `timeout_seconds`

That means the orchestrator is already close to the architecture you wanted: **tools are not hardcoded one-by-one in the planner loop**; they are loaded from config and executed through generic adapter branches.

---

## Is it RAG?

**Not by default.**

There is no standard “embed documents -> vector search -> retrieve chunks -> augment prompt” path in the orchestrator.

What it does have:

1. **Structured retrieval**
   - DB tool (`db_query`)
   - feature-store lookups
   - entity-context lookup from Postgres/SQLite/Feast-backed sources

2. **External retrieval**
   - `search` adapter

3. **Prompt grounding**
   - planner prompt includes tool metadata, schemas, descriptions, and examples
   - summarizer prompt includes execution outputs as ground truth

So this is better described as:

> **tool-augmented planning with structured retrieval**, not classic RAG.

If you want real RAG, you would need a new config-defined tool adapter for vector retrieval or document retrieval, then expose it in `config.json`.

---

## Is it an AI agent?

### Production backend

**Partly, but in a bounded way.**

The backend planner can:

1. read the tool/model menu,
2. generate a multi-step plan,
3. reference previous step outputs using `$steps.N.output.*`,
4. execute steps,
5. summarize results.

That is **agent-like orchestration**, but not an open-ended autonomous agent.

It does **not** normally:

- keep long-lived memory,
- freely invent tools,
- act in an infinite loop,
- do arbitrary external browsing,
- persist self-generated plans across sessions.

### Notebook experiments

`xlma-orchestrator.ipynb` contains stronger agent-style experiments:

- explicit multi-step mission loops,
- grammar-constrained JSON actions,
- tool execution history,
- repeated step-by-step reasoning.

Those notebooks are **prototypes**, not the production web backend path.

---

## How the production planner works

## A. Model selection and loading

The class name is still `DynamicXLAMOrchestrator`, but it can load more than xLAM:

- built-in default catalog is xLAM quantizations
- local `.gguf` files are auto-discovered
- discovered local Qwen GGUFs are recognized
- sidecar capability files can override metadata
- cached capability metadata is persisted

Current built-in catalog file:

- `backendMulti\ml\config\orchestrator_models.json`

Current default built-in variants:

- `xlam_7b_q6_k`
- `xlam_7b_q5_k_m`
- `xlam_7b_q4_k_m`
- `xlam_7b_q3_k_m`
- `xlam_7b_q2_k`

But tests show the runtime also supports local Qwen GGUF discovery:

- detects `qwen2-7b-instruct-q4_0.gguf`
- infers prompt mode `chat`
- can preserve selected xLAM variant when Qwen is added

So in practice:

- **xLAM is the default production family**
- **Qwen is supported if a compatible local GGUF exists**

### How the GGUF is loaded

`service.py` loads the LLM with:

- `huggingface_hub` for download
- `llama_cpp.Llama` for inference

It auto-detects hardware:

- Apple Metal
- NVIDIA CUDA
- CPU fallback

And chooses:

- `n_gpu_layers`
- `n_threads`
- `n_ctx`
- `n_batch`

from hardware plus env overrides.

---

## B. Prompt mode detection

The runtime inspects GGUF metadata and capability sidecars to determine whether a model should run in:

- **chat mode**
- **completion mode**

This matters because Qwen-style GGUFs often expose chat templates, while xLAM variants may behave better with completion prompts.

The runtime can also do a lightweight handshake/probe to check prompt behavior and cache the result.

Important consequence:

- **The backend is not tied to one exact prompt API.**
- It adapts to the discovered GGUF’s capabilities.

---

## C. Planning

When `run()` is called, the orchestrator:

1. normalizes query and features
2. may short-circuit to a configured direct response
3. extracts NL features
4. optionally waits for LLM warmup
5. filters allowed predictive models
6. applies forecast-horizon routing if needed
7. creates a plan

Planning happens in `_plan_execution()`.

There are **two planning modes**:

### 1. LLM planning

If a local GGUF planner is ready:

- it builds a planning prompt from config
- includes tool descriptions, schemas, examples, and workflow examples
- asks the local LLM to return JSON with reasoning + tool calls

The planner prompt template lives in:

- `backendMulti\ml\config\config.json` -> `orchestrator.prompts.planning`

Expected output shape:

```json
{
  "reasoning": "...",
  "tool_calls": [
    {
      "name": "db_tool",
      "arguments": {},
      "reasoning": "..."
    }
  ]
}
```

### 2. Heuristic fallback planning

If the LLM is unavailable or returns no usable plan:

- `_fallback_plan()` handles routing
- it uses extracted features
- `route_models()` ranks predictive models from metadata/examples/features
- it can route to:
  - predictive models
  - DB tool
  - stockout hybrid tool
  - simulation tool
  - external LLM API

So the system is **LLM-first when possible, heuristics-first when necessary**.

---

## D. Step execution

After planning, each step is executed in order.

The planner can reference previous outputs with:

- `$request.query`
- `$request.features.*`
- `$last.output.*`
- `$steps.1.output.*`

This is the mechanism that enables chained workflows like:

1. detect hospital stockout risk
2. extract worst blood group
3. fetch eligible donors for that blood group
4. rank donors

That is why the system behaves more like a **planner/executor** than a simple single-shot classifier.

---

## E. Result summarization

After execution:

- if planner mode is not LLM-based, the backend produces deterministic summaries/markdown tables
- if planner mode is LLM-based and the LLM is available, it runs a second summarization prompt

Summarizer template:

- `backendMulti\ml\config\config.json` -> `orchestrator.prompts.summarizer`

This summarizer is grounded only on factual execution results, not free-form memory.

---

## What tools exist today

The production orchestrator supports these adapter families:

- `db_query`
- `search`
- `llm_api`
- `simulation_query`
- `stockout_hybrid`

Execution is generic through `_execute_external_tool()`, which dispatches by adapter.

Configured tool examples live inside:

- `backendMulti\ml\config\config.json` -> `external_tools`

The most important configured tool in current routing is:

- `stockout_hybrid`

This tool is heavily used for hospital stockout / shortage workflows.

---

## How predictive models are selected

Predictive models are **not selected by hardcoded `if query contains X use model Y`** in the main path.

Instead:

1. `extract_nl_features()` parses values from text
2. `route_models()` scores candidate models by:
   - description overlap
   - alias overlap
   - feature-name overlap
   - good example overlap
   - bad example penalty
3. feature-store and entity resolution fill missing inputs

That means the predictive routing is:

> **metadata-driven + feature-driven + example-driven**

not pure hardcoding.

---

## How feature completion works

This part is important.

The orchestrator can route to a model even if the user does not provide every raw feature directly.

`backendMulti\ml\core\feature_store.py`:

- resolves direct request features
- resolves Feast online features when configured
- resolves entity context from DB-backed tables
- merges defaults and external context

So for some models, IDs or partial inputs are enough because the system can enrich them.

This is **not RAG**; it is **feature enrichment / online feature resolution**.

---

## How DB retrieval works

The DB tool is not a generic unrestricted SQL free-for-all in the planner prompt.

It is shaped by:

- `backendMulti\ml\config\db_reference_tables.json`

That file defines:

- logical tables (`donors`, `hospitals`)
- aliases
- columns
- seed sources
- field aliases
- query defaults
- NL query tokens
- filtering behavior

There is also a bootstrap command:

```powershell
python manage.py bootstrap_orchestrator_db
```

This seeds Postgres tables, or falls back to SQLite if Postgres bootstrap fails.

---

## How direct responses work

Some questions do not need tools at all.

`config.json` contains:

- `orchestrator.dynamic_planning.direct_response`

Example:

- conceptual donor category questions are answered directly from config markdown

So not every user question invokes the planner or tools.

---

## How workflow examples work

This is one of the most important parts for behavior.

`config.json` includes:

- `orchestrator.dynamic_planning.workflow_examples`

Examples already encoded:

1. stockout -> donor outreach
2. stockout compare hospitals
3. stockout rank hospitals
4. explicit blood-group donor outreach

These examples are used in two ways:

- to enrich the LLM planning prompt
- to provide heuristic fallback plans when LLM planning is unavailable

So current behavior is not purely “the model figures everything out”.
It is strongly shaped by **configured workflows**.

---

## What `qwen-orchestrator.ipynb` does

This notebook is **not the production backend**.
It is a separate experiment to fine-tune a small Qwen model for tool selection/orchestration.

### What it trains

Base model:

- `Qwen/Qwen2.5-1.5B-Instruct`

Training style:

- LoRA / PEFT
- 4-bit quantized base model
- SFT (`trl.SFTTrainer`)

### What data it uses

It first **generates a synthetic dataset**:

- tool names are randomized in many styles
- argument names are randomized
- distractor tools are injected
- target tool is derived from domain/query pairs

The goal is to make the model rely on:

- tool descriptions
- input schemas
- query meaning

instead of memorizing fixed names.

That is actually a smart dataset design for tool-selection robustness.

### What the model learns to output

The training target is a JSON plan like:

```json
{
  "plan_type": "single_step",
  "tool_use": {
    "tool_name": "...",
    "arguments": {...}
  }
}
```

### Training config in notebook

Observed notebook choices:

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- 4-bit quantization
- LoRA rank `r=32`
- `lora_alpha=64`
- dropout `0.05`
- one epoch in the visible training cell
- batch size `2`
- grad accumulation `8`
- learning rate `2e-4`

### What it is for

This notebook is a **small tool-selection planner experiment**.

It is useful if you want:

- a lightweight learned planner
- LoRA adapters instead of full GGUF deployment
- custom fine-tuning for your own tool schemas

### What it is not

- it is not directly wired into `backendMulti\ml\orchestrator\service.py`
- it is not the current production path
- it is not the current xLAM GGUF runtime

---

## What `xlma-orchestrator.ipynb` does

This notebook is also **experimental**, but much closer in spirit to the production backend.

It shows:

- local GGUF loading via `llama_cpp`
- xLAM as the planning model
- grammar-constrained JSON outputs
- multi-step planning and execution
- autonomous tool loop experiments

### Main experimental pattern

It defines:

- a model registry
- a universal inference wrapper
- a GGUF model wrapper
- an `MLOrchestrator`
- later, a stronger autonomous agent loop with environment tools

### Key traits

1. **Grammar-constrained JSON**
   - uses `LlamaGrammar.from_json_schema(...)`
   - forces strict plan JSON

2. **xLAM GGUF inference**
   - downloads xLAM GGUF from Hugging Face
   - runs via `llama_cpp.Llama`

3. **Multi-step plans**
   - step list with reasoning and arguments

4. **Agent-style mission loop**
   - environment state
   - tool history
   - repeated reasoning/action cycles

### Important difference from production

The notebook uses grammar constraints heavily.

The current production `service.py`:

- defines a plan schema
- parses planner JSON
- but does **not currently enforce `LlamaGrammar` during planner generation**

So the notebook is in some ways **stricter** than the current backend runtime.

---

## What `SLM_PIOS.ipynb` is

This notebook is a **separate function-calling fine-tuning example** for Phi-3.5-mini using synthetic data and LoRA.

It is not the active backend orchestrator.

It is relevant only as:

- another experimentation branch for learned tool calling
- evidence that the repo explored small-model fine-tuning approaches

---

## So how is it “trained” today?

## Production backend

The production backend orchestrator is **mostly not trained inside this repo**.

For the live planner path, it usually relies on:

- pre-trained GGUF models such as xLAM
- optionally auto-detected local Qwen GGUFs

That means:

- the backend itself is mainly **configured and prompted**
- not trained end-to-end here

## Notebook experiments

Training exists in notebooks, especially:

- `qwen-orchestrator.ipynb`
- `SLM_PIOS.ipynb`

Those train LoRA adapters for tool selection / function calling on synthetic datasets.

So the honest summary is:

> **Production backend orchestration is prompt-driven + config-driven over pretrained/local GGUFs.**
>
> **Repo-local training exists mainly in notebook experiments, not as the deployed backend pipeline.**

---

## Is Qwen used in production right now?

**Potentially, yes, but not as the default built-in family.**

Current behavior:

- default catalog is xLAM-based
- local GGUF discovery can detect Qwen
- tests verify Qwen metadata, chat mode, and stable coexistence with xLAM

So if you drop a compatible Qwen GGUF in the orchestrator model directory, the backend can surface and use it.

But the naming, env vars, and default catalog still reflect the historical xLAM-first design.

---

## How to modify it safely

## 1. Change planner behavior

Edit:

- `backendMulti\ml\config\config.json`

Most important sections:

- `orchestrator.prompts.planning`
- `orchestrator.prompts.summarizer`
- `orchestrator.dynamic_planning.prefer_planner_terms`
- `orchestrator.dynamic_planning.workflow_examples`
- `orchestrator.dynamic_planning.row_scoring`
- `orchestrator.dynamic_planning.direct_response`
- `orchestrator.fallback_order`
- `orchestrator.planning_limits`

Use this if you want to change:

- when the LLM planner is preferred
- how workflows chain tools
- how summaries are written
- which fallback tool comes first

---

## 2. Add or modify external tools

Edit:

- `backendMulti\ml\config\config.json` -> `external_tools`

Each tool must define:

- id
- name
- type
- enabled
- description
- input_schema
- output_schema
- adapter
- entrypoint
- permissions
- timeout_seconds

If your tool fits an existing adapter family, prefer config-only changes.

If you need a truly new adapter type, then you must extend:

- validation in `_load_external_tool_definitions()`
- execution in `_execute_external_tool()`

---

## 3. Add/change planner model variants

Edit:

- `backendMulti\ml\config\orchestrator_models.json`

or use API / DB-backed catalog updates via:

- `/api/ml/orchestrator/models/`
- `/api/ml/orchestrator/models/selected/`
- `/api/ml/orchestrator/models/<model_id>/download/`

Variant fields include:

- `id`
- `name`
- `description`
- `repo_id`
- `filename`
- `size_mb`
- `size_bytes`
- `min_ram_gb`
- `min_vram_mb`
- `n_ctx`
- `n_batch`

---

## 4. Change DB retrieval behavior

Edit:

- `backendMulti\ml\config\db_reference_tables.json`

Use this to change:

- which tables exist logically
- aliases
- field mappings
- seed sources
- allowed query defaults
- NL intent mapping

---

## 5. Change feature extraction / entity resolution

Edit:

- `backendMulti\ml\config\feature_store_dynamic.json`

Use this to change:

- regex patterns for age/BMI/hospital IDs/etc.
- component aliases
- table mappings
- feature-match table mappings
- entity aliases

---

## 6. Change predictive model metadata

Edit database-backed `MLModelConfig` rows or the loader source that populates them.

Those rows shape:

- model descriptions
- examples
- defaults
- feature lists
- routing scores

If model metadata is weak, the planner and fallback router become weak too.

---

## 7. Switch xLAM to Qwen cleanly

If you want Qwen to be the real primary planner, do all of these:

1. Put the Qwen GGUF in the orchestrator model directory.
2. Add it to the orchestrator catalog or rely on auto-discovery.
3. Select it through the orchestrator model APIs or DB.
4. Confirm its `prompt_mode` resolves correctly.
5. Test planning and summarization outputs for your workflow examples.

Recommended files/API:

- `backendMulti\ml\config\orchestrator_models.json`
- `/api/ml/orchestrator/models/`
- `/api/ml/orchestrator/models/selected/`
- `/api/ml/orchestrator/warmup/`

Note:

- Qwen chat-template behavior differs from xLAM completion behavior.
- The runtime already tries to adapt, but prompt quality can still shift.

---

## 8. If you want a stricter planner

The notebook prototype uses JSON grammar constraints.

If you want production planning to be more stable, one strong improvement would be:

- add grammar-constrained JSON generation in `service.py`

Right now production parses JSON heuristically from LLM output.
That works, but grammar constraints would reduce malformed planner responses.

---

## 9. If you want true RAG

Add a new config-defined tool adapter, for example:

- `vector_search`
- `document_retrieval`

Then:

1. validate it in external tool config loading
2. execute it through a generic adapter branch
3. expose schemas/examples in tool metadata
4. include it in planner config

Do **not** hardcode “if query contains policy then call rag_tool”.
Let the planner use the configured metadata.

---

## Current hardcoded parts that still exist

The system is mostly config-driven, but some hardcoded defaults remain in code:

1. built-in default xLAM catalog in `service.py`
2. supported adapter family names in `_load_external_tool_definitions()`
3. some env var names and fallback path logic
4. some stockout-routing helper behavior in generic code

These are not ideal if your goal is total config-only extensibility.

Most important limitation:

> Adding a brand-new adapter type still requires code changes.

But adding a new tool **within an existing adapter family** is already mostly config-only.

---

## What I would change if you want this cleaner

### High priority

1. move default xLAM catalog out of code entirely
2. move supported adapter registry into config or a generic plugin registry
3. add grammar-constrained planner output in production
4. separate “planner LLM” naming from xLAM-specific historical names

### Medium priority

1. add explicit capability metadata per GGUF in catalog/sidecar
2. formalize Qwen as a first-class orchestrator variant in config
3. add real RAG adapter if document retrieval is needed

---

## Practical modification checklist

If you want to modify behavior, start here:

### Change orchestration logic without code changes

- `backendMulti\ml\config\config.json`
- `backendMulti\ml\config\orchestrator_models.json`
- `backendMulti\ml\config\db_reference_tables.json`
- `backendMulti\ml\config\feature_store_dynamic.json`

### Change runtime code

- `backendMulti\ml\orchestrator\service.py`
- `backendMulti\ml\core\external_tools.py`
- `backendMulti\ml\core\feature_extraction.py`
- `backendMulti\ml\core\feature_store.py`

### Change web/API behavior

- `backendMulti\ml\api\views.py`

### Change training experiments

- `ml-backend\notebooks\qwen-orchestrator.ipynb`
- `ml-backend\notebooks\xlma-orchestrator.ipynb`
- `ml-backend\notebooks\SLM_PIOS.ipynb`

---

## Final conclusion

The current `backendMulti` orchestrator is:

- **not classic RAG**
- **not a fully autonomous general agent**
- **a config-driven planner/executor over predictive models and external tools**
- **LLM-assisted when a local GGUF is available**
- **heuristic fallback-driven when it is not**

### xLAM

- historical default planner family
- production-oriented GGUF path
- loaded locally with `llama_cpp`

### Qwen

- supported through local GGUF discovery in backend tests/runtime
- also explored in notebooks through LoRA fine-tuning
- not the hardcoded default production family today

### Training

- **production path:** mostly uses pre-trained GGUFs + prompts + config
- **notebook path:** trains small planners/function-callers with synthetic data and LoRA

If your goal is to evolve this into a fully dynamic AI-agent platform, the current base is usable, but the remaining xLAM-specific defaults and adapter lists should be pushed further into config/plugin registries.
