## Core project goal

This project is a blood donor platform with an AI agent based on tool-calling LLMs.

The AI agent must be fully dynamic and configurable. Tools, models, APIs, Python functions, policies, schemas, permissions, routing behavior, and execution rules must be added, removed, updated, or disabled through configuration files, not by changing core application code.

## Non-negotiable rule: no hardcoded tools

Never hardcode tool names, model names, API routes, thresholds, file paths, prompts, policies, enum values, permissions, or routing logic directly in code unless explicitly asked.

Before adding any new feature, ask:

> Can this be represented in config?

If yes, implement it through config.

## Required architecture

Use a config-driven tool registry.

The code should load tools from configuration, validate them, register them, and expose them to the AI agent dynamically.


## Tool configuration rule

Every tool must be defined in config with fields like:

```yaml
id: donor_eligibility_classifier
name: Donor Eligibility Classifier
type: model
enabled: true
description: Predicts whether a donor is eligible
input_schema:
  age: integer
  weight: number
  last_donation_date: string
output_schema:
  eligible: boolean
  reason: string
adapter: classification_model
entrypoint: models.eligibility.predict
permissions:
  - read_donor_profile
timeout_seconds: 10
```

The agent must discover and use tools from config. Do not manually import or register each tool in business logic.

## Adding a new tool

When adding a new tool, do not edit core agent code unless the platform needs a new generic adapter type.

Correct:

* Add or update a config file.
* Reuse existing registry, loader, executor, and adapter interfaces.
* Add tests proving the tool is loaded from config.

Incorrect:

* Adding `if tool_name == "..."`
* Adding hardcoded tool lists.
* Adding fixed model names in code.
* Adding static imports for specific tools in the agent loop.
* Adding special-case routing logic for one tool.

## Dynamic routing

Tool selection must be driven by metadata, schemas, permissions, and config.

Avoid:

```python
if task == "classify_donor":
    return donor_classifier()
```

## Configuration validation

All config must be validated at startup.

Invalid tool config should fail clearly with useful errors.

Required validation:

* unique tool id
* supported tool type
* valid input schema
* valid output schema
* valid adapter
* valid permissions
* timeout present
* enabled field present

## Security and safety

This is a healthcare-adjacent blood donor platform.

Do not expose private donor data unnecessarily.

Never hardcode secrets, tokens, API keys, credentials, or private URLs.

Use environment variables or secret managers.

Any AI-generated recommendation must be treated as decision support, not final medical judgment.

The system must preserve auditability:

* log which tool was called
* log tool version/config version
* log input/output metadata where safe
* avoid logging sensitive personal data

## Testing requirements

For every new tool or config change:

* Add config validation tests.
* Add registry loading tests.
* Add executor tests.
* Add failure-case tests for missing/invalid config.
* Confirm no hardcoded references were introduced.

Before finishing, search the diff for hardcoding risks:

* fixed tool names
* fixed model names
* fixed file paths
* special-case `if/else` routing
* duplicated config values
* direct secrets
* manually maintained tool lists

## Refactoring rule

If existing code is hardcoded, refactor toward config-driven behavior instead of adding more hardcoded logic.

When fixing bugs, do not patch around the issue with special cases. Find the generic configurable mechanism and fix that.

## Final response requirement

When completing a task, explain:

1. Which config files were added or changed.
2. Which core code paths were kept generic.
3. How a future tool can be added without code changes.
4. What tests were added or updated.
5. Whether any hardcoded values remain, and why.

## Critical instruction for Codex

If you are about to type a literal value into code, stop and decide whether it belongs in config. Prefer config unless the value is a true language/framework constant.

