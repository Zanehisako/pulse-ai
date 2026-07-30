# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# Project Context
- Frame evaluations around thesis-readiness and defendability, not production deployment — the project is built on synthetic data and is academic in nature. Confidence: 0.75

# Academic Writing
- Use precise, defensible ML terminology — describe what was actually implemented, not what the term implies in literature (e.g., say "discrete-time hazard classifier" instead of "hazard model" unless Cox regression or survival analysis was actually used). Confidence: 0.70
- Keep conceptual definitions and explanations in the main thesis body (Chapters 1–3), not in the appendix; reserve the appendix for supplementary material like extra math, code details, and label-generation rules. Confidence: 0.75

# Diagrams
- Flow diagrams should go left-to-right; if the diagram is too large, make it vertical and flow top-to-bottom. Confidence: 0.70

# Orchestrator
- Production orchestrator toolset should be limited to: 11 predictive models + db_tool + llm_api only. Confidence: 0.70

