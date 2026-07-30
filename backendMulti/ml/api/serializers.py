from rest_framework import serializers


class EmptySerializer(serializers.Serializer):
    pass


class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField()


class InitializingResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField(required=False, allow_blank=True)


class PredictRequestSerializer(serializers.Serializer):
    model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )
    features = serializers.DictField(required=False, default=dict)
    inputs = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list,
    )
    query = serializers.CharField(required=False, allow_blank=True, default="")
    prediction_timeframe = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )
    forecast_timeframe = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )
    prediction_horizon_days = serializers.FloatField(
        required=False, allow_null=True, default=None
    )
    forecast_horizon_days = serializers.FloatField(
        required=False, allow_null=True, default=None
    )
    horizon_days = serializers.FloatField(required=False, allow_null=True, default=None)

    def validate(self, data):
        features = data.get("features") or {}
        inputs = [
            dict(row)
            for row in (data.get("inputs") or [])
            if isinstance(row, dict)
        ]
        data["input_rows"] = inputs
        data["features"] = dict(features)
        for key in (
            "prediction_timeframe",
            "forecast_timeframe",
            "prediction_horizon_days",
            "forecast_horizon_days",
            "horizon_days",
        ):
            value = data.get(key)
            if value not in (None, ""):
                data["features"].setdefault(key, value)
                for row in data["input_rows"]:
                    row.setdefault(key, value)
        return data


class NLPredictRequestSerializer(serializers.Serializer):
    query = serializers.CharField(required=False, allow_blank=True)
    text = serializers.CharField(required=False, allow_blank=True)
    prompt = serializers.CharField(required=False, allow_blank=True)
    features = serializers.DictField(default=dict)
    model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )
    top_k = serializers.IntegerField(
        required=False, min_value=1, max_value=50, allow_null=True, default=None
    )
    prediction_timeframe = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )
    forecast_timeframe = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )
    prediction_horizon_days = serializers.FloatField(
        required=False, allow_null=True, default=None
    )
    forecast_horizon_days = serializers.FloatField(
        required=False, allow_null=True, default=None
    )
    horizon_days = serializers.FloatField(required=False, allow_null=True, default=None)

    def validate(self, data):
        q = data.get("query") or data.get("text") or data.get("prompt")
        if not q or not q.strip():
            raise serializers.ValidationError(
                "Provide a non-empty query (or text/prompt)."
            )
        data["query"] = q.strip()
        features = dict(data.get("features") or {})
        for key in (
            "prediction_timeframe",
            "forecast_timeframe",
            "prediction_horizon_days",
            "forecast_horizon_days",
            "horizon_days",
        ):
            value = data.get(key)
            if value not in (None, ""):
                features.setdefault(key, value)
        data["features"] = features
        return data


class MLIndexEndpointsSerializer(serializers.Serializer):
    health = serializers.CharField()
    models = serializers.CharField()
    predict_nl = serializers.CharField()
    predict_stockout_hybrid = serializers.CharField(required=False)
    orchestrator_status = serializers.CharField()
    config_runtime = serializers.CharField(required=False)
    orchestrator_models = serializers.CharField(required=False)
    orchestrator_select = serializers.CharField(required=False)
    orchestrator_warmup = serializers.CharField(required=False)
    tools = serializers.CharField(required=False)


class MLIndexResponseSerializer(serializers.Serializer):
    service = serializers.CharField()
    version = serializers.CharField()
    ml_ready = serializers.BooleanField()
    admin = serializers.CharField()
    endpoints = MLIndexEndpointsSerializer()


class HealthResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField(required=False, allow_blank=True)
    models_total = serializers.IntegerField(required=False)
    models_loaded = serializers.IntegerField(required=False)
    orchestrator = serializers.JSONField(required=False)
    config_runtime = serializers.JSONField(required=False)
    orchestrator_config = serializers.JSONField(required=False)


class ModelSummarySerializer(serializers.Serializer):
    model_id = serializers.CharField()
    aliases = serializers.ListField(child=serializers.CharField())
    slug = serializers.CharField(required=False)
    description = serializers.CharField()
    model_type = serializers.CharField()
    status = serializers.CharField()
    load_error = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    feature_names = serializers.ListField(child=serializers.CharField())
    feature_info = serializers.JSONField(required=False)
    examples = serializers.ListField(child=serializers.JSONField(), required=False)
    defaults = serializers.JSONField(required=False)
    predict_endpoint = serializers.CharField(required=False)
    generic_predict_endpoint = serializers.CharField(required=False)


class ModelStatsSummarySerializer(serializers.Serializer):
    model_id = serializers.CharField()
    identifier = serializers.CharField()
    stats = serializers.JSONField()
    made_at = serializers.CharField(required=False, allow_blank=True)


class ModelListResponseSerializer(serializers.Serializer):
    models = ModelSummarySerializer(many=True)


class RemoveModelResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()


class ModelsStatsListResponseSerializer(serializers.Serializer):
    models = ModelStatsSummarySerializer(many=True)


class ModelDetailResponseSerializer(ModelSummarySerializer):
    file_path = serializers.CharField()


class ModelStatsDetailResponseSerializer(ModelStatsSummarySerializer):
    file_path = serializers.CharField()


class RemoveModelStatsResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()


class ConfigReloadResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    changed = serializers.BooleanField(required=False)
    file_sync = serializers.JSONField(required=False)
    runtime_config_changed = serializers.BooleanField(required=False)
    orchestrator_catalog_changed = serializers.BooleanField(required=False)
    external_tool_reload = serializers.JSONField(required=False)
    config_hot_reload = serializers.JSONField(required=False)
    models_total = serializers.IntegerField(required=False)
    models_loaded = serializers.IntegerField(required=False)
    config_runtime = serializers.JSONField(required=False)
    orchestrator_config = serializers.JSONField(required=False)


class RuntimeConfigStatusSerializer(serializers.Serializer):
    source = serializers.CharField()
    config_key = serializers.CharField()
    last_source_error = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    last_reload_reason = serializers.CharField()
    last_reload_changed = serializers.BooleanField()
    last_reload_at = serializers.FloatField(required=False, allow_null=True)
    reload_count = serializers.IntegerField()
    models_total = serializers.IntegerField(required=False)
    active_models_total = serializers.IntegerField(required=False)
    model_config_file = serializers.CharField(required=False, allow_blank=True)
    last_file_sync = serializers.JSONField(required=False, allow_null=True)
    config_hot_reload = serializers.JSONField(required=False)
    last_updated_at = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    db_ready = serializers.BooleanField(required=False)


class ControlStatusResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    ml_ready = serializers.BooleanField()
    models_total = serializers.IntegerField(required=False)
    models_loaded = serializers.IntegerField(required=False)
    models = serializers.ListField(child=serializers.JSONField(), required=False)
    orchestrator = serializers.JSONField(required=False)
    orchestrator_models = serializers.ListField(
        child=serializers.JSONField(),
        required=False,
    )
    selected_model_id = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
    )
    active_model_id = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
    )
    config_runtime = serializers.JSONField(required=False)
    orchestrator_config = serializers.JSONField(required=False)
    refresh_error = serializers.CharField(required=False, allow_blank=True)


class RuntimeConfigRequestSerializer(serializers.Serializer):
    models = serializers.ListField(child=serializers.DictField(), default=list)


class RuntimeConfigMutationResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    runtime_config_changed = serializers.BooleanField()
    orchestrator_catalog_changed = serializers.BooleanField()
    models_total = serializers.IntegerField()
    models_loaded = serializers.IntegerField()
    config_runtime = RuntimeConfigStatusSerializer()
    orchestrator_config = serializers.JSONField()


class OrchestratorVariantSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField(allow_blank=True)
    repo_id = serializers.CharField()
    filename = serializers.CharField()
    n_ctx = serializers.IntegerField()
    n_batch = serializers.IntegerField()


class ExternalFallbackStatusSerializer(serializers.Serializer):
    enabled = serializers.BooleanField()
    order = serializers.ListField(child=serializers.CharField())
    db_enabled = serializers.BooleanField()
    db_tool_name = serializers.CharField()
    db_csv_path = serializers.CharField()
    db_csv_exists = serializers.BooleanField()
    db_sqlite_configured = serializers.BooleanField()
    search_enabled = serializers.BooleanField()
    search_api_configured = serializers.BooleanField()
    llm_api_configured = serializers.BooleanField()


class OrchestratorStatusResponseSerializer(serializers.Serializer):
    llm_ready = serializers.BooleanField()
    llm_n_ctx = serializers.IntegerField(required=False, allow_null=True)
    llm_attempted = serializers.BooleanField()
    llm_error = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    planner_mode = serializers.CharField()
    init_mode = serializers.CharField()
    init_in_progress = serializers.BooleanField()
    init_elapsed_s = serializers.FloatField(required=False, allow_null=True)
    resource_profile = serializers.JSONField()
    selected_variant = OrchestratorVariantSerializer(required=False, allow_null=True)
    selected_model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    active_model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    selected_model_path = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    selected_model_exists = serializers.BooleanField()
    selected_model_size_mb = serializers.FloatField(required=False, allow_null=True)
    active_download_path = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    active_download_size_mb = serializers.FloatField(required=False, allow_null=True)
    model_dir = serializers.CharField()
    available_models_total = serializers.IntegerField()
    external_fallback = ExternalFallbackStatusSerializer()
    orchestrator_config = serializers.JSONField(required=False)


class OrchestratorPlanStepSerializer(serializers.Serializer):
    tool = serializers.CharField()
    arguments = serializers.JSONField()


class OrchestratorPlanSerializer(serializers.Serializer):
    reasoning = serializers.CharField()
    steps = OrchestratorPlanStepSerializer(many=True)


class PredictionExecutionResultSerializer(serializers.Serializer):
    tool = serializers.CharField()
    result = serializers.JSONField(required=False)
    success = serializers.BooleanField()


class PredictionResponseSerializer(serializers.Serializer):
    results = serializers.JSONField(required=False)
    tools_used = serializers.ListField(child=serializers.CharField(), required=False)
    execution_results = serializers.JSONField(required=False)
    planner_mode = serializers.CharField(required=False, allow_blank=True)
    error = serializers.CharField(required=False, allow_blank=True)


class ToolRunRequestSerializer(serializers.Serializer):
    query = serializers.CharField(required=False, allow_blank=True, default="")
    arguments = serializers.DictField(required=False, default=dict)
    features = serializers.DictField(required=False, default=dict)

    def validate(self, data):
        arguments = dict(data.get("arguments") or {})
        arguments.update(dict(data.get("features") or {}))
        data["arguments"] = arguments
        return data


class ToolSummarySerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    adapter = serializers.CharField()
    enabled = serializers.BooleanField()
    available = serializers.BooleanField(required=False)
    description = serializers.CharField(required=False, allow_blank=True)
    aliases = serializers.ListField(child=serializers.CharField(), required=False)
    input_schema = serializers.JSONField(required=False)
    output_schema = serializers.JSONField(required=False)
    run_endpoint = serializers.CharField(required=False)


class ToolListResponseSerializer(serializers.Serializer):
    tools = ToolSummarySerializer(many=True)


class OrchestratorModelListResponseSerializer(serializers.Serializer):
    models = serializers.ListField(child=serializers.JSONField())
    selected_model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    active_model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    config = serializers.JSONField(required=False)


class OrchestratorSelectRequestSerializer(serializers.Serializer):
    model_id = serializers.CharField()
    download = serializers.BooleanField(required=False, default=True)
    wait = serializers.BooleanField(required=False, default=False)


class OrchestratorCatalogRequestSerializer(serializers.Serializer):
    models = serializers.ListField(child=serializers.DictField(), default=list)
    selected_model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )


class OrchestratorSelectionResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    selected_model_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    changed = serializers.BooleanField(required=False)
    download_triggered = serializers.BooleanField(required=False)
    updated_at = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    orchestrator_config = serializers.JSONField(required=False)
    orchestrator_status = OrchestratorStatusResponseSerializer()


class OrchestratorWarmupRequestSerializer(serializers.Serializer):
    wait = serializers.BooleanField(required=False, default=False)


class OrchestratorDownloadRequestSerializer(serializers.Serializer):
    wait = serializers.BooleanField(required=False, default=False)


# ── Training Scheduler ─────────────────────────────────────────────────────


class SchedulerJobSerializer(serializers.Serializer):
    id = serializers.CharField()
    name = serializers.CharField()
    next_run = serializers.CharField(required=False, allow_null=True)
    trigger = serializers.CharField()


class SchedulerStatusResponseSerializer(serializers.Serializer):
    jobs = SchedulerJobSerializer(many=True)


class SchedulerReloadResponseSerializer(serializers.Serializer):
    message = serializers.CharField()
    scripts = serializers.ListField(child=serializers.CharField())
    jobs = SchedulerJobSerializer(many=True)


class SchedulerRunRequestSerializer(serializers.Serializer):
    model_key = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )


class SchedulerRunResponseSerializer(serializers.Serializer):
    results = serializers.DictField()


class TrainingRunSerializer(serializers.Serializer):
    """A manual ("run now") training run tracked by the run manager."""

    run_id = serializers.CharField()
    status = serializers.ChoiceField(choices=["running", "completed", "failed"])
    model_key = serializers.CharField(required=False, allow_null=True)
    started_at = serializers.CharField()
    finished_at = serializers.CharField(required=False, allow_null=True)
    results = serializers.DictField(required=False, allow_null=True)
    error = serializers.CharField(required=False, allow_null=True)


class SchedulerRunStartResponseSerializer(serializers.Serializer):
    run = TrainingRunSerializer()
    started = serializers.BooleanField()
    message = serializers.CharField()


class SchedulerRunStatusResponseSerializer(serializers.Serializer):
    run = TrainingRunSerializer(required=False, allow_null=True)


class SchedulerConfigScriptSerializer(serializers.Serializer):
    script = serializers.CharField()
    frequency = serializers.CharField()
    time = serializers.CharField(required=False)
    day_of_week = serializers.CharField(required=False)
    day_of_month = serializers.IntegerField(required=False)
    enabled = serializers.BooleanField()
    description = serializers.CharField(required=False, allow_blank=True)


class SchedulerConfigResponseSerializer(serializers.Serializer):
    defaults = serializers.DictField()
    scripts = serializers.DictField(child=SchedulerConfigScriptSerializer())
    post_training = serializers.DictField()


class SchedulerConfigUpdateSerializer(serializers.Serializer):
    """Accepts a full or partial training_schedule.json payload."""

    defaults = serializers.DictField(required=False)
    scripts = serializers.DictField(required=False)
    post_training = serializers.DictField(required=False)


# ── Drift Monitoring ───────────────────────────────────────────────────


class DriftFeatureResultSerializer(serializers.Serializer):
    feature_name = serializers.CharField()
    dtype = serializers.CharField()
    test_name = serializers.CharField()
    statistic = serializers.FloatField()
    p_value = serializers.FloatField(required=False, allow_null=True)
    threshold = serializers.FloatField()
    drift_detected = serializers.BooleanField()
    severity = serializers.CharField()
    details = serializers.JSONField(required=False)


class DriftReportSummarySerializer(serializers.Serializer):
    id = serializers.IntegerField()
    model_id = serializers.CharField()
    mlflow_run_id = serializers.CharField()
    drift_detected = serializers.BooleanField()
    drift_score = serializers.FloatField()
    severity = serializers.CharField()
    features_checked = serializers.IntegerField()
    features_drifted = serializers.IntegerField()
    reference_size = serializers.IntegerField()
    current_size = serializers.IntegerField()
    check_type = serializers.CharField()
    checked_at = serializers.DateTimeField()


class DriftReportDetailSerializer(DriftReportSummarySerializer):
    feature_details = serializers.JSONField()


class DriftStatusResponseSerializer(serializers.Serializer):
    models = serializers.ListField(child=DriftReportSummarySerializer())


class DriftReportsListResponseSerializer(serializers.Serializer):
    reports = DriftReportDetailSerializer(many=True)
    count = serializers.IntegerField()


class DriftRunRequestSerializer(serializers.Serializer):
    model_name = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )
    check_type = serializers.CharField(required=False, default="manual")


class DriftRunResponseSerializer(serializers.Serializer):
    results = serializers.DictField()


# ── External Model Upload ──────────────────────────────────────────────────


class ModelUploadResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    model_id = serializers.CharField()
    model_type = serializers.CharField()
    file_path = serializers.CharField()
    features = serializers.ListField(child=serializers.CharField())
    is_active = serializers.BooleanField()
    message = serializers.CharField(required=False, allow_blank=True)


class ModelUploadRequestSerializer(serializers.Serializer):
    """Multipart form schema for the model upload endpoint.

    model_file — the serialized model artifact (.pkl / .ubj / .pt / .pth / .joblib)
    config     — JSON string describing the model (model_id, model_type, features, …)
    """

    model_file = serializers.FileField(help_text="Model artifact file (.pkl, .ubj, .pt, .pth, .joblib)")
    config = serializers.CharField(
        help_text=(
            'JSON string. Required keys: model_id, model_type, features. '
            'Example: {"model_id":"donor_v1","model_type":"sklearn","features":["age","weight"]}'
        )
    )
