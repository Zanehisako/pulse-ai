from django.apps import apps
from django.db.models import Count, Sum

from .widget_sources import get_metric_config, get_widget_source


def execute_dynamic_query(config):
    source_id = config.get("model")
    group_by = config.get("group_by")
    metric = config.get("metric", "count")
    source = get_widget_source(source_id)

    allowed_group_by = {option["value"] for option in source["group_by"]}
    if group_by not in allowed_group_by:
        raise Exception("Group by not allowed")

    metric_config = get_metric_config(source, metric)
    field = metric_config.get("field") or config.get("field") or "id"

    app_label = source.get("app_label", "workspace")
    Model = apps.get_model(app_label, source["model"])

    queryset = Model.objects.all()

    if metric == "count":
        data = queryset.values(group_by).annotate(value=Count(field))

    elif metric == "sum":
        data = queryset.values(group_by).annotate(value=Sum(field))

    else:
        raise Exception("Invalid metric")

    return list(data)
