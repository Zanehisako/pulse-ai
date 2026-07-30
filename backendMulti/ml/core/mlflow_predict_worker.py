from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, default=str))


def main() -> int:
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
        try:
            import django
            from django.apps import apps

            if not apps.ready:
                django.setup()
        except Exception:
            pass

        from ml.core.prediction import run_prediction
        from ml.core.registry import ModelRuntime

        payload = json.loads(sys.stdin.read() or "{}")
        runtime_payload = payload.get("runtime")
        features = payload.get("features")
        feature_rows = payload.get("feature_rows")
        if not isinstance(runtime_payload, dict):
            raise ValueError("Worker payload must include a runtime object.")
        if feature_rows is not None and not isinstance(feature_rows, list):
            raise ValueError("feature_rows must be a list when provided.")
        if feature_rows is None and not isinstance(features, dict):
            raise ValueError("Worker payload must include features or feature_rows.")

        runtime = ModelRuntime(
            model_id=str(runtime_payload.get("model_id") or ""),
            aliases=[
                str(item)
                for item in runtime_payload.get("aliases", [])
                if str(item).strip()
            ],
            slug=str(runtime_payload.get("slug") or runtime_payload.get("model_id") or ""),
            file_path=runtime_payload.get("file_path") or "",
            model_type=str(runtime_payload.get("model_type") or "mlflow"),
            description=str(runtime_payload.get("description") or ""),
            feature_names=[
                str(item)
                for item in runtime_payload.get("feature_names", [])
                if str(item).strip()
            ],
            feature_info=(
                runtime_payload.get("feature_info")
                if isinstance(runtime_payload.get("feature_info"), dict)
                else {}
            ),
            examples=(
                runtime_payload.get("examples")
                if isinstance(runtime_payload.get("examples"), list)
                else []
            ),
            defaults=(
                runtime_payload.get("defaults")
                if isinstance(runtime_payload.get("defaults"), dict)
                else {}
            ),
            status=str(runtime_payload.get("status") or "loaded"),
            load_error=runtime_payload.get("load_error"),
        )
        if isinstance(runtime.file_path, str) and runtime.file_path.startswith("/"):
            runtime.file_path = Path(runtime.file_path)

        if feature_rows is not None:
            outputs = [
                run_prediction(runtime, row)
                for row in feature_rows
                if isinstance(row, dict)
            ]
            _write({"outputs": outputs})
        else:
            output = run_prediction(runtime, features)
            _write({"output": output})
        return 0
    except Exception as exc:
        _write({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
