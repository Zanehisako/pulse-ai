import os
import sys
import threading
import logging

from anyio import Path
from django.apps import AppConfig
from pathlib import Path

logger = logging.getLogger(__name__)


class MlConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ml"
    verbose_name = "Machine Learning"

    def ready(self):
        if os.environ.get("DJANGO_SKIP_ML_INIT") == "1":
            return

        # running_command = sys.argv[1] if len(sys.argv) > 1 else ""
        # if running_command not in {"runserver", "daphne", "runworker"}:
        #     return

        # if running_command == "runserver" and os.environ.get("RUN_MAIN") != "true":
        #     return
        argv0 = Path(sys.argv[0]).stem.lower() if sys.argv else ""
        is_daphne = argv0 == "daphne" or any("daphne" in a for a in sys.argv)
        is_runserver = sys.argv[1] == "runserver" if len(sys.argv) > 1 else False

        if not (is_daphne or is_runserver):
            return
        if is_runserver and os.environ.get("RUN_MAIN") != "true" and "--noreload" not in sys.argv:
            return

        def _deferred_init():
            import time
            time.sleep(2)
            try:
                from ml.core.startup import initialize
                initialize()
            except Exception as e:
                logger.exception("PIOS ML initialization failed: %s", e)

        threading.Thread(target=_deferred_init, name="pios-ml-init", daemon=True).start()
