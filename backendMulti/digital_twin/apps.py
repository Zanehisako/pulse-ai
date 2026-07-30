from django.apps import AppConfig


class DigitalTwinConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "digital_twin"

    def ready(self) -> None:
        from digital_twin.config import validate_digital_twin_config

        validate_digital_twin_config()
