from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import ComparisonRequest, CustomScenarioCreate, SingleRunRequest
from .simulator_service import (
    create_custom_scenario,
    get_app_metadata,
    load_cached_summary,
    run_comparison,
    run_single,
)
from .config_routes import router as config_router
from .studio_routes import router as studio_router
from .validation_routes import router as validation_router
from .ws_digital_twin import router as ws_twin_router
from .ws_simulation import router as ws_sim_router

STATIC_ROOT = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_ROOT / "index.html"


def create_app(*, include_web_app: bool = True) -> FastAPI:
    app = FastAPI(
        title="Simulation Studio",
        version="0.1.0",
        description="A fresh simulator dashboard for comparing blood supply scenarios and strategies.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(validation_router)
    app.include_router(ws_sim_router)
    app.include_router(ws_twin_router)
    app.include_router(studio_router)
    app.include_router(config_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta")
    def meta() -> dict:
        return get_app_metadata()

    @app.get("/api/evaluations/cached-summary")
    def cached_summary() -> dict:
        return load_cached_summary()

    @app.post("/api/evaluations/compare")
    def compare(request: ComparisonRequest) -> dict:
        return run_comparison(request)

    @app.post("/api/evaluations/run")
    def evaluate_single(request: SingleRunRequest) -> dict:
        return run_single(request)

    @app.post("/api/custom-scenarios")
    def create_scenario(payload: CustomScenarioCreate) -> dict:
        return create_custom_scenario(payload)

    if include_web_app:
        app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(INDEX_PATH)

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            requested = (STATIC_ROOT / full_path).resolve()
            if (
                requested.exists()
                and requested.is_file()
                and STATIC_ROOT in requested.parents
            ):
                return FileResponse(requested)
            return FileResponse(INDEX_PATH)

    return app


app = create_app()
api_app = create_app(include_web_app=False)
