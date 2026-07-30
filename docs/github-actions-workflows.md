# GitHub Actions Workflows

This document explains the automation files in `.github/workflows/` so the CI/CD setup is easy to review

## Overview

The repository currently has three GitHub Actions workflows:

| Workflow file                              | Purpose                                                                           | Main trigger                                                  |
| ------------------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `.github/workflows/tests.yml`              | Runs Python tests with coverage gates for `backendMulti` and `simulation_studio`. | Every push and pull request                                   |
| `.github/workflows/python-smoke-tests.yml` | Runs a focused ML/backend import smoke test for the training schedule.            | Pull requests and `dev` pushes that touch backend or ML files |
| `.github/workflows/release.yml`            | Builds and publishes the Flutter Android APK.                                     | Pushes to `dev` that touch `mobile-app/**`                    |

The most important quality workflow is `tests.yml`. It is the main continuous integration gate for the Python services.

## `.github/workflows/tests.yml`

This workflow runs automatically on every push and pull request. It has read-only repository permissions and cancels older in-progress runs for the same branch, which keeps CI feedback fast.

### Job: `backendMulti tests`

This job tests the Django backend in `backendMulti`.

What it does:

- Starts a PostgreSQL 16 service container for the Django test database.
- Uses Python 3.11.
- Installs backend dependencies with `python -m pip install -e ".[dev]"`.
- Runs pytest across the backend app folders.
- Measures coverage for the backend modules that currently have meaningful unit/integration coverage.
- Fails the workflow if tests fail or the configured backend coverage gate is below 80%.

The job uses dummy CI environment values only:

- `DJANGO_SECRET_KEY=ci-dummy-secret`
- `KEYCLOAK_CLIENT_SECRET=ci-dummy-client-secret`
- `DB_USER=admin`
- `DB_PASSWORD=admin`

These are not production secrets. They exist only so tests can boot safely in CI.

External services:

- PostgreSQL is started because the Django settings use PostgreSQL by default.
- Keycloak is not started because the current backend tests mock Keycloak behavior where needed.
- MongoDB, MLflow, and Redis are not started because the current CI test scope does not require them.

Coverage gate:

```bash
--cov-fail-under=80
```

This means the `backendMulti` job fails if the configured backend coverage gate is lower than 80%.
The backend tests still run across all app folders; the coverage gate is scoped to tested core modules instead of every Django glue file, admin file, management command, legacy CRUD view, and experimental ML utility.

### Job: `simulation_studio tests`

This job tests the FastAPI simulation service in `simulation_studio`.

What it does:

- Uses Python 3.11.
- Installs Simulation Studio dependencies with `python -m pip install -e ".[dev]" pytest-cov`.
- Runs tests under `simulation_studio/tests`.
- Measures coverage for the tested Simulation Studio runtime modules.
- Fails the workflow if tests fail or Simulation Studio coverage is below 80%.

External services:

- No database or Docker service is needed.
- The current tests use FastAPI `TestClient`, temporary files, and monkeypatching, so they run in-process.

Coverage gate:

```bash
--cov-fail-under=80
```

This means the `simulation_studio` job fails if the configured Simulation Studio runtime coverage gate is lower than 80%.
The job still runs all tests under `simulation_studio/tests`; the coverage gate avoids counting training scripts, benchmark scripts, generated diagram scripts, and experimental RL modules that are not part of the CI smoke surface.

## `.coveragerc`

The Python test workflow uses the root `.coveragerc` file to keep coverage reports fair and readable.

It excludes files that should not count as production logic, such as:

- tests
- migrations
- `__init__.py`
- Django settings and ASGI/WSGI boot files
- virtual environments and build folders
- notebooks, datasets, generated model artifacts, and static assets
- archived or generated simulation/training result folders

This keeps the 80% gate focused on maintainable application code.

## `.github/workflows/python-smoke-tests.yml`

This is a smaller smoke-test workflow that existed before the broader Python CI workflow.

It runs only when backend or ML files change, and only on:

- pull requests
- pushes to the `dev` branch

Its job is `Training schedule imports`.

What it checks:

- Installs a focused set of dependencies.
- Runs:

```bash
python -m pytest ml-backend/tests/test_training_schedule_imports.py -q
```

This does not replace the main Python CI workflow. It is a quick guard for the ML training schedule import path.

## `.github/workflows/release.yml`

This workflow builds the Flutter Android APK when mobile code changes on the `dev` branch.

What it does:

- Uses Java 17.
- Uses Flutter stable.
- Runs `flutter pub get`.
- Builds a release APK.
- Publishes or updates a GitHub Release using the generated version and GitHub run number.

This workflow is release automation, not the main test gate.

## Current CI Scope

Current automated test coverage in `tests.yml` is intentionally limited to Python services:

- `backendMulti`
- `simulation_studio`

The following are not yet included in `tests.yml`:

- `web-app/pios-web`
- `mobile-app`
- `ml-backend`

They can be added later as separate jobs, so each service remains understandable and failures are easy to diagnose.

## Local Verification Commands

To reproduce the backend CI job locally, start PostgreSQL first:

```bash
docker compose -f infrastructure/docker/docker-compose.local.yaml up -d postgres
```

Then run the backend test command from `backendMulti`:

```bash
cd backendMulti
python -m pytest tests authApp workspace backendMulti bloodbag centres communications dashboard inventory media ml notifications testUrl alerts --cov=alerts.services --cov=alerts.models --cov=alerts.serializers --cov=inventory.forecast_ml_bridge --cov=inventory.models --cov=inventory.seed_data --cov=inventory.simulation --cov=ml.api.serializers --cov=ml.core.feature_extraction --cov=ml.models --cov-config=../.coveragerc --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=80
```

To reproduce the Simulation Studio CI job locally, run this from `simulation_studio`:

```bash
cd ../simulation_studio
python -m pytest tests --cov=app.main --cov=app.schemas --cov=app.simulator_service --cov=app.studio_routes --cov=app.studio_service --cov=simulator.calibration --cov=simulator.core --cov=simulator.eval_metrics --cov=simulator.scenarios --cov-config=../.coveragerc --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=80
```

## Security Notes

The workflows do not contain real secrets. Any value that looks like a secret is a dummy CI value used only for tests.

If future workflows need real credentials, they should use GitHub Secrets instead of hardcoding them in YAML files.
