from __future__ import annotations

import importlib.util
import os
import sys
from itertools import chain
from pathlib import Path
from typing import Iterable


SIMULATOR_DIR = Path(__file__).resolve().parent
ML_BACKEND_DIR = SIMULATOR_DIR.parent
REPO_ROOT = ML_BACKEND_DIR.parent

REQUIRED_RUNTIME_MODULES = ("dreamerv3", "embodied", "portal", "jax")

DEFAULT_RUNTIME_CANDIDATES = (
    SIMULATOR_DIR / "cache" / "dreamerv3-official",
    SIMULATOR_DIR / "cache" / "dreamerv3-pkgs-metal011",
    SIMULATOR_DIR / "cache" / "dreamerv3-pkgs",
    ML_BACKEND_DIR / "cache" / "dreamerv3-official",
    ML_BACKEND_DIR / "cache" / "dreamerv3-pkgs-metal011",
    ML_BACKEND_DIR / "cache" / "dreamerv3-pkgs",
    REPO_ROOT / "cache" / "dreamerv3-official",
    REPO_ROOT / "cache" / "dreamerv3-pkgs-metal011",
    REPO_ROOT / "cache" / "dreamerv3-pkgs",
    REPO_ROOT / "third_party" / "dreamerv3",
    REPO_ROOT / "third_party" / "dreamerv3-pkgs-metal011",
    REPO_ROOT / "third_party" / "dreamerv3-pkgs",
    REPO_ROOT / "vendor" / "dreamerv3",
    REPO_ROOT / "vendor" / "dreamerv3-pkgs-metal011",
    REPO_ROOT / "vendor" / "dreamerv3-pkgs",
    Path("/tmp/dreamerv3-official"),
    Path("/tmp/dreamerv3-pkgs"),
)


def _probe_optional_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        normalized = path.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _iter_env_candidates() -> tuple[Path, ...]:
    return _unique_paths(
        Path(value)
        for value in (
            os.environ.get("PIOS_DREAMERV3_RUNTIME"),
            os.environ.get("PIOS_DREAMERV3_SRC"),
            os.environ.get("PIOS_DREAMERV3_PKGS"),
        )
        if value
    )


def _looks_like_checkout_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "dreamerv3" / "configs.yaml").exists()
        and (path / "embodied").is_dir()
    )


def _looks_like_package_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).exists()
        for name in ("dreamerv3", "embodied", "portal", "elements", "jax", "jaxlib")
    )


def _candidate_import_paths(path: Path) -> tuple[Path, ...]:
    candidates = [path]
    if path.name in {"dreamerv3", "embodied"}:
        candidates.append(path.parent)
    if path.is_dir():
        candidates.extend(sorted(path.glob("lib/python*/site-packages")))
    return _unique_paths(candidates)


def discover_official_runtime_paths(
    extra_candidates: Iterable[Path | str] = (),
) -> tuple[str, ...]:
    ordered: list[Path] = []
    env_candidates = _iter_env_candidates()
    extra = tuple(Path(candidate) for candidate in extra_candidates)
    if env_candidates:
        default_fallbacks = tuple(
            candidate for candidate in DEFAULT_RUNTIME_CANDIDATES if _looks_like_checkout_root(candidate)
        )
    else:
        default_fallbacks = DEFAULT_RUNTIME_CANDIDATES
    all_candidates = chain(env_candidates, extra, default_fallbacks)
    for candidate in all_candidates:
        for resolved in _candidate_import_paths(candidate.expanduser()):
            if _looks_like_checkout_root(resolved) or _looks_like_package_dir(resolved):
                ordered.append(resolved.resolve())
    return tuple(str(path) for path in _unique_paths(ordered))


def inject_official_runtime_paths(
    extra_candidates: Iterable[Path | str] = (),
) -> tuple[str, ...]:
    paths = discover_official_runtime_paths(extra_candidates)
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    return paths


def resolve_official_source_root(
    extra_candidates: Iterable[Path | str] = (),
) -> Path | None:
    all_candidates = chain(
        _iter_env_candidates(),
        (Path(candidate) for candidate in extra_candidates),
        DEFAULT_RUNTIME_CANDIDATES,
    )
    for candidate in all_candidates:
        for resolved in _candidate_import_paths(candidate.expanduser()):
            if _looks_like_checkout_root(resolved):
                return resolved.resolve()
    return None


def missing_official_runtime_modules() -> tuple[str, ...]:
    return tuple(name for name in REQUIRED_RUNTIME_MODULES if not _probe_optional_module(name))


def format_official_runtime_error(exc: Exception) -> str:
    source_root = resolve_official_source_root()
    import_paths = discover_official_runtime_paths()
    missing = missing_official_runtime_modules()
    searched = tuple(str(path) for path in _unique_paths(chain(_iter_env_candidates(), DEFAULT_RUNTIME_CANDIDATES)))
    parts = [
        "Official DreamerV3 dependencies could not be imported.",
        "Expected either a DreamerV3 checkout containing `dreamerv3/` and `embodied/`, or a package directory with those modules installed.",
    ]
    if missing:
        parts.append(f"Missing modules in the active runtime: {', '.join(missing)}.")
    if source_root is not None:
        parts.append(f"Resolved DreamerV3 source root: {source_root}.")
    if import_paths:
        parts.append(f"Injected runtime import paths: {', '.join(import_paths)}.")
    if searched:
        parts.append(f"Searched runtime locations: {', '.join(searched)}.")
    parts.append(
        "If you have a local checkout of the official repo, point `PIOS_DREAMERV3_SRC` at its root. "
        "Then install its requirements plus the editable package into this Python environment."
    )
    parts.append(f"Original error: {exc}")
    return " ".join(parts)
