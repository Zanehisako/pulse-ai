#!/usr/bin/env python3
"""Select a local Python interpreter from a project's pyproject metadata."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for old bootstrap Python
    tomllib = None


class ConfigError(ValueError):
    pass


def read_requires_python(project_dir: str | Path) -> str:
    pyproject_path = Path(project_dir) / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ConfigError(f"pyproject.toml not found: {pyproject_path}")

    raw = pyproject_path.read_bytes()
    if tomllib is not None:
        data = tomllib.loads(raw.decode("utf-8"))
        requirement = data.get("project", {}).get("requires-python")
    else:
        requirement = _read_requires_python_without_tomllib(raw.decode("utf-8"))

    if not requirement:
        raise ConfigError(f"[project].requires-python missing in {pyproject_path}")
    return str(requirement).strip()


def _read_requires_python_without_tomllib(text: str) -> str | None:
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and stripped.startswith("requires-python"):
            match = re.match(r'requires-python\s*=\s*["\']([^"\']+)["\']', stripped)
            return match.group(1) if match else None
    return None


def version_tuple(version: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(version, tuple):
        return tuple((version + (0, 0, 0))[:3])
    parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def interpreter_version(interpreter: str) -> tuple[int, int, int] | None:
    try:
        result = subprocess.run(
            [
                interpreter,
                "-c",
                "import sys; print('.'.join(map(str, sys.version_info[:3])))",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return version_tuple(result.stdout.strip())


def supports_requirement(version: tuple[int, int, int], requirement: str) -> bool:
    for clause in requirement.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if clause.startswith("~="):
            minimum = version_tuple(clause[2:].strip())
            if version < minimum:
                return False
            upper = _compatible_release_upper_bound(clause[2:].strip())
            if version >= upper:
                return False
            continue

        match = re.match(r"(>=|<=|==|!=|>|<)\s*(.+)", clause)
        if not match:
            raise ConfigError(f"Unsupported requires-python clause: {clause}")

        operator, expected_raw = match.groups()
        expected = version_tuple(expected_raw)
        if operator == ">=" and version < expected:
            return False
        if operator == "<=" and version > expected:
            return False
        if operator == ">" and version <= expected:
            return False
        if operator == "<" and version >= expected:
            return False
        if operator == "==" and version != expected:
            return False
        if operator == "!=" and version == expected:
            return False
    return True


def _compatible_release_upper_bound(version: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", version)]
    if len(parts) <= 2:
        return (parts[0] + 1, 0, 0)
    return (parts[0], parts[1] + 1, 0)


def iter_python_candidates() -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []

    override = os.environ.get("PIOS_PYTHON")
    if override:
        return [override]

    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not path_dir:
            continue
        directory = Path(path_dir)
        try:
            names = sorted(directory.iterdir())
        except OSError:
            continue
        for path in names:
            if not re.fullmatch(r"python(?:3(?:\.\d+)?)?(?:\.exe)?", path.name):
                continue
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = str(path)
            if resolved in seen or not os.access(path, os.X_OK):
                continue
            seen.add(resolved)
            candidates.append(str(path))
    return candidates


def select_interpreter(requirement: str, candidates: list[str] | None = None) -> str:
    candidates = candidates if candidates is not None else iter_python_candidates()
    compatible: list[tuple[tuple[int, int, int], str]] = []
    for candidate in candidates:
        version = interpreter_version(candidate)
        if version is not None and supports_requirement(version, requirement):
            compatible.append((version, candidate))

    if compatible:
        return sorted(compatible, reverse=True)[0][1]

    if os.environ.get("PIOS_PYTHON"):
        raise ConfigError(
            f"PIOS_PYTHON does not satisfy requires-python '{requirement}': "
            f"{os.environ['PIOS_PYTHON']}"
        )
    raise ConfigError(
        f"No Python interpreter on PATH satisfies requires-python '{requirement}'. "
        "Install a matching Python or set PIOS_PYTHON=/path/to/python."
    )


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            "Usage: select_python.py <select|check|requirement> <project-dir> [interpreter]",
            file=sys.stderr,
        )
        return 2

    command = argv[1]
    project_dir = argv[2]
    try:
        requirement = read_requires_python(project_dir)
        if command == "requirement":
            print(requirement)
            return 0
        if command == "select":
            print(select_interpreter(requirement))
            return 0
        if command == "check":
            if len(argv) != 4:
                print("check requires an interpreter path", file=sys.stderr)
                return 2
            version = interpreter_version(argv[3])
            return 0 if version and supports_requirement(version, requirement) else 1
    except ConfigError as exc:
        print(f"Python runtime configuration error: {exc}", file=sys.stderr)
        return 1

    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
