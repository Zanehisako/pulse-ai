"""Context-window fitting for the orchestrator planner LLM.

All architecture-specific memory accounting is delegated to llama.cpp itself:

* native context window  -> ``llama_model_n_ctx_train`` on a real model load
* model tensor bytes     -> ``llama_model_size``
* per-candidate context  -> the KV cache and compute-buffer reservations that
  llama.cpp prints while constructing a ``llama_context`` (``llama_kv_cache:
  size = ... MiB`` and ``sched_reserve: <device> compute buffer size = ...``)

The staging ``llama_get_memory_breakdown`` API is not used because it returns
a C++ ``std::map`` (unreachable from ctypes), and this llama.cpp build aborts
on ``no_alloc`` (``GGML_ASSERT(!ml.no_alloc)``). The printed reservations are
produced by the same internal accounting that powers that breakdown, so the
estimate matches what the real load will allocate.

macOS overcommits anonymous memory, so a candidate that does not fit does not
fail fast: it swaps instead. The fitter therefore measures the per-token KV
slope with two small probes and never probes candidates above the affordable
upper bound, then binary-searches real probes over the safe range.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

KV_CACHE_SIZE_RE = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB")
COMPUTE_BUFFER_SIZE_RE = re.compile(
    r"sched_reserve:\s+\S+? compute buffer size =\s*([\d.]+) MiB"
)
OUTPUT_BUFFER_SIZE_RE = re.compile(
    r"llama_context:\s+\S+\s+output buffer size =\s*([\d.]+) MiB"
)


@dataclass(frozen=True)
class DeviceMemory:
    """Per-device memory requirement and availability for a candidate."""

    required_bytes: int
    free_bytes: int
    reserve_bytes: int

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.free_bytes - self.reserve_bytes


Preflight = Callable[[int], list[DeviceMemory]]


def align_down(value: int, alignment: int = 256) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return value - value % alignment


def choose_context_size(
    *,
    native_context: int,
    preflight: Preflight,
    minimum_context: int = 4096,
    maximum_context: int | None = None,
    alignment: int = 256,
) -> int:
    """Largest context that fits on every device, found by binary search.

    Mirrors llama.cpp's own fit logic: search the largest aligned context
    that the preflight reports as fitting in memory, never below
    ``minimum_context``.
    """
    if native_context <= 0:
        raise ValueError("GGUF model did not provide a valid native context")

    upper = native_context

    if maximum_context is not None:
        upper = min(upper, maximum_context)

    lower = min(minimum_context, upper)

    lower = max(alignment, align_down(lower, alignment))
    upper = max(lower, align_down(upper, alignment))

    best: int | None = None
    low_units = lower // alignment
    high_units = upper // alignment

    while low_units <= high_units:
        middle_units = (low_units + high_units) // 2
        candidate = middle_units * alignment

        device_usage = preflight(candidate)
        fits = bool(device_usage) and all(device.fits for device in device_usage)

        if fits:
            best = candidate
            low_units = middle_units + 1
        else:
            high_units = middle_units - 1

    if best is None:
        raise MemoryError(
            f"Model does not fit even at {lower:,} context tokens"
        )

    return best


def free_memory_bytes() -> int:
    """Best-effort currently free physical memory in bytes (0 if unknown)."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) // (1024 * 1024) * (1024 * 1024)
    except (AttributeError, ValueError, OSError):
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
    return 0


def _parse_context_memory(stderr_text: str) -> tuple[int, int, int] | None:
    """Parse llama.cpp's own buffer reservations from context construction.

    Returns ``(kv_bytes, compute_bytes, output_bytes)`` or ``None`` when the
    expected accounting lines are missing (candidate could not be judged).
    """
    kv = sum(
        int(round(float(match) * 1024 * 1024))
        for match in KV_CACHE_SIZE_RE.findall(stderr_text)
    )
    compute = sum(
        int(round(float(match) * 1024 * 1024))
        for match in COMPUTE_BUFFER_SIZE_RE.findall(stderr_text)
    )
    output = sum(
        int(round(float(match) * 1024 * 1024))
        for match in OUTPUT_BUFFER_SIZE_RE.findall(stderr_text)
    )
    if kv <= 0 and compute <= 0:
        return None
    return kv, compute, output


def _build_model_params(runtime_config: dict[str, Any]) -> Any:
    """llama_model_params matching the real inference load."""
    import llama_cpp

    params = llama_cpp.llama_model_default_params()
    params.n_gpu_layers = int(runtime_config.get("n_gpu_layers") or 0)
    params.use_mmap = bool(runtime_config.get("use_mmap", True))
    params.use_mlock = bool(runtime_config.get("use_mlock", False))
    params.vocab_only = False
    return params


def _build_context_params(runtime_config: dict[str, Any], n_ctx: int) -> Any:
    """llama_context_params mirroring llama-cpp-python's Llama constructor."""
    import llama_cpp

    params = llama_cpp.llama_context_default_params()
    params.n_ctx = max(1, int(n_ctx))
    params.n_batch = max(1, int(runtime_config.get("n_batch") or 512))
    params.n_ubatch = max(
        1, int(runtime_config.get("n_ubatch") or params.n_batch)
    )
    params.n_seq_max = max(1, int(runtime_config.get("n_seq_max") or 1))
    params.embeddings = False
    flash_attn = bool(runtime_config.get("flash_attn", False))
    params.flash_attn_type = (
        llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
        if flash_attn
        else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
    )
    params.offload_kqv = bool(runtime_config.get("offload_kqv", False))
    if runtime_config.get("swa_full") is not None:
        params.swa_full = bool(runtime_config["swa_full"])
    if runtime_config.get("type_k") is not None:
        params.type_k = runtime_config["type_k"]
    if runtime_config.get("type_v") is not None:
        params.type_v = runtime_config["type_v"]
    return params


def load_probe_model(
    model_path: str, runtime_config: dict[str, Any]
) -> Any:
    """Load a real llama.cpp model object for context probing."""
    from llama_cpp._internals import LlamaModel

    return LlamaModel(
        path_model=model_path,
        params=_build_model_params(runtime_config),
        verbose=False,
    )


def probe_context_memory(
    model: Any,
    n_ctx: int,
    runtime_config: dict[str, Any],
) -> tuple[int, int, int] | None:
    """Construct a candidate context and return its llama.cpp-reported memory.

    Returns ``(model_bytes, kv_bytes, compute_bytes)`` where the context
    reservation is ``kv + compute + output``, or ``None`` when the candidate
    could not be created or its accounting lines could not be parsed.
    """
    buffer = io.StringIO()
    try:
        from llama_cpp._internals import LlamaContext

        with redirect_stderr(buffer):
            ctx = LlamaContext(
                model=model,
                params=_build_context_params(runtime_config, n_ctx),
                verbose=False,
            )
        parsed = _parse_context_memory(buffer.getvalue())
    except Exception as exc:  # allocation failure surfaces as exceptions
        logger.debug("context probe at n_ctx=%s failed: %s", n_ctx, exc)
        return None
    if parsed is None:
        return None
    try:
        ctx.close()
    except Exception:
        pass
    return model.size(), parsed[0], parsed[1] + parsed[2]


_DEVICE_MEMORY_UNFITTABLE = 2**62


def _device_memory_for(
    required_bytes: int,
    free_bytes: int,
    reserve_bytes: int,
) -> list[DeviceMemory]:
    return [
        DeviceMemory(
            required_bytes=max(0, required_bytes),
            free_bytes=max(0, free_bytes),
            reserve_bytes=max(0, reserve_bytes),
        )
    ]


def _fit_cache_key(
    model_path: str,
    runtime_config: dict[str, Any],
    minimum_context: int,
    maximum_context: int | None,
    alignment: int,
    reserve_bytes: int,
    probe_step: int,
) -> str:
    payload = {
        "model_path": str(model_path),
        "runtime": runtime_config,
        "minimum_context": minimum_context,
        "maximum_context": maximum_context,
        "alignment": alignment,
        "reserve_bytes": reserve_bytes,
        "probe_step": probe_step,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_fit_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path or not Path(cache_path).exists():
        return {}
    try:
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_fit_cache(
    cache_path: Path, cache_key: str, entry: dict[str, Any]
) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = load_fit_cache(path)
        cache[cache_key] = entry
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(
            json.dumps(cache, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        logger.warning("Could not persist context fit cache: %s", exc)


def cached_fit_still_fits(
    entry: dict[str, Any], free_bytes: int, reserve_bytes: int
) -> bool:
    required = int(entry.get("required_bytes") or 0)
    return required > 0 and free_bytes - reserve_bytes >= required


@dataclass(frozen=True)
class FitResult:
    chosen_context: int
    native_context: int
    model_bytes: int
    context_bytes: int
    compute_bytes: int
    free_bytes: int
    required_bytes: int
    cached: bool = False
    probes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_context": self.chosen_context,
            "native_context": self.native_context,
            "model_bytes": self.model_bytes,
            "context_bytes": self.context_bytes,
            "compute_bytes": self.compute_bytes,
            "free_bytes": self.free_bytes,
            "required_bytes": self.required_bytes,
            "cached": self.cached,
            "probes": self.probes,
            "fitted_at": time.time(),
        }


def fit_context(
    *,
    model_path: str,
    runtime_config: dict[str, Any],
    minimum_context: int = 4096,
    maximum_context: int | None = None,
    alignment: int = 256,
    reserve_bytes: int = 1024 * 1024 * 1024,
    probe_step: int = 4096,
    free_bytes: int | None = None,
    cache_path: Path | None = None,
    cache_key: str | None = None,
) -> FitResult:
    """Fit the largest context window for the given model and runtime.

    Probes are real llama.cpp contexts created from one shared model load, so
    the verdict (and the reported reservation) is llama.cpp's own. A cached
    result is reused only while it still fits the currently free memory.
    """
    if cache_key and cache_path:
        cache = load_fit_cache(cache_path)
        entry = cache.get(cache_key)
        if isinstance(entry, dict) and entry.get("chosen_context"):
            current_free = free_bytes if free_bytes is not None else free_memory_bytes()
            if cached_fit_still_fits(entry, current_free, reserve_bytes):
                return FitResult(
                    chosen_context=int(entry["chosen_context"]),
                    native_context=int(entry.get("native_context") or 0),
                    model_bytes=int(entry.get("model_bytes") or 0),
                    context_bytes=int(entry.get("context_bytes") or 0),
                    compute_bytes=int(entry.get("compute_bytes") or 0),
                    free_bytes=current_free,
                    required_bytes=int(entry.get("required_bytes") or 0),
                    cached=True,
                )
        cache_path = None

    if free_bytes is None:
        free_bytes = free_memory_bytes()
    if free_bytes <= 0:
        raise RuntimeError(
            "Cannot fit context: free memory could not be determined"
        )

    model = load_probe_model(model_path, runtime_config)
    try:
        native_context = int(model.n_ctx_train() or 0)
        if native_context <= 0:
            raise RuntimeError(
                "Cannot fit context: model did not report a native context"
            )

        def probe(n_ctx: int) -> tuple[int, int, int] | None:
            return probe_context_memory(model, n_ctx, runtime_config)

        minimum = max(alignment, align_down(minimum_context, alignment))
        base_probe = probe(minimum)
        if base_probe is None:
            raise MemoryError(
                f"Model does not fit even at {minimum:,} context tokens"
            )
        model_bytes, base_kv, base_compute = base_probe

        upper = native_context
        if maximum_context is not None:
            upper = min(upper, maximum_context)

        step_probe = probe(minimum + probe_step)
        if step_probe is not None:
            _, step_kv, _ = step_probe
            slope = max(0.0, (step_kv - base_kv) / float(max(1, probe_step)))
            if slope <= 0:
                logger.warning(
                    "Context probe slope could not be measured at %s tokens "
                    "(base kv=%s, step kv=%s); capping search at %s",
                    probe_step,
                    base_kv,
                    step_kv,
                    minimum,
                )
                upper = minimum
            else:
                overhead = model_bytes + base_compute + base_kv
                affordable = int((free_bytes - reserve_bytes - overhead) / slope)
                upper = min(
                    upper,
                    max(minimum, align_down(minimum + affordable, alignment)),
                )
        else:
            logger.warning(
                "Context slope probe at %s tokens could not be judged; "
                "capping search at %s",
                minimum + probe_step,
                minimum,
            )
            upper = minimum

        def preflight(n_ctx: int) -> list[DeviceMemory]:
            probe_result = probe(n_ctx)
            if probe_result is None:
                return _device_memory_for(
                    _DEVICE_MEMORY_UNFITTABLE, free_bytes, reserve_bytes
                )
            _, kv, compute = probe_result
            probed[n_ctx] = (kv, compute)
            required = model_bytes + kv + compute
            return _device_memory_for(required, free_bytes, reserve_bytes)

        probed: dict[int, tuple[int, int]] = {}
        chosen = choose_context_size(
            native_context=native_context,
            preflight=preflight,
            minimum_context=minimum,
            maximum_context=upper,
            alignment=alignment,
        )
    finally:
        try:
            model.close()
        except Exception:
            pass

    chosen_kv, chosen_compute = probed.get(chosen, (base_kv, base_compute))
    result = FitResult(
        chosen_context=chosen,
        native_context=native_context,
        model_bytes=model_bytes,
        context_bytes=chosen_kv,
        compute_bytes=chosen_compute,
        free_bytes=free_bytes,
        required_bytes=model_bytes + chosen_kv + chosen_compute,
        probes=len(probed),
    )

    if cache_path and cache_key:
        save_fit_cache(
            cache_path,
            cache_key,
            {
                **result.to_dict(),
                "required_bytes": result.required_bytes,
            },
        )
    return result
