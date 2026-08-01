import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.orchestrator.context_fitting import (
    DeviceMemory,
    FitResult,
    _fit_cache_key,
    _parse_context_memory,
    align_down,
    cached_fit_still_fits,
    choose_context_size,
    load_fit_cache,
    save_fit_cache,
)


def _always_fits_preflight(free_bytes: int = 1024 * 1024, reserve_bytes: int = 0):
    return lambda n_ctx: [DeviceMemory(1024, free_bytes, reserve_bytes)]


def _fits_up_to_preflight(limit: int, free_bytes: int = 1024 * 1024):
    def preflight(n_ctx: int) -> list[DeviceMemory]:
        required = 1024 if n_ctx <= limit else 2**62
        return [DeviceMemory(required, free_bytes, 0)]

    return preflight


class ChooseContextSizeTests(unittest.TestCase):
    NATIVE = 262144

    def test_native_window_is_used_when_it_fits(self):
        self.assertEqual(
            choose_context_size(
                native_context=self.NATIVE,
                preflight=_always_fits_preflight(),
            ),
            262144,
        )

    def test_largest_fitting_window_is_chosen(self):
        self.assertEqual(
            choose_context_size(
                native_context=self.NATIVE,
                preflight=_fits_up_to_preflight(100000),
                minimum_context=4096,
                alignment=256,
            ),
            99840,  # align_down(100000, 256)
        )

    def test_maximum_context_cap_is_honoured(self):
        self.assertEqual(
            choose_context_size(
                native_context=self.NATIVE,
                preflight=_always_fits_preflight(),
                maximum_context=8192,
            ),
            8192,
        )

    def test_candidates_are_aligned_to_256(self):
        chosen = choose_context_size(
            native_context=self.NATIVE,
            preflight=_fits_up_to_preflight(10000),
            minimum_context=4096,
            alignment=256,
        )
        self.assertEqual(chosen % 256, 0)
        self.assertLessEqual(chosen, 10000)

    def test_memory_error_when_minimum_does_not_fit(self):
        with self.assertRaises(MemoryError):
            choose_context_size(
                native_context=self.NATIVE,
                preflight=_fits_up_to_preflight(0),
                minimum_context=4096,
            )

    def test_empty_preflight_result_does_not_fit(self):
        with self.assertRaises(MemoryError):
            choose_context_size(
                native_context=self.NATIVE,
                preflight=lambda n_ctx: [],
                minimum_context=4096,
            )

    def test_invalid_native_context_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_context_size(
                native_context=0,
                preflight=lambda n: [DeviceMemory(1, 1, 0)],
            )

    def test_reserve_bytes_reduces_available_memory(self):
        free_bytes = 4096
        required = 2048
        reserve = 2049
        preflight = lambda n_ctx: [DeviceMemory(required, free_bytes, reserve)]
        with self.assertRaises(MemoryError):
            choose_context_size(
                native_context=self.NATIVE,
                preflight=preflight,
                minimum_context=4096,
            )
        reserve = 1024
        preflight = lambda n_ctx: [DeviceMemory(required, free_bytes, reserve)]
        self.assertEqual(
            choose_context_size(
                native_context=self.NATIVE,
                preflight=preflight,
                minimum_context=4096,
                maximum_context=8192,
            ),
            8192,
        )


class AlignDownTests(unittest.TestCase):
    def test_align_down_rounds_down(self):
        self.assertEqual(align_down(4096, 256), 4096)
        self.assertEqual(align_down(4200, 256), 4096)
        self.assertEqual(align_down(255, 256), 0)

    def test_align_down_rejects_bad_alignment(self):
        with self.assertRaises(ValueError):
            align_down(4096, 0)


class ContextMemoryParseTests(unittest.TestCase):
    SAMPLE = (
        "llama_model_load: model size = 6.96 GiB (6960054464 bytes)\n"
        "llama_kv_cache: size = 100.25 MiB\n"
        "sched_reserve: Metal compute buffer size = 120.50 MiB\n"
        "sched_reserve: CPU compute buffer size = 3.00 MiB\n"
        "llama_context: Metal output buffer size = 8.00 MiB\n"
    )

    def test_parses_llama_owned_accounting(self):
        kv_bytes, compute_bytes, output_bytes = _parse_context_memory(self.SAMPLE)
        self.assertIsNotNone(kv_bytes)
        self.assertIsNotNone(compute_bytes)
        self.assertIsNotNone(output_bytes)
        self.assertEqual(kv_bytes, int(round(100.25 * 1024 * 1024)))
        # All per-device compute buffers are reserved, so they are summed.
        self.assertEqual(
            compute_bytes,
            int(round((120.50 + 3.00) * 1024 * 1024)),
        )
        self.assertEqual(output_bytes, int(round(8.00 * 1024 * 1024)))

    def test_unparseable_stderr_returns_none(self):
        self.assertIsNone(
            _parse_context_memory("llama_model_load: logging noisy output")
        )
        self.assertIsNone(_parse_context_memory(""))


class FitCacheTests(unittest.TestCase):
    def test_fit_cache_key_is_stable(self):
        key1 = _fit_cache_key(
            "/tmp/model.gguf",
            {"n_batch": 512, "n_gpu_layers": -1},
            4096,
            None,
            256,
            1073741824,
            4096,
        )
        key2 = _fit_cache_key(
            "/tmp/model.gguf",
            {"n_gpu_layers": -1, "n_batch": 512},
            4096,
            None,
            256,
            1073741824,
            4096,
        )
        self.assertEqual(key1, key2)
        key3 = _fit_cache_key(
            "/tmp/model.gguf",
            {"n_batch": 512, "n_gpu_layers": -1},
            4096,
            None,
            256,
            1073741824,
            2048,
        )
        self.assertNotEqual(key1, key3)

    def test_save_and_load_roundtrip(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "context_fit_cache.json"
            entry = FitResult(
                chosen_context=4096,
                native_context=262144,
                model_bytes=100,
                context_bytes=50,
                compute_bytes=10,
                free_bytes=1000,
                required_bytes=160,
                cached=False,
            ).to_dict()
            save_fit_cache(cache_path, "model-a", entry)
            saved = load_fit_cache(cache_path)
            self.assertEqual(saved["model-a"]["chosen_context"], 4096)

    def test_cached_fit_still_fits_checks_current_free_memory(self):
        entry = {"required_bytes": 100}
        self.assertTrue(cached_fit_still_fits(entry, 200, 0))
        self.assertTrue(cached_fit_still_fits(entry, 150, 50))
        self.assertFalse(cached_fit_still_fits(entry, 149, 50))
        self.assertFalse(cached_fit_still_fits({}, 1000, 0))


if __name__ == "__main__":
    unittest.main()
