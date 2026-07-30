import tempfile
import unittest
from pathlib import Path

from infrastructure.scripts import select_python


class SelectPythonTests(unittest.TestCase):
    def test_requires_python_loaded_from_pyproject(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "pyproject.toml").write_text(
                '[project]\nname = "demo"\nrequires-python = ">=3.11,<3.14"\n',
                encoding="utf-8",
            )

            self.assertEqual(select_python.read_requires_python(tmp), ">=3.11,<3.14")

    def test_version_requirement_bounds(self):
        requirement = ">=3.11,<3.14"

        self.assertTrue(select_python.supports_requirement((3, 11, 0), requirement))
        self.assertTrue(select_python.supports_requirement((3, 13, 9), requirement))
        self.assertFalse(select_python.supports_requirement((3, 10, 9), requirement))
        self.assertFalse(select_python.supports_requirement((3, 14, 0), requirement))

    def test_selects_highest_compatible_candidate(self):
        requirement = ">=3.11,<3.14"
        candidates = ["/tmp/python-old", "/tmp/python-min", "/tmp/python-max", "/tmp/python-new"]

        original = select_python.interpreter_version
        versions = {
            "/tmp/python-old": (3, 10, 9),
            "/tmp/python-min": (3, 11, 15),
            "/tmp/python-max": (3, 13, 8),
            "/tmp/python-new": (3, 14, 0),
        }
        try:
            select_python.interpreter_version = versions.get
            self.assertEqual(
                select_python.select_interpreter(requirement, candidates),
                "/tmp/python-max",
            )
        finally:
            select_python.interpreter_version = original


if __name__ == "__main__":
    unittest.main()
