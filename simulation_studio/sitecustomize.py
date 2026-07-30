from __future__ import annotations

import sys
from pathlib import Path


# When the server is launched from inside `simulation_studio/`, Python's import
# path points at this directory instead of the repository root. Adding the
# parent directory restores imports like `simulation_studio.app.main:app`.
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
