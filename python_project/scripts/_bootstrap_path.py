"""Path bootstrap shared by every script in this folder.

Adds ``python_project/src`` to ``sys.path`` so scripts can be run from
the repository root as

    python python_project/scripts/01_run_data_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
