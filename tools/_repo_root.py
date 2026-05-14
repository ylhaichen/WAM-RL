"""CLI helper for running repo tools as script paths.

When Python executes `python tools/foo.py`, `sys.path[0]` is `tools/`, not the
repository root. Tools that import `tools.*` or `wan_va.*` must add the root
explicitly before those imports.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
