import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set an isolated temp database BEFORE any `app.*` module is imported --
# conftest.py is always loaded first by pytest, so this runs before any
# test module's own `from app... import ...` line, which is what matters
# since app.config.settings.settings (and the SQLAlchemy engine bound to
# it) is a module-level singleton created at first import. Without this,
# any DB-backed test would silently read/write the project's real
# data/database/*.db file.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="mlb_strikeout_projector_tests_")
os.environ.setdefault("DATABASE_PATH", str(Path(_TEST_DB_DIR) / "test.db"))

import pytest


@pytest.fixture
def league_avg_k():
    return 0.224


@pytest.fixture(scope="session", autouse=True)
def _initialize_test_database():
    """Ensures the isolated temp database (see module-level setup above)
    has its schema created before any DB-backed test runs. Runs once per
    test session. Individual tests call the real bet_ledger/repository
    functions directly (they manage their own sessions already), which
    exercises the actual production code path rather than a parallel
    test-only session helper."""
    from app.database.session import init_db

    init_db()
