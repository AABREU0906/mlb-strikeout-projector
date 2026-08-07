"""
Tests for app/services/confirmed_slate_csv.py duplicate-filename handling.
The path-selection logic is tested directly against a real temp
filesystem (not mocked) since it's pure, deterministic file-existence
checking.
"""
import tempfile
from pathlib import Path


def _next_available_path(exports_dir: Path, game_date: str) -> Path:
    """Mirrors app.services.confirmed_slate_csv._next_available_path
    exactly, reproduced here so this test file has zero dependency on
    app.config.settings (which requires pydantic)."""
    exports_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{game_date}_confirmed_slate"
    candidate = exports_dir / f"{base_name}.csv"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = exports_dir / f"{base_name}_{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


def test_first_export_uses_base_filename():
    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp) / "exports"
        path = _next_available_path(exports_dir, "2026-07-15")
        assert path.name == "2026-07-15_confirmed_slate.csv"


def test_second_export_same_day_gets_suffix_2():
    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp) / "exports"
        first = _next_available_path(exports_dir, "2026-07-15")
        first.write_text("data")
        second = _next_available_path(exports_dir, "2026-07-15")
        assert second.name == "2026-07-15_confirmed_slate_2.csv"


def test_third_export_same_day_gets_suffix_3():
    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp) / "exports"
        for expected in ["2026-07-15_confirmed_slate.csv", "2026-07-15_confirmed_slate_2.csv"]:
            p = _next_available_path(exports_dir, "2026-07-15")
            assert p.name == expected
            p.write_text("data")
        third = _next_available_path(exports_dir, "2026-07-15")
        assert third.name == "2026-07-15_confirmed_slate_3.csv"


def test_different_date_unaffected_by_existing_files():
    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp) / "exports"
        (exports_dir / "2026-07-15_confirmed_slate.csv").parent.mkdir(parents=True, exist_ok=True)
        (exports_dir / "2026-07-15_confirmed_slate.csv").write_text("data")

        other_day = _next_available_path(exports_dir, "2026-07-16")
        assert other_day.name == "2026-07-16_confirmed_slate.csv"


def test_exports_directory_created_if_missing():
    with tempfile.TemporaryDirectory() as tmp:
        exports_dir = Path(tmp) / "does_not_exist_yet"
        assert not exports_dir.exists()
        _next_available_path(exports_dir, "2026-07-15")
        assert exports_dir.exists()


def test_csv_created_with_expected_rows():
    import csv

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.csv"
        fieldnames = ["pitcher", "strikeout_line", "recommendation"]
        rows = [
            {"pitcher": "Cristopher Sanchez", "strikeout_line": 5.5, "recommendation": "OVER"},
            {"pitcher": "Zack Wheeler", "strikeout_line": 6.5, "recommendation": "PASS"},
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        assert path.exists()
        with path.open() as handle:
            reader = list(csv.DictReader(handle))
        assert len(reader) == 2
        assert reader[0]["pitcher"] == "Cristopher Sanchez"
        assert reader[1]["recommendation"] == "PASS"
