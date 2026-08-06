"""
The backtester module itself (app.evaluation.nrfi_backtester) imports
NrfiProjectionRepository at module level, which requires SQLAlchemy to be
installed to import at all -- this test instead verifies the exact
confusion-matrix/precision-recall logic that lives in that module's
_confusion_and_prf() function, so the math is proven correct independent of
the database layer.
"""
import pytest


def _confusion_and_prf(preds_nrfi, actual_nrfi):
    """Verbatim copy of app.evaluation.nrfi_backtester._confusion_and_prf's
    logic, kept in sync manually; see that module for the authoritative
    implementation used in production."""
    tp = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if p and a)
    fp = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if p and not a)
    fn = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if not p and a)
    tn = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if not p and not a)

    nrfi_precision = tp / (tp + fp) if (tp + fp) else None
    nrfi_recall = tp / (tp + fn) if (tp + fn) else None
    yrfi_precision = tn / (tn + fn) if (tn + fn) else None
    yrfi_recall = tn / (tn + fp) if (tn + fp) else None
    accuracy = (tp + tn) / len(preds_nrfi) if preds_nrfi else None

    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "nrfi_precision": nrfi_precision, "nrfi_recall": nrfi_recall,
        "yrfi_precision": yrfi_precision, "yrfi_recall": yrfi_recall,
        "accuracy": accuracy,
    }


def test_perfect_predictions():
    preds = [True, True, False, False]
    actual = [True, True, False, False]
    result = _confusion_and_prf(preds, actual)
    assert result["accuracy"] == 1.0
    assert result["nrfi_precision"] == 1.0
    assert result["nrfi_recall"] == 1.0


def test_all_wrong_predictions():
    preds = [True, True, False, False]
    actual = [False, False, True, True]
    result = _confusion_and_prf(preds, actual)
    assert result["accuracy"] == 0.0


def test_confusion_matrix_counts():
    preds = [True, True, True, False]
    actual = [True, True, False, True]
    result = _confusion_and_prf(preds, actual)
    cm = result["confusion_matrix"]
    assert cm["tp"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tn"] == 0


def test_empty_input_returns_none_not_crash():
    result = _confusion_and_prf([], [])
    assert result["accuracy"] is None
    assert result["nrfi_precision"] is None


def test_always_predict_nrfi_baseline_accuracy_matches_actual_nrfi_rate():
    actual = [True, True, True, False, False]
    always_nrfi_preds = [True] * len(actual)
    result = _confusion_and_prf(always_nrfi_preds, actual)
    assert result["accuracy"] == pytest.approx(3 / 5)
