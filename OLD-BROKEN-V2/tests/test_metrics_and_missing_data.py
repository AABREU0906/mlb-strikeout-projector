from app.evaluation.metrics import bias, brier_score, calibration_buckets, log_loss, mae, medae, over_under_accuracy, rmse


def test_mae_empty_returns_none():
    assert mae([]) is None


def test_mae_basic():
    assert mae([1, -1, 2, -2]) == 1.5


def test_rmse_penalizes_large_errors_more_than_mae():
    errors = [1, 1, 1, 5]
    assert rmse(errors) > mae(errors)


def test_bias_direction():
    # predicted - actual; consistently over-projecting -> positive bias
    assert bias([1, 2, 1, 2]) > 0
    assert bias([-1, -2, -1, -2]) < 0


def test_brier_score_perfect_predictions():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_score_worst_case():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_log_loss_confident_wrong_prediction_penalized_heavily():
    confident_right = log_loss([0.95], [1])
    confident_wrong = log_loss([0.05], [1])
    assert confident_wrong > confident_right


def test_calibration_buckets_cover_all_predictions():
    preds = [0.1, 0.3, 0.5, 0.7, 0.9]
    outcomes = [0, 0, 1, 1, 1]
    buckets = calibration_buckets(preds, outcomes, n_buckets=5)
    assert sum(b["n"] for b in buckets) == len(preds)


def test_over_under_accuracy_all_correct():
    assert over_under_accuracy([True, False, True], [True, False, True]) == 1.0


def test_over_under_accuracy_all_wrong():
    assert over_under_accuracy([True, False], [False, True]) == 0.0


def test_medae_robust_to_outlier():
    errors = [1, 1, 1, 100]
    assert medae(errors) == 1
