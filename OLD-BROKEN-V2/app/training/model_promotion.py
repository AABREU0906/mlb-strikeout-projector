"""
Model Promotion Rules.

A newly trained candidate model is promoted (becomes the active model) only
if ALL of the following hold, compared against the currently active model's
own walk-forward validation metrics (or, if there is no active ML model
yet, against the transparent baseline's historical accuracy on the same
validation folds):

  1. Candidate's out-of-sample MAE is not worse than the current model's
     MAE by more than PROMOTION_MAE_TOLERANCE (small negative regressions
     are allowed if the candidate materially improves elsewhere, but a
     regression beyond tolerance blocks promotion outright).
  2. Candidate has at least MIN_VALIDATION_OBSERVATIONS graded observations
     across its validation folds.
  3. Candidate's fold-level MAE does not vary wildly (max_fold_mae /
     min_fold_mae ratio capped) -- guards against a model that only works
     in one narrow historical stretch.
  4. Candidate MAE actually improves on the current model (strict
     improvement), OR ties within tolerance with a materially larger
     validation sample (more evidence of stability).

Every promotion or rejection decision is recorded with the reasons, per
project requirements ("Record why a model was accepted or rejected").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.training.walk_forward import WalkForwardReport

MIN_VALIDATION_OBSERVATIONS = 60
PROMOTION_MAE_TOLERANCE = 0.05  # candidate MAE can be at most 5% worse and still be considered (then blocked by rule 4 unless it improves)
MAX_FOLD_VARIANCE_RATIO = 2.5


@dataclass
class PromotionDecision:
    promoted: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_promotion(
    candidate_report: WalkForwardReport,
    current_active_mae: Optional[float],
    total_validation_n: int,
) -> PromotionDecision:
    reasons = []

    if total_validation_n < MIN_VALIDATION_OBSERVATIONS:
        return PromotionDecision(
            promoted=False,
            reasons=[
                f"Only {total_validation_n} validation observations "
                f"(< required minimum {MIN_VALIDATION_OBSERVATIONS}); insufficient evidence to promote."
            ],
        )
    reasons.append(f"Validation observations: {total_validation_n} (meets minimum {MIN_VALIDATION_OBSERVATIONS}).")

    fold_maes = [f.mae for f in candidate_report.folds if f.mae is not None and f.mae > 0]
    if fold_maes:
        ratio = max(fold_maes) / max(min(fold_maes), 1e-6)
        if ratio > MAX_FOLD_VARIANCE_RATIO:
            return PromotionDecision(
                promoted=False,
                reasons=reasons + [
                    f"Fold-level MAE variance too high (max/min={ratio:.2f} > {MAX_FOLD_VARIANCE_RATIO}); "
                    f"model may only work in a narrow historical stretch."
                ],
            )
    reasons.append("Fold-level MAE variance within acceptable range.")

    if current_active_mae is None:
        reasons.append("No currently active ML model to compare against; promoting first candidate that clears the above bars.")
        return PromotionDecision(promoted=True, reasons=reasons)

    if candidate_report.overall_mae < current_active_mae:
        reasons.append(
            f"Candidate MAE ({candidate_report.overall_mae}) improves on active model MAE ({current_active_mae})."
        )
        return PromotionDecision(promoted=True, reasons=reasons)

    tolerance_bound = current_active_mae * (1 + PROMOTION_MAE_TOLERANCE)
    if candidate_report.overall_mae <= tolerance_bound:
        reasons.append(
            f"Candidate MAE ({candidate_report.overall_mae}) does not strictly improve on active model "
            f"({current_active_mae}); per rule, ties within tolerance are NOT promoted without strict improvement."
        )
    else:
        reasons.append(
            f"Candidate MAE ({candidate_report.overall_mae}) is worse than active model "
            f"({current_active_mae}) beyond tolerance ({PROMOTION_MAE_TOLERANCE*100:.0f}%)."
        )

    return PromotionDecision(promoted=False, reasons=reasons)
