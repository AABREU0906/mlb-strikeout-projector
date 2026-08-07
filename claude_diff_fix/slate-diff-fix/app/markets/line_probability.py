"""
Shared over/under/push probability calculation from a strikeout
probability distribution.

BUG THIS FIXES: the previous inline calculation in
app/reporting/display.py computed `model_under_probability = 1.0 -
model_over_probability` unconditionally. That is only correct for a
half-point line (e.g. 4.5), where no exact-match ("push") outcome is
possible. For a WHOLE-NUMBER line (e.g. 5.0), `over = P(k > 5)` correctly
excludes a push at k=5, but `1 - over` then equals `P(k <= 5)`, which
silently folds the push probability into "under" rather than keeping it
separate -- exactly the "whole-number push handling" the audit calls out.

This module is the single place that computes over/under/push probability
so app/reporting/display.py and the projection validator both call the
same function and can never compute it two different (and potentially
inconsistent) ways.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LineProbabilities:
    line: float
    is_whole_number_line: bool
    over_probability: float
    under_probability: float
    push_probability: float

    @property
    def total(self) -> float:
        return self.over_probability + self.under_probability + self.push_probability


def compute_line_probabilities(probability_by_k: dict, line: float) -> LineProbabilities:
    """probability_by_k keys are integer strikeout totals (0..15, where 15
    represents "15 or more")."""
    is_whole = math.isclose(line, round(line), abs_tol=1e-9)

    if is_whole:
        line_int = round(line)
        over = sum(p for k, p in probability_by_k.items() if k > line_int)
        under = sum(p for k, p in probability_by_k.items() if k < line_int)
        push = probability_by_k.get(line_int, 0.0)
    else:
        floor_line = math.floor(line)
        over = sum(p for k, p in probability_by_k.items() if k > floor_line)
        under = sum(p for k, p in probability_by_k.items() if k < line)
        push = 0.0

    return LineProbabilities(
        line=line,
        is_whole_number_line=is_whole,
        over_probability=over,
        under_probability=under,
        push_probability=push,
    )
