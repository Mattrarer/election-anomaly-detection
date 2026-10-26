"""Statistical screening tools for precinct-level election returns.

The tests in this package flag *statistical irregularities*. An irregularity is a
reason to look closer, not evidence of fraud: digit tests are sensitive to precinct
size, competitiveness and strategic behaviour (Deckert, Myagkov & Ordeshook, 2011;
Mebane, 2011).
"""

from .digits import (
    benford_first_digit_probs,
    benford_second_digit_probs,
    digit_test,
    first_digits,
    last_digits,
    second_digits,
    simulated_null,
)
from .outliers import robust_zscore, flag_precincts

__all__ = [
    "benford_first_digit_probs",
    "benford_second_digit_probs",
    "digit_test",
    "first_digits",
    "second_digits",
    "last_digits",
    "simulated_null",
    "robust_zscore",
    "flag_precincts",
]

__version__ = "0.1.0"
