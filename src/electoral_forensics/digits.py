"""Digit-based tests for vote counts: Benford first digit, Benford second digit (2BL)
and last-digit uniformity.

All functions take an array of non-negative integer counts. Counts too small to have
the requested digit are dropped (e.g. single-digit counts have no second digit), and
the number of values actually used is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def benford_first_digit_probs() -> np.ndarray:
    """P(first digit = d) for d = 1..9 under Benford's law."""
    d = np.arange(1, 10)
    return np.log10(1 + 1 / d)


def benford_second_digit_probs() -> np.ndarray:
    """P(second digit = d) for d = 0..9 under Benford's law (the '2BL' distribution)."""
    k = np.arange(1, 10)[:, None]
    d = np.arange(0, 10)[None, :]
    return np.log10(1 + 1 / (10 * k + d)).sum(axis=0)


def _clean(counts) -> np.ndarray:
    x = np.asarray(counts, dtype=float)
    x = x[np.isfinite(x)]
    if np.any(x < 0):
        raise ValueError("vote counts must be non-negative")
    return x.astype(np.int64)


def first_digits(counts, min_value: int = 1) -> np.ndarray:
    """First significant digit of each count >= max(min_value, 1)."""
    x = _clean(counts)
    x = x[x >= max(min_value, 1)]
    return (x // 10 ** np.floor(np.log10(x)).astype(np.int64)).astype(int)


def second_digits(counts, min_value: int = 10) -> np.ndarray:
    """Second significant digit of each count >= max(min_value, 10)."""
    x = _clean(counts)
    x = x[x >= max(min_value, 10)]
    p = np.floor(np.log10(x)).astype(np.int64)
    return ((x // 10 ** (p - 1)) % 10).astype(int)


def last_digits(counts, min_value: int = 10) -> np.ndarray:
    """Units digit of each count >= min_value.

    Small counts are excluded because their last digit is not close to uniform
    (Beber & Scacco, 2012 use the same kind of threshold).
    """
    x = _clean(counts)
    x = x[x >= min_value]
    return (x % 10).astype(int)


@dataclass
class DigitTestResult:
    test: str
    n: int
    observed: np.ndarray
    expected: np.ndarray
    chi2: float
    p_value: float
    mean_digit: float
    expected_mean: float
    mad: float  # mean absolute deviation between observed and expected proportions

    def as_dict(self) -> dict:
        return {
            "test": self.test,
            "n": self.n,
            "chi2": round(self.chi2, 3),
            "p_value": self.p_value,
            "mean_digit": round(self.mean_digit, 4),
            "expected_mean": round(self.expected_mean, 4),
            "mad": round(self.mad, 5),
        }


def simulated_null(counts, kind: str = "second", n_sims: int = 500, seed: int = 0,
                   min_value: int | None = None) -> dict:
    """Monte Carlo null that keeps the observed precinct-size distribution.

    Benford's law is only an approximation for vote counts: whether 2BL or a uniform
    last digit should hold depends on how large precincts are. Here each count is
    replaced by a Poisson draw with the same mean ("honest noise" around the observed
    count) and the chi-square statistic is recomputed. The Monte Carlo p-value is the
    share of simulations at least as extreme as the observed statistic.
    """
    x = _clean(counts)
    obs = digit_test(x, kind, min_value)
    rng = np.random.default_rng(seed)
    sims = np.array([digit_test(rng.poisson(x), kind, min_value).chi2 for _ in range(n_sims)])
    return {
        "test": kind,
        "observed_chi2": round(obs.chi2, 3),
        "benford_p_value": obs.p_value,
        "null_chi2_median": round(float(np.median(sims)), 3),
        "monte_carlo_p_value": round(float((np.sum(sims >= obs.chi2) + 1) / (n_sims + 1)), 4),
        "share_of_null_sims_rejected_at_1pct": round(
            float(np.mean([s > stats.chi2.ppf(0.99, obs.observed.size - 1) for s in sims])), 3),
    }


def digit_test(counts, kind: str = "second", min_value: int | None = None) -> DigitTestResult:
    """Chi-square goodness-of-fit of a digit distribution.

    kind: "first" (Benford 1BL), "second" (Benford 2BL) or "last" (uniform 0-9).
    """
    if kind == "first":
        digits = first_digits(counts, min_value or 1)
        support, probs = np.arange(1, 10), benford_first_digit_probs()
    elif kind == "second":
        digits = second_digits(counts, min_value or 10)
        support, probs = np.arange(0, 10), benford_second_digit_probs()
    elif kind == "last":
        digits = last_digits(counts, min_value or 10)
        support, probs = np.arange(0, 10), np.full(10, 0.1)
    else:
        raise ValueError("kind must be 'first', 'second' or 'last'")

    n = digits.size
    if n == 0:
        raise ValueError("no counts large enough for this test")
    observed = np.array([(digits == d).sum() for d in support])
    expected = probs * n
    chi2, p = stats.chisquare(observed, expected)
    return DigitTestResult(
        test=kind,
        n=int(n),
        observed=observed,
        expected=expected,
        chi2=float(chi2),
        p_value=float(p),
        mean_digit=float(digits.mean()),
        expected_mean=float((support * probs).sum()),
        mad=float(np.abs(observed / n - probs).mean()),
    )
