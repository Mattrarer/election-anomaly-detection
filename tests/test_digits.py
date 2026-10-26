import numpy as np
import pytest

from electoral_forensics.digits import (
    benford_first_digit_probs,
    benford_second_digit_probs,
    digit_test,
    first_digits,
    last_digits,
    second_digits,
)


def test_benford_probabilities_sum_to_one():
    assert benford_first_digit_probs().sum() == pytest.approx(1.0)
    assert benford_second_digit_probs().sum() == pytest.approx(1.0)


def test_known_second_digit_mean():
    # The 2BL mean is 4.187 (Mebane, 2006)
    p = benford_second_digit_probs()
    assert (np.arange(10) * p).sum() == pytest.approx(4.187, abs=1e-3)


def test_digit_extraction():
    x = [7, 12, 305, 4096, 90]
    assert first_digits(x).tolist() == [7, 1, 3, 4, 9]
    assert second_digits(x).tolist() == [2, 0, 0, 0]  # 7 has no second digit
    assert last_digits(x, min_value=10).tolist() == [2, 5, 6, 0]


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        digit_test([-1, 20, 30])


def test_benford_data_passes_and_uniform_data_fails():
    rng = np.random.default_rng(0)
    benford_like = np.floor(10 ** rng.uniform(1, 5, 20_000)).astype(int)   # log-uniform
    uniform = rng.integers(100, 1000, 20_000)                               # flat 100-999
    assert digit_test(benford_like, "first").p_value > 0.01
    assert digit_test(uniform, "first").p_value < 1e-6


def test_last_digit_detects_rounding():
    rng = np.random.default_rng(1)
    honest = rng.integers(10, 5_000, 5_000)
    rounded = (honest // 5) * 5          # counts ending only in 0 or 5
    assert digit_test(honest, "last").p_value > 0.01
    r = digit_test(rounded, "last")
    assert r.p_value < 1e-10
    assert set(np.nonzero(r.observed)[0]) == {0, 5}


def test_simulated_null_does_not_reject_honest_poisson_counts():
    from electoral_forensics.digits import simulated_null
    rng = np.random.default_rng(5)
    honest = rng.poisson(rng.lognormal(5, 1, 3_000))
    out = simulated_null(honest, "last", n_sims=200, seed=1)
    assert out["monte_carlo_p_value"] > 0.01
