import numpy as np
import pandas as pd

from electoral_forensics.outliers import flag_precincts, robust_zscore


def test_robust_z_is_insensitive_to_one_extreme_value():
    x = pd.Series([0.50, 0.51, 0.49, 0.52, 0.48, 0.99])
    z = robust_zscore(x)
    assert abs(z.iloc[-1]) > 10
    assert (z.iloc[:-1].abs() < 2).all()


def test_zero_mad_returns_nan():
    assert robust_zscore(pd.Series([1.0, 1.0, 1.0])).isna().all()


def _frame():
    rng = np.random.default_rng(3)
    rows = []
    for county, base in (("A", 0.3), ("B", 0.7)):
        for i in range(200):
            s = base + rng.normal(0, 0.03)
            rows.append({"county": county, "precinct": f"{county}{i}", "two_party": 500,
                         "dem_share": s, "mode_gap": rng.normal(0.1, 0.02)})
    df = pd.DataFrame(rows)
    df.loc[0, "dem_share"] = 0.95   # planted anomaly in county A
    df.loc[1, "two_party"] = 20     # tiny precinct: never flagged
    df.loc[1, "dem_share"] = 0.99
    return df


def test_flags_within_county_and_respects_min_votes():
    out = flag_precincts(_frame(), min_votes=100, z_threshold=4)
    assert out.loc[0, "flag"]
    assert not out.loc[1, "flag"]
    # county B's higher baseline share is not itself an anomaly
    assert out.loc[out.county == "B", "flag"].mean() < 0.02
