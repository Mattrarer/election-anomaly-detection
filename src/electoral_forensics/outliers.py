"""Robust outlier screening for precinct results.

Precincts are compared with *their own county*, so that ordinary political geography
(urban vs. rural counties) is not mistaken for an anomaly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SD = 1.4826  # makes the MAD consistent with the SD under normality


def robust_zscore(x: pd.Series) -> pd.Series:
    """(x - median) / (1.4826 * MAD). Returns NaN where the MAD is zero."""
    med = x.median()
    mad = (x - med).abs().median() * MAD_TO_SD
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - med) / mad


def flag_precincts(
    df: pd.DataFrame,
    group: str = "county",
    share_col: str = "dem_share",
    gap_col: str | None = "mode_gap",
    min_votes: int = 100,
    z_threshold: float = 4.0,
    min_mode_votes: int = 25,
) -> pd.DataFrame:
    """Add within-group robust z-scores and a flag column.

    A precinct is flagged when |z| > z_threshold for the vote share, or for the
    gap between mail and in-person vote shares when that column is available.
    Precincts with fewer than ``min_votes`` two-party votes are not flagged, and the
    mode gap is only used when both modes have at least ``min_mode_votes`` votes,
    because shares from tiny counts are extremely noisy.
    """
    out = df.copy()
    eligible = out["two_party"] >= min_votes
    out["z_share"] = np.nan
    out.loc[eligible, "z_share"] = (
        out.loc[eligible].groupby(group)[share_col].transform(robust_zscore)
    )
    flags = out["z_share"].abs() > z_threshold

    if gap_col and gap_col in out:
        ok = eligible & out[gap_col].notna()
        if {"ed_votes", "mail_votes"} <= set(out.columns):
            ok &= (out["ed_votes"] >= min_mode_votes) & (out["mail_votes"] >= min_mode_votes)
        out["z_gap"] = np.nan
        out.loc[ok, "z_gap"] = out.loc[ok].groupby(group)[gap_col].transform(robust_zscore)
        gap_flag = out["z_gap"].abs() > z_threshold
        flags |= gap_flag
    else:
        gap_flag = pd.Series(False, index=out.index)

    out["flag_share"] = (out["z_share"].abs() > z_threshold).fillna(False)
    out["flag_gap"] = gap_flag.fillna(False)
    out["flag"] = flags.fillna(False)
    return out
