"""Loading and reshaping OpenElections precinct files.

OpenElections publishes one long CSV per state and election
(https://github.com/openelections). The same data are mirrored on Kaggle
("Open Elections Data: USA"). Rows are precinct x office x candidate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PA_2020_URL = (
    "https://raw.githubusercontent.com/openelections/openelections-data-pa/"
    "master/2020/20201103__pa__general__precinct.csv"
)

DEM = "Joseph R. Biden"
REP = "Donald J. Trump"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_openelections(path_or_url: str | Path) -> pd.DataFrame:
    """Read an OpenElections precinct CSV and normalise numeric columns."""
    df = pd.read_csv(path_or_url, low_memory=False)
    for col in ("votes", "election_day", "mail", "absentee", "provisional"):
        if col in df:
            df[col] = _to_num(df[col])
    # Some counties include a county-level TOTAL row next to their precincts.
    df = df[~df["precinct"].astype(str).str.fullmatch(r"\s*total\s*", case=False)]
    return df


def presidential_precincts(df: pd.DataFrame, dem: str = DEM, rep: str = REP) -> pd.DataFrame:
    """One row per precinct with candidate totals, two-party share and, where the
    county reports it, the vote split by mode (election day vs. mail/absentee)."""
    p = df[(df["office"] == "President") & df["candidate"].isin([dem, rep])].copy()
    p["party_key"] = np.where(p["candidate"] == dem, "dem", "rep")
    p["mail_total"] = p[["mail", "absentee"]].sum(axis=1, min_count=1)

    wide = p.pivot_table(
        index=["county", "precinct"],
        columns="party_key",
        values=["votes", "election_day", "mail_total"],
        aggfunc="sum",
        dropna=False,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    out = pd.DataFrame({
        "county": wide["county"],
        "precinct": wide["precinct"],
        "dem": wide["votes_dem"],
        "rep": wide["votes_rep"],
    })
    out["two_party"] = out["dem"] + out["rep"]
    out["dem_share"] = out["dem"] / out["two_party"].replace(0, np.nan)

    ed_two = wide["election_day_dem"] + wide["election_day_rep"]
    mail_two = wide["mail_total_dem"] + wide["mail_total_rep"]
    # pivot_table turns missing breakdowns into 0; treat 0/0 as "not reported"
    has_modes = (ed_two > 0) & (mail_two > 0)
    out["ed_share"] = np.where(has_modes, wide["election_day_dem"] / ed_two, np.nan)
    out["mail_share"] = np.where(has_modes, wide["mail_total_dem"] / mail_two, np.nan)
    out["ed_votes"] = np.where(has_modes, ed_two, np.nan)
    out["mail_votes"] = np.where(has_modes, mail_two, np.nan)
    out["mode_gap"] = out["mail_share"] - out["ed_share"]
    return out.dropna(subset=["dem", "rep"])
