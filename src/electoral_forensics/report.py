"""Command-line report: digit tests + within-county outlier screening.

Usage:
    python -m electoral_forensics.report --input data/pa_2020.csv --out reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import PA_2020_URL, load_openelections, presidential_precincts
from .digits import digit_test, simulated_null
from .outliers import flag_precincts

INK, MUTED, DEM_C, REP_C, EXP_C = "#1F2328", "#6E7781", "#2B6CD9", "#CF222E", "#8C959F"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def plot_digit_tests(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, kind, title in ((axes[0], "second", "Second digit vs. Benford (2BL)"),
                            (axes[1], "last", "Last digit vs. uniform")):
        x = np.arange(10)
        w = 0.38
        for i, (cand, color) in enumerate((("Biden", DEM_C), ("Trump", REP_C))):
            r = results[cand][kind]
            ax.bar(x + (i - 0.5) * w, r.observed / r.n, w, color=color, label=f"{cand} (n={r.n:,})")
        exp = results["Biden"][kind].expected / results["Biden"][kind].n
        ax.plot(x, exp, "o-", color=EXP_C, ms=4, lw=1.2, label="Expected")
        ax.set_xticks(x)
        ax.set_xlabel("Digit")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        _style(ax)
    axes[0].set_ylabel("Share of precincts")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outliers(flagged: pd.DataFrame, path: Path) -> None:
    d = flagged.dropna(subset=["z_gap"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(d["ed_share"], d["mail_share"], s=4, color=MUTED, alpha=0.35,
               label=f"Precincts with both modes reported (n={len(d):,})")
    f = d[d["flag_gap"]]
    ax.scatter(f["ed_share"], f["mail_share"], s=16, color=REP_C,
               label=f"Unusual gap for their county (n={len(f)})")
    ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls="--")
    ax.set_xlabel("Biden share — election day")
    ax.set_ylabel("Biden share — mail / absentee")
    ax.set_title("Mail vs. in-person vote share by precinct", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(input_path: str, out_dir: str, z: float = 4.0, min_votes: int = 100,
        n_sims: int = 500) -> dict:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    raw = load_openelections(input_path)
    prec = presidential_precincts(raw)

    results = {}
    for cand, col in (("Biden", "dem"), ("Trump", "rep")):
        results[cand] = {k: digit_test(prec[col], kind=k) for k in ("first", "second", "last")}

    # Is a Benford rejection meaningful? Compare against a null that keeps precinct sizes.
    null = {cand: {k: simulated_null(prec[col], kind=k, n_sims=n_sims) for k in ("second", "last")}
            for cand, col in (("Biden", "dem"), ("Trump", "rep"))}
    # Last-digit tests are sensitive to small counts: re-run with stricter thresholds.
    thresholds = {cand: {str(mv): round(digit_test(prec[col], "last", min_value=mv).p_value, 4)
                         for mv in (10, 50, 100, 200)}
                  for cand, col in (("Biden", "dem"), ("Trump", "rep"))}

    flagged = flag_precincts(prec, z_threshold=z, min_votes=min_votes)
    with_modes = flagged["mode_gap"].notna()
    summary = {
        "precincts": int(len(prec)),
        "counties": int(prec["county"].nunique()),
        "total_votes": {"Biden": int(prec["dem"].sum()), "Trump": int(prec["rep"].sum())},
        "precincts_with_mode_breakdown": int(with_modes.sum()),
        "counties_with_mode_breakdown": int(prec.loc[with_modes, "county"].nunique()),
        "digit_tests": {c: {k: r.as_dict() for k, r in res.items()} for c, res in results.items()},
        "size_preserving_null": null,
        "last_digit_p_by_min_count": thresholds,
        "outliers": {
            "z_threshold": z,
            "min_two_party_votes": min_votes,
            "eligible_precincts": int((flagged["two_party"] >= min_votes).sum()),
            "flagged": int(flagged["flag"].sum()),
            "flagged_share": round(float(flagged["flag"].mean()), 4),
            "flagged_on_share": int(flagged["flag_share"].sum()),
            "flagged_on_mode_gap": int(flagged["flag_gap"].sum()),
            "flagged_by_county_top5": flagged[flagged["flag"]]["county"].value_counts().head(5).to_dict(),
        },
    }

    plot_digit_tests(results, out / "figures" / "digit_tests.png")
    plot_outliers(flagged, out / "figures" / "mode_gap_outliers.png")
    cols = ["county", "precinct", "dem", "rep", "two_party", "dem_share", "ed_share",
            "mail_share", "mode_gap", "ed_votes", "mail_votes", "z_share", "z_gap",
            "flag_share", "flag_gap"]
    (flagged[flagged["flag"]][cols]
        .sort_values("z_gap", key=lambda s: s.abs(), ascending=False)
        .to_csv(out / "flagged_precincts.csv", index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=PA_2020_URL, help="OpenElections precinct CSV (path or URL)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--z", type=float, default=4.0, help="robust z-score threshold")
    ap.add_argument("--min-votes", type=int, default=100)
    ap.add_argument("--sims", type=int, default=500, help="Monte Carlo draws for the null")
    a = ap.parse_args()
    s = run(a.input, a.out, a.z, a.min_votes, a.sims)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
