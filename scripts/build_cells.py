"""Clean BRFSS microdata and aggregate it to MRP cells.

Usage:
    python scripts/build_cells.py [--xpt data/raw/LLCP2025.XPT]

Writes:
    data/processed/cells_2025.csv     state x age x sex x education: n, y (git-ignored)
    data/processed/states_2025.csv    51 geographies with model index (git-ignored)
    reports/exclusions_2025.csv       exclusion log (committed; aggregate counts only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brfss_mrp.data import YEAR, aggregate_cells, clean, load_raw, state_table

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpt", type=Path, default=ROOT / "data" / "raw" / f"LLCP{YEAR}.XPT")
    args = parser.parse_args()

    raw = load_raw(args.xpt)
    clean_df, log = clean(raw)
    states = state_table(clean_df)
    cells = aggregate_cells(clean_df, states)

    processed = ROOT / "data" / "processed"
    reports = ROOT / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    cells.to_csv(processed / f"cells_{YEAR}.csv", index=False)
    states.to_csv(processed / f"states_{YEAR}.csv", index=False)
    log.to_csv(reports / f"exclusions_{YEAR}.csv", index=False)

    print("Exclusions (applied in order):")
    print(log.to_string(index=False))
    missing = states.loc[~states["in_sample"], "state_name"].tolist()
    n_possible = len(states) * 6 * 2 * 4
    print(f"\nGeographies with data: {int(states['in_sample'].sum())} of {len(states)}; no data: {missing}")
    print(f"Respondents: {len(clean_df):,}; diagnosed diabetes: {int(clean_df['diabetes'].sum()):,} "
          f"(unweighted {clean_df['diabetes'].mean():.4f})")
    print(f"Non-empty cells: {len(cells):,} of {n_possible:,} possible (51 x 6 x 2 x 4)")
    print(f"Respondents per non-empty cell: min {cells['n'].min()}, median {cells['n'].median():.0f}, "
          f"max {cells['n'].max()}")
    assert cells["n"].sum() == len(clean_df) and cells["y"].sum() == clean_df["diabetes"].sum()


if __name__ == "__main__":
    main()
