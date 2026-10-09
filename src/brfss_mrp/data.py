"""Load, clean, and aggregate BRFSS 2025 microdata into MRP cell counts.

Every variable name and code below was checked against the 2025 BRFSS codebook
(codebook25_llcp-v2-508.html, from
https://www.cdc.gov/brfss/annual_data/annual_2025.html) and against frequency
tables computed from LLCP2025.XPT, which matched the codebook exactly.

Survey weights (_LLCPWT) are deliberately not used: in MRP, adjustment for
non-representative sampling comes from poststratifying to census population
counts, not from the design weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 2025
RAW_COLUMNS = ["_STATE", "DIABETE4", "_AGEG5YR", "_SEX", "_EDUCAG"]

# 50 states + DC (FIPS -> (abbreviation, name)). The analysis scope.
STATES: dict[int, tuple[str, str]] = {
    1: ("AL", "Alabama"), 2: ("AK", "Alaska"), 4: ("AZ", "Arizona"),
    5: ("AR", "Arkansas"), 6: ("CA", "California"), 8: ("CO", "Colorado"),
    9: ("CT", "Connecticut"), 10: ("DE", "Delaware"),
    11: ("DC", "District of Columbia"), 12: ("FL", "Florida"),
    13: ("GA", "Georgia"), 15: ("HI", "Hawaii"), 16: ("ID", "Idaho"),
    17: ("IL", "Illinois"), 18: ("IN", "Indiana"), 19: ("IA", "Iowa"),
    20: ("KS", "Kansas"), 21: ("KY", "Kentucky"), 22: ("LA", "Louisiana"),
    23: ("ME", "Maine"), 24: ("MD", "Maryland"), 25: ("MA", "Massachusetts"),
    26: ("MI", "Michigan"), 27: ("MN", "Minnesota"), 28: ("MS", "Mississippi"),
    29: ("MO", "Missouri"), 30: ("MT", "Montana"), 31: ("NE", "Nebraska"),
    32: ("NV", "Nevada"), 33: ("NH", "New Hampshire"), 34: ("NJ", "New Jersey"),
    35: ("NM", "New Mexico"), 36: ("NY", "New York"),
    37: ("NC", "North Carolina"), 38: ("ND", "North Dakota"), 39: ("OH", "Ohio"),
    40: ("OK", "Oklahoma"), 41: ("OR", "Oregon"), 42: ("PA", "Pennsylvania"),
    44: ("RI", "Rhode Island"), 45: ("SC", "South Carolina"),
    46: ("SD", "South Dakota"), 47: ("TN", "Tennessee"), 48: ("TX", "Texas"),
    49: ("UT", "Utah"), 50: ("VT", "Vermont"), 51: ("VA", "Virginia"),
    53: ("WA", "Washington"), 54: ("WV", "West Virginia"),
    55: ("WI", "Wisconsin"), 56: ("WY", "Wyoming"),
}

# DIABETE4: 1 Yes; 2 Yes, female told only during pregnancy; 3 No;
# 4 No, pre-diabetes or borderline; 7 Don't know; 9 Refused; blank missing.
# Outcome follows CDC's "diagnosed diabetes" definition: 1 -> 1; 2, 3, 4 -> 0.
DIABETES_CODING = {1: 1, 2: 0, 3: 0, 4: 0}

# _AGEG5YR: 1 = 18-24, 2 = 25-29, ..., 13 = 80+, 14 = don't know/refused/missing.
# Collapsed to six groups.
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_CODING = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4,
              10: 5, 11: 5, 12: 5, 13: 5}

# _SEX: 1 Male, 2 Female (no missing values in 2025).
SEX_LABELS = ["Male", "Female"]
SEX_CODING = {1: 0, 2: 1}

# _EDUCAG: 1 did not graduate HS; 2 graduated HS; 3 attended college/technical
# school; 4 graduated college/technical school; 9 don't know/missing.
EDUC_LABELS = ["Less than HS", "HS graduate", "Some college", "College graduate"]
EDUC_CODING = {1: 0, 2: 1, 3: 2, 4: 3}

CELL_KEYS = ["state_fips", "age", "sex", "educ"]


def load_raw(xpt_path: str | Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Read only the needed columns from the BRFSS SAS transport file."""
    parts = []
    for chunk in pd.read_sas(xpt_path, format="xport", chunksize=chunksize, encoding="latin-1"):
        missing = set(RAW_COLUMNS) - set(chunk.columns)
        if missing:
            raise KeyError(f"Columns not found in {xpt_path}: {sorted(missing)}")
        parts.append(chunk[RAW_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exclusions in a fixed order and recode to 0-based integer indices.

    Returns (respondent-level clean data, exclusion log). Each log row records
    how many respondents a step removed, given that earlier steps already ran.
    """
    df = raw.copy()
    log = [{"step": "Start: all respondents in file", "removed": 0, "remaining": len(df)}]

    def exclude(mask: pd.Series, reason: str) -> None:
        nonlocal df
        removed = int(mask.sum())
        df = df.loc[~mask]
        log.append({"step": reason, "removed": removed, "remaining": len(df)})

    exclude(~df["_STATE"].isin(STATES), "Outside 50 states + DC (Guam, Puerto Rico)")
    exclude(~df["DIABETE4"].isin(DIABETES_CODING),
            "Diabetes: don't know (7), refused (9), or missing")
    exclude(~df["_AGEG5YR"].isin(AGE_CODING), "Age: don't know/refused/missing (_AGEG5YR=14)")
    exclude(~df["_SEX"].isin(SEX_CODING), "Sex: missing")
    exclude(~df["_EDUCAG"].isin(EDUC_CODING), "Education: don't know/missing (_EDUCAG=9)")

    clean_df = pd.DataFrame({
        "state_fips": df["_STATE"].astype(int),
        "age": df["_AGEG5YR"].map(AGE_CODING).astype(int),
        "sex": df["_SEX"].map(SEX_CODING).astype(int),
        "educ": df["_EDUCAG"].map(EDUC_CODING).astype(int),
        "diabetes": df["DIABETE4"].map(DIABETES_CODING).astype(int),
    }).reset_index(drop=True)
    return clean_df, pd.DataFrame(log)


def state_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    """All 51 geographies with a 0-based model index and whether BRFSS has data."""
    observed = set(clean_df["state_fips"].unique())
    rows = [
        {"state_idx": i, "state_fips": fips, "state_abbr": abbr, "state_name": name,
         "in_sample": fips in observed}
        for i, (fips, (abbr, name)) in enumerate(sorted(STATES.items()))
    ]
    return pd.DataFrame(rows)


def aggregate_cells(clean_df: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Collapse respondents to state x age x sex x education cells with n and y.

    Only cells with at least one respondent are returned; empty cells carry no
    likelihood information and are filled in later by poststratification.
    """
    cells = (clean_df.groupby(CELL_KEYS, as_index=False)
             .agg(n=("diabetes", "size"), y=("diabetes", "sum")))
    cells = cells.merge(states[["state_fips", "state_idx", "state_abbr"]], on="state_fips",
                        how="left", validate="many_to_one")
    cells["age_label"] = np.array(AGE_LABELS)[cells["age"]]
    cells["sex_label"] = np.array(SEX_LABELS)[cells["sex"]]
    cells["educ_label"] = np.array(EDUC_LABELS)[cells["educ"]]
    return cells[["state_idx", "state_fips", "state_abbr", "age", "sex", "educ",
                  "age_label", "sex_label", "educ_label", "n", "y"]]
