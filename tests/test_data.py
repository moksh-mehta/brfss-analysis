import pandas as pd
import pytest

from brfss_mrp.data import aggregate_cells, clean, state_table

# Columns: _STATE, DIABETE4, _AGEG5YR, _SEX, _EDUCAG
ROWS = [
    (1, 1, 1, 1, 4),       # AL, yes, 18-24, M, college grad      -> kept, y=1
    (1, 2, 2, 2, 2),       # gestational only                     -> kept, y=0
    (1, 4, 3, 2, 2),       # pre-diabetes, age 30-34 -> 25-34     -> kept, y=0
    (1, 3, 13, 1, 1),      # no, 80+ -> 65+                       -> kept, y=0
    (1, 3, 8, 1, 1),       # no, 55-59 -> 45-64                   -> kept, y=0
    (72, 1, 5, 1, 2),      # Puerto Rico                          -> excluded
    (1, 7, 5, 1, 2),       # diabetes don't know                  -> excluded
    (1, 9, 5, 1, 2),       # diabetes refused                     -> excluded
    (1, None, 5, 1, 2),    # diabetes missing                     -> excluded
    (1, 1, 14, 1, 2),      # age missing                          -> excluded
    (1, 1, 5, 1, 9),       # education missing                    -> excluded
    (56, 1, 3, 2, 2),      # WY, yes, 30-34 -> 25-34              -> kept, y=1
]
RAW = pd.DataFrame(ROWS, columns=["_STATE", "DIABETE4", "_AGEG5YR", "_SEX", "_EDUCAG"], dtype=float)


@pytest.fixture
def cleaned():
    return clean(RAW)


def test_outcome_coding(cleaned):
    df, _ = cleaned
    assert len(df) == 6
    assert df["diabetes"].tolist() == [1, 0, 0, 0, 0, 1]


def test_recodes(cleaned):
    df, _ = cleaned
    assert df["age"].tolist() == [0, 1, 1, 4, 3, 1]
    assert df["sex"].tolist() == [0, 1, 1, 0, 0, 1]
    assert df["educ"].tolist() == [3, 1, 1, 0, 0, 1]


def test_exclusion_log_accounts_for_every_row(cleaned):
    df, log = cleaned
    assert log["removed"].tolist() == [0, 1, 3, 1, 0, 1]
    assert log["remaining"].iloc[0] - log["removed"].sum() == log["remaining"].iloc[-1] == len(df)


def test_state_table_covers_51_with_index(cleaned):
    df, _ = cleaned
    states = state_table(df)
    assert len(states) == 51
    assert states["state_idx"].tolist() == list(range(51))
    assert set(states.loc[states["in_sample"], "state_abbr"]) == {"AL", "WY"}


def test_aggregation_preserves_totals(cleaned):
    df, _ = cleaned
    cells = aggregate_cells(df, state_table(df))
    assert cells["n"].sum() == len(df)
    assert cells["y"].sum() == df["diabetes"].sum()
    # AL F 25-34 HS-grad: gestational-only + pre-diabetes respondents share a cell.
    cell = cells.query("state_abbr == 'AL' and age == 1 and sex == 1 and educ == 1")
    assert cell[["n", "y"]].values.tolist() == [[2, 0]]
