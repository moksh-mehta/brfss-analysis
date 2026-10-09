# brfss-mrp

State-level prevalence of diagnosed diabetes in the US, estimated with
**multilevel regression and poststratification (MRP)** on the CDC Behavioral
Risk Factor Surveillance System (BRFSS), implemented in
[NumPyro](https://num.pyro.ai/) (JAX).

> **Status:** work in progress. No results are reported yet. Every number that
> appears in this README will be produced by code in this repository.

## Project layout

```
src/brfss_mrp/     library code (data cleaning, models, poststratification)
scripts/           command-line entry points (data download)
notebooks/         exploratory analysis and figures
tests/             pytest suite
data/raw/          downloaded BRFSS files (git-ignored, never committed)
data/processed/    cleaned/aggregated data (git-ignored)
```

## Setup

Requires Python 3.10+. The pinned versions in `requirements.txt` were resolved
on Python 3.10, which caps JAX at 0.6.x (newer JAX needs Python 3.11+).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest            # smoke test: JAX on CPU + a tiny NUTS fit
```

## Data

Source: CDC BRFSS **2025** combined landline and cell phone file
(`LLCP2025XPT.zip`, released August 2026), from
<https://www.cdc.gov/brfss/annual_data/annual_2025.html>. The microdata is
public but is not redistributed here.

```bash
python scripts/download_brfss.py --year 2025   # -> data/raw/LLCP2025.XPT + MANIFEST.json (SHA-256)
python scripts/build_cells.py                  # -> data/processed/cells_2025.csv
```

### Variables (checked against the 2025 codebook)

| Concept | BRFSS variable | Coding used |
|---|---|---|
| Diagnosed diabetes | `DIABETE4` | 1 = "Yes" (code 1). 0 = "No" (3), "pre-diabetes/borderline" (4), "only during pregnancy" (2). Don't know/refused/missing excluded. |
| State | `_STATE` | FIPS code; scope is the 50 states + DC |
| Age | `_AGEG5YR` | 5-year bands collapsed to 18-24, 25-34, 35-44, 45-54, 55-64, 65+ |
| Sex | `_SEX` | Male, Female |
| Education | `_EDUCAG` | Less than HS, HS graduate, Some college, College graduate |

Survey weights are not used. In MRP, the adjustment for non-representative
sampling comes from poststratifying to census population counts.

### Exclusions (applied in this order; from `reports/exclusions_2025.csv`)

| Step | Removed | Remaining |
|---|---:|---:|
| All respondents in file | | 356,158 |
| Outside 50 states + DC (Guam, Puerto Rico) | 5,772 | 350,386 |
| Diabetes: don't know, refused, or missing | 763 | 349,623 |
| Age missing (`_AGEG5YR` = 14) | 6,487 | 343,136 |
| Sex missing | 0 | 343,136 |
| Education missing (`_EDUCAG` = 9) | 1,435 | 341,701 |

**Analysis sample:** 341,701 respondents, 49,802 with diagnosed diabetes,
in 2,301 non-empty state x age x sex x education cells.

**Coverage gap:** California, Mississippi, and Nevada did not meet CDC's
minimum data requirements in 2025, so they have no respondents. The model still
produces estimates for them, but these come only from the multilevel
structure, not from any survey data in those states.

## Models

### Model 1: state varying intercepts (centered)

```
y_c     ~ Binomial(n_c, logistic(alpha[state_c]))    c = 2,301 non-empty cells
alpha_s ~ Normal(mu, sigma)                          s = 51 geographies
mu      ~ Normal(0, 1.5),   sigma ~ HalfNormal(1)    (logit scale)
```

```bash
python scripts/fit_model1.py                # NUTS, 4 chains x (1,000 warmup + 1,000 draws), seed 20251
python scripts/compare_parameterizations.py # centered vs non-centered, same settings
```

This is a baseline. `alpha_s` describes the respondents in each state's
sample, not its adult population, so it is not yet a prevalence estimate.
That requires demographic predictors and poststratification.

**Convergence** (`reports/model1_diagnostics.json`): 0 divergences in 4,000
draws; max R-hat 1.0042; min bulk ESS 4,825; min tail ESS 2,211; no draws hit
the maximum tree depth.

**Why centered, not non-centered.** The non-centered form
(`alpha_s = mu + sigma * z_s`) is the usual default for hierarchical models,
and it was fitted first. It failed the convergence checks
(`reports/model1_parameterization.csv`):

| | Centered | Non-centered |
|---|---:|---:|
| Divergences | 0 | 0 |
| Max R-hat | 1.0042 | 1.0191 |
| Min bulk ESS | 4,825 | 168 |
| Min tail ESS | 2,211 | 353 |
| Median leapfrog steps per draw | 7 | 15 |
| Posterior mean of `mu` | -1.7569 | -1.7587 |
| Posterior mean of `sigma` | 0.2101 | 0.2098 |

Every observed state has thousands of respondents, so each `alpha_s` is
tightly determined by its own data. In the non-centered form, that forces `mu`
and every `z_s` to move together: their posterior correlation has a median of
-0.55 (minimum -0.74) across observed states, and NUTS mixes slowly along that
ridge. Non-centering pays off when groups have little data. That is likely to
be true for the finer demographic interactions in later models.
