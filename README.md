# brfss-mrp

State-level prevalence of diagnosed diabetes in the US, estimated with
**multilevel regression and poststratification (MRP)** on the CDC Behavioral
Risk Factor Surveillance System (BRFSS), implemented in
[NumPyro](https://num.pyro.ai/) (JAX).

> **Status:** work in progress. Done so far: data pipeline, a baseline
> state-intercepts model (Model 1), and a demographic model with prior and
> posterior predictive checks (Model 2). Poststratified prevalence estimates are
> not yet produced. Every result reported here (counts, estimates, diagnostics)
> is produced by code in this repository.

## Project layout

```
src/brfss_mrp/     library code (data cleaning, models, NUTS fitting)
scripts/           command-line entry points (download, build cells, fit models)
notebooks/         exploratory analysis (empty so far)
tests/             pytest suite
reports/           small derived tables and figures cited in this README (committed)
outputs/           fitted models, e.g. ArviZ .nc files (git-ignored)
data/raw/          downloaded BRFSS files (git-ignored, never committed)
data/processed/    cleaned/aggregated data (git-ignored)
```

## Setup

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
| Age | `_AGEG5YR` | 5-year bands collapsed to 18-24, 25-34, 35-44, 45-64, 65+, matching the age groups of ACS table B15001 (the planned poststratification frame) |
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
in 1,918 non-empty state x age x sex x education cells (of 1,920 possible
for the 48 geographies with data).

**Coverage gap:** California, Mississippi, and Nevada did not meet CDC's
minimum data requirements in 2025, so they have no respondents. The model still
produces estimates for them, but these come only from the multilevel
structure, not from any survey data in those states.

## Models

### Model 1: state varying intercepts (centered)

```
y_c     ~ Binomial(n_c, logistic(alpha[state_c]))    c = 1,918 non-empty cells
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
draws; max R-hat 1.0033; min bulk ESS 4,607; min tail ESS 2,637; no draws hit
the maximum tree depth.

**Why centered, not non-centered.** The non-centered form
(`alpha_s = mu + sigma * z_s`) is the usual default for hierarchical models,
and it was fitted first. It failed the convergence checks
(`reports/model1_parameterization.csv`):

| | Centered | Non-centered |
|---|---:|---:|
| Divergences | 0 | 0 |
| Max R-hat | 1.0033 | 1.0185 |
| Min bulk ESS | 4,607 | 238 |
| Min tail ESS | 2,637 | 394 |
| Median leapfrog steps per draw | 7 | 15 |
| Posterior mean of `mu` | -1.7572 | -1.7552 |
| Posterior mean of `sigma` | 0.2103 | 0.2093 |

Every observed state has thousands of respondents, so each `alpha_s` is
tightly determined by its own data. In the non-centered form, that forces `mu`
and every `z_s` to move together: their posterior correlation has a median of
-0.58 (minimum -0.76) across observed states, and NUTS mixes slowly along that
ridge. Non-centering pays off when groups have little data. That is likely to
be true for the finer demographic interactions in later models.

### Model 2: state, age, and education intercepts + sex

```
y_c   ~ Binomial(n_c, logistic(eta_c))
eta_c = mu + a_state[s] + a_age[a] + a_educ[e] + b_sex * sex_c     sex_c = -0.5 male, +0.5 female
a_state ~ Normal(0, sigma_state),  a_age ~ Normal(0, sigma_age),  a_educ ~ Normal(0, sigma_educ)
mu ~ Normal(-2, 1),  b_sex ~ Normal(0, 0.5)
sigma_state ~ HalfNormal(0.5),  sigma_age ~ HalfNormal(1),  sigma_educ ~ HalfNormal(0.5)
```

```bash
python scripts/fit_model2.py                  # prior check, NUTS fit (dense mass), posterior check
python scripts/fit_model2.py --diagonal-mass  # same fit with NUTS's default mass matrix, for comparison
```

**Prior predictive check** (`reports/model2_prior_predictive.csv`, `.png`).
Before fitting, data were simulated from the priors alone. With Model 1's
generic scales applied to every term, the logit-scale effects add up and push
prevalence toward 0 and 1: the median simulated cell prevalence was 0.50, and
16% of simulated cells were above 0.90, which is not plausible for diabetes.
The chosen priors center `mu` near logit(0.12), a rough prior guess that
adult diabetes prevalence is low but not rare, and narrow the state, education, and sex scales. Age
stays wide because diabetes rises steeply with age. Under the chosen priors,
the simulated cell prevalence has median 0.12 and 90% interval 0.01-0.64, with
0.7% of cells above 0.90. The observed overall sample prevalence (0.146) falls
at the 47th percentile of the prior predictive distribution, so the priors are
centered sensibly without ruling out other values.

**Convergence.** With NUTS's default diagonal mass matrix, the fit failed the
checks (`reports/model2_diagnostics_diagonal_mass.json`): max R-hat 1.0128,
min bulk ESS 295, median 511 leapfrog steps per draw, and 11 draws at the
maximum tree depth. The data determine sums such as `mu + a_age[65+]` very
precisely but not their parts: `mu` and every `a_age` and `a_educ` can shift
together. That leaves a long, narrow, linear ridge in the posterior. A dense
mass matrix lets NUTS learn this correlation during warmup. With it
(`reports/model2_diagnostics.json`): 0 divergences, max R-hat 1.0043, min bulk
ESS 4,120, min tail ESS 2,345, median 15 leapfrog steps, no draws at the depth
limit. `mu` alone stays uncertain (posterior SD 0.55), because the data cannot
separate it from the average age effect. Predicted prevalences depend only on
the sums, so this does not affect them.

| Parameter | Posterior mean | 94% HDI |
|---|---:|---|
| `b_sex` (female - male, log-odds) | -0.224 | -0.243 to -0.206 |
| `sigma_state` | 0.182 | 0.148 to 0.222 |
| `sigma_age` | 1.331 | 0.727 to 2.049 |
| `sigma_educ` | 0.435 | 0.173 to 0.754 |

**Posterior predictive check** (`reports/model2_ppc_groups.csv`, `.png`).
Prevalence in replicated datasets was compared with the observed prevalence
for each group. Observed values fall inside the 90% posterior predictive
interval for all 5 age groups, all 4 education levels, both sexes, and all 48
geographies with data. They do not for combinations the model has no term for:

| Grouping | Groups inside 90% interval |
|---|---:|
| Age x sex | 1 of 10 |
| Age x education | 14 of 20 |

The age x sex misfit is systematic. The model has a single sex effect (women
lower), but in the sample women aged 25-44 have *higher* prevalence than men
(25-34: 0.026 vs 0.021; 35-44: 0.059 vs 0.055), while men aged 45 and older
have higher prevalence, and the gap widens with age (65+: 0.243 vs 0.196).
At the cell level, 1,738 of 1,918 cells (90.6%) fall inside their 90%
interval. Cell-level coverage looks fine because most cells are small, and
the noise in a small cell hides this structure.

## License

Code: MIT (see `LICENSE`). BRFSS data is published by the CDC and is not
included in this repository.
