# Bayesian Multilevel Analysis of Instructional Quality and Mathematics Achievement (PISA 2022, Colombia)

Reproducibility repository for the Bayesian analysis of how instructional quality
relates to mathematics achievement among Colombian 15-year-olds in PISA 2022. It
contains all analysis and plotting code, the processed data, and the generated
posterior draws and figures.

## What the analysis does

A Bayesian two-level varying-intercept model is fitted to the Colombian PISA 2022
sample (7,376 students in 260 schools). Students are nested within schools, and a
school-specific intercept is allowed to vary. Four student-level predictors
(socioeconomic status, sex, grade of enrolment, grade repetition) and four
school-level instructional indices (disciplinary climate, teacher support, and two
cognitive-activation facets) enter the model. Posterior draws are obtained with a
Gibbs sampler run on each of the ten plausible values and pooled, with the school
survey weight applied at the second level following Mang et al. (2021).

## Repository layout

```
pisa-col-bayes/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/            # place official COL_STU_QQQ.SAV and COL_SCHOOL.sav here
│   ├── processed/      # cached CSV extracts (included)
│   └── comparison/     # cross-national descriptive CSVs (included)
├── src/
│   ├── paths.py                          # central path configuration
│   ├── sav_reader.py                     # self-contained SPSS .sav reader
│   ├── data_prep.py                      # builds the analytic dataset
│   ├── 01_extract_data.py                # SPSS -> CSV (needs raw files)
│   ├── 02_fit_bayesian_model.py          # Gibbs sampler, null model, R-hat
│   ├── 03_plot_bayesian_results.py       # forest, ridgeline, ICC figures
│   └── 04_plot_descriptive_comparisons.py# grade and repetition comparison figures
├── results/            # posterior draws and summary (included)
└── figures/            # rendered PNG figures (included)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib. No internet access is required.

## Reproducing the analysis

The repository ships with the cached data extracts and the posterior results, so you
can regenerate the figures immediately, or rerun the full model.

```bash
# (Optional) Step 1 - regenerate the CSV extracts from the raw SPSS files.
#            Requires data/raw/COL_STU_QQQ.SAV and data/raw/COL_SCHOOL.sav.
python src/01_extract_data.py

# Verify the analytic sample (should print 7,376 students in 260 schools).
python src/data_prep.py

# Step 2 - fit the Bayesian model (writes results/). Takes a few minutes.
python src/02_fit_bayesian_model.py

# Step 3 - regenerate all figures (writes figures/).
python src/03_plot_bayesian_results.py
python src/04_plot_descriptive_comparisons.py
```

To rerun only the figures from the included results, skip steps 1 and 2.

## Model details

- Varying-intercept (random-intercept) two-level Gaussian regression.
- Weakly informative priors: diffuse normal on regression coefficients, vague
  inverse-gamma on both variance components.
- Gibbs sampler: 4,000 iterations per plausible value, 1,500 warm-up, thinned;
  post-warm-up draws concatenated across the 10 plausible values into a pooled
  posterior of 12,500 samples (the Bayesian analogue of Rubin's rules).
- School weight `W_SCHGRNRABWT` scales each school's contribution at level 2; the
  student weight is fixed to 1 (Mang et al., 2021).
- Convergence assessed with the Gelman-Rubin statistic across four chains (all
  parameters R-hat <= 1.01).

## Outputs

`results/`
- `bayes_draws.npz` - pooled posterior draws (beta, icc, tau2, sig2, colnames)
- `null_icc_draws.npy` - posterior draws of the unconditional-model ICC
- `bayes_summary.csv` - posterior mean, SD, 95% credible interval, probability of direction

`figures/`
- `fig_bayes_forest.png` - posterior means and 80%/95% credible intervals for all terms
- `fig_bayes_school_posteriors.png` - posterior densities of the four school-level coefficients
- `fig_bayes_icc.png` - null vs. conditional ICC posterior densities
- `fig_grade_distribution.png` - grade distribution, Colombia vs. five high-performing systems
- `fig_grade_repetition.png` - grade repetition, Colombia vs. five high-performing systems

## Key results (pooled posterior)

| Predictor | Posterior mean | 95% CrI |
|---|---:|---|
| Socioeconomic status (ESCS) | 7.35 | [5.39, 8.99] |
| Gender (male) | 19.59 | [16.64, 22.40] |
| Grade 11 (vs. 7) | 64.18 | [55.64, 72.64] |
| Grade repetition (yes) | -10.40 | [-14.21, -6.51] |
| Disciplinary climate (school) | 28.40 | [17.34, 39.00] |
| Teacher support (school) | -18.75 | [-30.26, -7.14] |
| Cognitive activation: reasoning (school) | 36.22 | [23.22, 49.52] |
| Cognitive activation: math thinking (school) | -31.27 | [-47.84, -14.42] |

Unconditional ICC: 0.36 (95% CrI [0.31, 0.41]); conditional ICC approximately 0.23.

## Data source

PISA 2022 database, OECD: https://www.oecd.org/pisa/data/2022database/
The raw microdata are the property of the OECD and are not redistributed here.

## License

Code released under the MIT License (see `LICENSE`). The PISA data are subject to
the OECD's terms of use.

## Authors

Ricardo L. Gómez (rleon.gomez@udea.edu.co) and Walter Castro Gordillo
(wfcastro@udea.edu.co), University of Antioquia, College of Education, Medellín,
Colombia.
