"""Bayesian two-level varying-intercept model for PISA 2022 Colombia mathematics.

Model (for each plausible value pv):
    y_ij = x_ij' beta + u_j + e_ij
    u_j  ~ N(0, tau2)            (school random intercept)
    e_ij ~ N(0, sigma2)         (student residual)

Estimation: Gibbs sampler with conjugate priors.
    beta   ~ N(0, 1e6 * I)        (weakly informative)
    sigma2 ~ Inverse-Gamma(a0, b0), a0=b0=0.001
    tau2   ~ Inverse-Gamma(a0, b0)

Weights follow Mang et al. (2021): level-1 weight = 1 for all students; the
school weight (W_SCHGRNRABWT) enters at level 2. We implement this by weighting
each school's contribution to the random-intercept and tau2 updates by its
(normalized) school weight, leaving the student level unweighted (w=1).

Plausible values are handled in the fully Bayesian analogue of Rubin's rules:
we run the sampler on each PV and concatenate the post-burn-in posterior draws
across the 10 PVs, so the pooled posterior reflects both sampling and imputation
uncertainty.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
from data_prep import build_analysis_data, PV, SCH_QUAL
import paths

RNG = np.random.default_rng(20240605)

PREDICTORS = ["ESCS", "Male", "Grade8", "Grade9", "Grade10", "Grade11",
              "RepeatYes"] + SCH_QUAL
PRETTY = {
    "Intercept": "Intercept",
    "ESCS": "ESCS",
    "Male": "Gender (Male)",
    "Grade8": "Grade: 8",
    "Grade9": "Grade: 9",
    "Grade10": "Grade: 10",
    "Grade11": "Grade: 11",
    "RepeatYes": "Grade Repetition (Yes)",
    "SCH_DISCLIM": "Disciplinary Climate (DISCLIM)",
    "SCH_TEACHSUP": "Teacher Support (TEACHSUP)",
    "SCH_COGACRCO": "Cognitive Activation: Foster Reasoning (COGACRCO)",
    "SCH_COGACMCO": "Cognitive Activation: Encourage Mathematical Thinking (COGACMCO)",
}


def gibbs_one_pv(y, X, school_idx, sch_w, n_iter=4000, burn=1500, thin=2, seed=1):
    """Run Gibbs sampler for one plausible value. Returns posterior draws."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    J = school_idx.max() + 1

    # school weights normalized to sum to J (so prior strength is comparable)
    w = sch_w / sch_w.mean()

    XtX = X.T @ X
    # priors
    b0_prec = 1e-6                      # beta prior precision
    a0 = b0 = 1e-3
    # init
    beta = np.linalg.solve(XtX + b0_prec * np.eye(p), X.T @ y)
    u = np.zeros(J)
    sigma2 = np.var(y - X @ beta)
    tau2 = sigma2 * 0.5

    # precompute per-school student lists
    members = [np.where(school_idx == j)[0] for j in range(J)]
    nj = np.array([len(m) for m in members])

    draws_beta = []
    draws_sigma2 = []
    draws_tau2 = []
    draws_icc = []

    for it in range(n_iter):
        # --- beta | rest : Bayesian weighted LS with offset u ---
        resid = y - u[school_idx]
        prec = XtX / sigma2 + b0_prec * np.eye(p)
        mean_rhs = (X.T @ resid) / sigma2
        L = np.linalg.cholesky(prec)
        mu = np.linalg.solve(prec, mean_rhs)
        z = rng.standard_normal(p)
        beta = mu + np.linalg.solve(L.T, z)

        # --- u_j | rest : per-school, school-weighted shrinkage (vectorized) ---
        r = y - X @ beta
        sj = np.bincount(school_idx, weights=r, minlength=J)
        prec_j = nj / sigma2 + w / tau2
        mean_j = (sj / sigma2) / prec_j
        u = mean_j + rng.standard_normal(J) / np.sqrt(prec_j)

        # --- sigma2 | rest (student level, unweighted, w=1) ---
        e = y - X @ beta - u[school_idx]
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * np.sum(e ** 2)
        sigma2 = 1.0 / rng.gamma(a_n, 1.0 / b_n)

        # --- tau2 | rest (school level, school-weighted) ---
        a_t = a0 + 0.5 * np.sum(w > 0)
        b_t = b0 + 0.5 * np.sum(w * u ** 2)
        tau2 = 1.0 / rng.gamma(a_t, 1.0 / b_t)

        if it >= burn and (it - burn) % thin == 0:
            draws_beta.append(beta.copy())
            draws_sigma2.append(sigma2)
            draws_tau2.append(tau2)
            draws_icc.append(tau2 / (tau2 + sigma2))

    return (np.array(draws_beta), np.array(draws_sigma2),
            np.array(draws_tau2), np.array(draws_icc))


def run_full_model():
    df, stu, sch = build_analysis_data()
    # design matrix (centering not applied to match original raw-coefficient scale)
    Xcols = PREDICTORS
    X = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) for c in Xcols])
    colnames = ["Intercept"] + Xcols

    codes, uniques = pd.factorize(df["CNTSCHID"].values)
    school_idx = codes.astype(int)
    # school weight per school (one value per school)
    sch_w = (df.groupby("CNTSCHID")["W_SCHGRNRABWT"].first()
             .reindex(uniques).values.astype(float))

    all_beta, all_icc, all_tau2, all_sig2 = [], [], [], []
    for k, pv in enumerate(PV):
        y = df[pv].values.astype(float)
        b, s2, t2, icc = gibbs_one_pv(y, X, school_idx, sch_w,
                                      n_iter=4000, burn=1500, thin=2, seed=100 + k)
        all_beta.append(b)
        all_icc.append(icc)
        all_tau2.append(t2)
        all_sig2.append(s2)
        print(f"  PV{k+1}: beta[ESCS]={b[:,1].mean():.3f} icc={icc.mean():.3f}")

    # pool draws across PVs (Bayesian analogue of Rubin's rules)
    beta_draws = np.vstack(all_beta)          # (total_draws, p)
    icc_draws = np.concatenate(all_icc)
    tau2_draws = np.concatenate(all_tau2)
    sig2_draws = np.concatenate(all_sig2)

    return df, colnames, beta_draws, icc_draws, tau2_draws, sig2_draws


def summarize(colnames, beta_draws, icc_draws):
    rows = []
    for i, name in enumerate(colnames):
        d = beta_draws[:, i]
        mean = d.mean()
        sd = d.std(ddof=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        # posterior prob of direction
        p_pos = (d > 0).mean()
        pd_ = max(p_pos, 1 - p_pos)            # probability of direction
        rows.append(dict(term=name, predictor=PRETTY.get(name, name),
                         estimate=mean, sd=sd, ci_lower=lo, ci_upper=hi,
                         p_direction=pd_,
                         excludes_zero=(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def run_null_model(seed_base=500, n_iter=3000, burn=1000, thin=2):
    """Intercept-only model to estimate the unconditional ICC."""
    df, stu, sch = build_analysis_data()
    codes, uniques = pd.factorize(df["CNTSCHID"].values)
    school_idx = codes.astype(int)
    sch_w = (df.groupby("CNTSCHID")["W_SCHGRNRABWT"].first()
             .reindex(uniques).values.astype(float))
    iccs = []
    for k, pv in enumerate(PV):
        y = df[pv].values.astype(float)
        X = np.ones((len(df), 1))
        _, _, _, icc = gibbs_one_pv(y, X, school_idx, sch_w,
                                    n_iter=n_iter, burn=burn, thin=thin, seed=seed_base + k)
        iccs.append(icc)
    return np.concatenate(iccs)


def rhat_check(n_chains=4, n_iter=4000, burn=1500, thin=2, seed_base=900):
    """Gelman-Rubin potential scale reduction on PV1, across chains."""
    df, stu, sch = build_analysis_data()
    codes, uniques = pd.factorize(df["CNTSCHID"].values)
    school_idx = codes.astype(int)
    sch_w = (df.groupby("CNTSCHID")["W_SCHGRNRABWT"].first()
             .reindex(uniques).values.astype(float))
    X = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) for c in PREDICTORS])
    y = df["PV1MATH"].values.astype(float)
    chains = []
    for s in range(n_chains):
        b, _, _, _ = gibbs_one_pv(y, X, school_idx, sch_w,
                                  n_iter=n_iter, burn=burn, thin=thin, seed=seed_base + s)
        chains.append(b)
    chains = np.array(chains)
    names = ["Intercept"] + PREDICTORS

    def rhat(x):
        m, n = x.shape
        B = n * x.mean(1).var(ddof=1)
        W = x.var(1, ddof=1).mean()
        var = (n - 1) / n * W + B / n
        return float(np.sqrt(var / W))

    return {nm: rhat(chains[:, :, i]) for i, nm in enumerate(names)}


if __name__ == "__main__":
    print("Fitting Bayesian multilevel model across plausible values...")
    df, colnames, beta_draws, icc_draws, tau2_draws, sig2_draws = run_full_model()
    summ = summarize(colnames, beta_draws, icc_draws)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(summ.to_string(index=False))
    print(f"\nConditional ICC: mean={icc_draws.mean():.4f} "
          f"95% CrI [{np.percentile(icc_draws,2.5):.4f}, {np.percentile(icc_draws,97.5):.4f}]")
    np.savez(paths.DRAWS_NPZ,
             beta=beta_draws, icc=icc_draws, tau2=tau2_draws, sig2=sig2_draws,
             colnames=np.array(colnames, dtype=object))
    summ.to_csv(paths.SUMMARY_CSV, index=False)

    print("\nFitting null (intercept-only) model for the unconditional ICC...")
    null_icc = run_null_model()
    np.save(paths.NULL_ICC_NPY, null_icc)
    print(f"Null ICC: mean={null_icc.mean():.4f} "
          f"95% CrI [{np.percentile(null_icc,2.5):.4f}, {np.percentile(null_icc,97.5):.4f}]")

    print("\nConvergence check (Gelman-Rubin R-hat, 4 chains on PV1):")
    for nm, r in rhat_check().items():
        print(f"  {nm:14s} {r:.4f}")
    print("\nAll results saved to the results/ folder.")

