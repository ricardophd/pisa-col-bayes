import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
OUT = str(paths.FIGURES)
ACCENT = "#2C3E50"
POS = "#1E7A46"
NEG = "#C0392B"
NEU = "#9AA4AD"
SCHOOL_C = "#6A4C93"

d = np.load(paths.DRAWS_NPZ, allow_pickle=True)
beta = d["beta"]
colnames = list(d["colnames"])
icc = d["icc"]
null_icc = np.load(paths.NULL_ICC_NPY)

label = {
    "Intercept": "Intercept",
    "ESCS": "ESCS (per SD)",
    "Male": "Male (vs. female)",
    "Grade8": "Grade 8 (vs. 7)",
    "Grade9": "Grade 9 (vs. 7)",
    "Grade10": "Grade 10 (vs. 7)",
    "Grade11": "Grade 11 (vs. 7)",
    "RepeatYes": "Grade repetition (yes)",
    "SCH_DISCLIM": "Disciplinary climate (school)",
    "SCH_TEACHSUP": "Teacher support (school)",
    "SCH_COGACRCO": "Cog. activation: reasoning (school)",
    "SCH_COGACMCO": "Cog. activation: math thinking (school)",
}

# ---------------------------------------------------------------------------
# FIGURE A: Posterior forest plot (mean + 95% CrI), excluding intercept
# ---------------------------------------------------------------------------
order = ["ESCS", "Male", "Grade8", "Grade9", "Grade10", "Grade11", "RepeatYes",
         "SCH_DISCLIM", "SCH_TEACHSUP", "SCH_COGACRCO", "SCH_COGACMCO"]
# display top-to-bottom; reverse so first item is at top
disp = order[::-1]

fig, ax = plt.subplots(figsize=(9, 6.4))
for i, nm in enumerate(disp):
    j = colnames.index(nm)
    x = beta[:, j]
    m = x.mean()
    lo, hi = np.percentile(x, [2.5, 97.5])
    lo80, hi80 = np.percentile(x, [10, 90])
    pdir = max((x > 0).mean(), (x < 0).mean())
    excl = (lo > 0 or hi < 0)
    col = POS if (excl and m > 0) else (NEG if (excl and m < 0) else NEU)
    # 95% thin, 80% thick
    ax.plot([lo, hi], [i, i], color=col, lw=1.6, zorder=2, solid_capstyle="round")
    ax.plot([lo80, hi80], [i, i], color=col, lw=5.0, alpha=0.35, zorder=2,
            solid_capstyle="round")
    ax.scatter([m], [i], s=62, color=col, zorder=3, edgecolor="white", linewidth=0.8)
    ax.text(hi + 1.6, i, f"{m:+.1f}", va="center", ha="left", fontsize=8.6,
            color=col, fontweight="bold" if excl else "normal")

ax.axvline(0, color="#888888", lw=1.0, linestyle="--", zorder=1)
ax.set_yticks(range(len(disp)))
ax.set_yticklabels([label[n] for n in disp])
ax.set_xlabel("Posterior coefficient (change in PISA mathematics score)")
ax.set_xlim(beta[:, [colnames.index(n) for n in order]].min() - 12,
            beta[:, [colnames.index(n) for n in order]].max() + 18)
ax.xaxis.grid(True, color="#ECECEC", linewidth=0.7)
ax.set_axisbelow(True)
# separator between student and school blocks (school = first 4 of disp)
ax.axhline(3.5, color="#DDDDDD", lw=0.8)
ax.text(ax.get_xlim()[0] + 1.5, 9.5, "Student level", fontsize=8.6, color="#777777", style="italic")
ax.text(ax.get_xlim()[0] + 1.5, 2.6, "School level", fontsize=8.6, color="#777777", style="italic")
legend = [Patch(facecolor=POS, label="95% CrI excludes 0 (positive)"),
          Patch(facecolor=NEG, label="95% CrI excludes 0 (negative)"),
          Patch(facecolor=NEU, label="95% CrI includes 0")]
ax.legend(handles=legend, frameon=False, fontsize=8.2, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_bayes_forest.png", bbox_inches="tight")
plt.close()
print("saved forest")

# ---------------------------------------------------------------------------
# FIGURE B: Posterior densities for the 4 school-level instructional predictors
# ---------------------------------------------------------------------------
school_terms = ["SCH_DISCLIM", "SCH_COGACRCO", "SCH_TEACHSUP", "SCH_COGACMCO"]
fig, ax = plt.subplots(figsize=(9, 6.2))
ridge_gap = 1.5
for k, nm in enumerate(school_terms):
    j = colnames.index(nm)
    x = beta[:, j]
    m = x.mean()
    lo, hi = np.percentile(x, [2.5, 97.5])
    excl = (lo > 0 or hi < 0)
    col = POS if (excl and m > 0) else (NEG if (excl and m < 0) else NEU)
    kde = stats.gaussian_kde(x)
    xs = np.linspace(x.min() - 5, x.max() + 5, 400)
    ys = kde(xs)
    ys = ys / ys.max() * 0.95
    base = (len(school_terms) - 1 - k) * ridge_gap
    ax.fill_between(xs, base, base + ys, color=col, alpha=0.45, zorder=2)
    ax.plot(xs, base + ys, color=col, lw=1.3, zorder=3)
    # 95% CrI bar just below the ridge baseline
    ax.plot([lo, hi], [base - 0.12, base - 0.12], color=col, lw=2.2, zorder=4)
    ax.scatter([m], [base - 0.12], s=30, color=col, zorder=5, edgecolor="white", linewidth=0.6)
    # label name to the left of the baseline
    ax.text(xs.min() - 2, base + 0.12, label[nm], ha="right", va="center", fontsize=9.4,
            color=ACCENT)
    # mean value at the density peak
    ax.text(m, base + ys.max() + 0.05, f"{m:+.1f}", ha="center", va="bottom",
            fontsize=9.0, color=col, fontweight="bold")

ax.axvline(0, color="#888888", lw=1.0, linestyle="--", zorder=1)
ax.set_yticks([])
ax.set_ylim(-0.4, (len(school_terms) - 1) * ridge_gap + 1.4)
ax.set_xlabel("Posterior coefficient (change in PISA mathematics score)")
ax.set_xlim(-66, 70)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, color="#ECECEC", linewidth=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_bayes_school_posteriors.png", bbox_inches="tight")
plt.close()
print("saved school posteriors")

# ---------------------------------------------------------------------------
# FIGURE C: ICC posterior densities (null vs conditional)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for arr, c, lab in [(null_icc, "#34699A", "Null (unconditional) model"),
                    (icc, "#B5651D", "Full (conditional) model")]:
    kde = stats.gaussian_kde(arr)
    xs = np.linspace(0.12, 0.46, 400)
    ys = kde(xs)
    ax.fill_between(xs, 0, ys, color=c, alpha=0.30)
    ax.plot(xs, ys, color=c, lw=1.8, label=lab)
    m = arr.mean()
    ax.axvline(m, color=c, lw=1.1, linestyle="--")
    ypk = float(kde(m)[0])
    ax.text(m, ypk * 1.02, f"{m:.2f}",
            ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")

ax.set_xlabel("Intraclass correlation (between-school variance share)")
ax.set_ylabel("Posterior density")
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_bayes_icc.png", bbox_inches="tight")
plt.close()
print("saved icc")
print("ALL DONE")
