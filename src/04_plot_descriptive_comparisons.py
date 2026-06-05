import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
UP = str(paths.DATA_COMPARISON)
OUT = str(paths.FIGURES)

country_names = {
    "COL": "Colombia", "SGP": "Singapore", "JPN": "Japan",
    "MAC": "Macao", "HKG": "Hong Kong", "TAP": "Chinese Taipei",
}
# Order: Colombia first, then top performers
order = ["COL", "SGP", "JPN", "MAC", "HKG", "TAP"]

COL_HL = "#C0392B"   # Colombia highlight (red)
OTHER  = "#5B8DB8"   # comparison countries (blue)
ACCENT = "#2C3E50"

# ----------------------------------------------------------------------------
# FIGURE: Grade-level distribution (grouped/stacked) -> use grouped bars by grade
# ----------------------------------------------------------------------------
grade = pd.read_csv(f"{UP}/student_grade_col_vs_top5_percent.csv", index_col=0)
grade.columns = [str(c) for c in grade.columns]
# Drop grade 12 (all ~0) and grade columns that are entirely zero
grade = grade[[c for c in grade.columns if grade[c].sum() > 0.05]]
grades = list(grade.columns)

fig, ax = plt.subplots(figsize=(9, 5.2))
n_countries = len(order)
n_grades = len(grades)
bar_w = 0.8 / n_countries
x = np.arange(n_grades)

for i, ctry in enumerate(order):
    vals = grade.loc[ctry, grades].values.astype(float)
    color = COL_HL if ctry == "COL" else OTHER
    alpha = 1.0 if ctry == "COL" else 0.55 + 0.07*i
    offset = (i - (n_countries-1)/2) * bar_w
    bars = ax.bar(x + offset, vals, bar_w, label=country_names[ctry],
                  color=color, alpha=alpha, edgecolor="white", linewidth=0.4)
    # Label only sizeable bars to avoid clutter
    for b, v in zip(bars, vals):
        if v >= 8:
            ax.text(b.get_x()+b.get_width()/2, v+1.4, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=7.2,
                    color=ACCENT, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([f"Grade {g}" for g in grades])
ax.set_ylabel("Percentage of 15-year-old students")
ax.set_ylim(0, 108)
ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.98))
ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_grade_distribution.png", bbox_inches="tight")
plt.close()
print("saved grade distribution")

# ----------------------------------------------------------------------------
# FIGURE: Grade repetition (single bar per country) with Japan annotated as n/a
# ----------------------------------------------------------------------------
rep = pd.read_csv(f"{UP}/student_repeat_col_vs_top5_percent.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8.4, 5.0))
rep_order = order
yvals = []
labels = []
for ctry in rep_order:
    yvals.append(rep.loc[ctry, "Yes"])
    labels.append(country_names[ctry])

xpos = np.arange(len(rep_order))
colors = [COL_HL if c == "COL" else OTHER for c in rep_order]
bars = ax.bar(xpos, yvals, 0.62, color=colors, edgecolor="white", linewidth=0.5)

for b, ctry in zip(bars, rep_order):
    v = b.get_height()
    if ctry == "JPN":
        ax.text(b.get_x()+b.get_width()/2, 1.0, "n/a*", ha="center", va="bottom",
                fontsize=9, color="#888888", style="italic")
    else:
        ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=9.5, color=ACCENT, fontweight="bold")

ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("Students reporting grade repetition (%)")
ax.set_ylim(0, max(yvals)+8)
ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.7)
ax.set_axisbelow(True)
ax.annotate("*Japan does not report grade repetition (national policy);\nvalues are missing for 100% of cases.",
            xy=(0.0, -0.16), xycoords="axes fraction", fontsize=8, color="#777777",
            ha="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_grade_repetition.png", bbox_inches="tight")
plt.close()
print("saved repetition")

print("Descriptive comparison figures saved to figures/.")
