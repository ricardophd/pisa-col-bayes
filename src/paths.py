"""Central path configuration so every script resolves files relative to the
repository root, regardless of where it is run from."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_COMPARISON = ROOT / "data" / "comparison"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Raw SPSS microdata (PISA 2022, Colombia). Place the official files here.
STUDENT_SAV = DATA_RAW / "COL_STU_QQQ.SAV"
SCHOOL_SAV = DATA_RAW / "COL_SCHOOL.sav"

# Cached CSV extracts (created by 01_extract_data.py)
STUDENT_CSV = DATA_PROCESSED / "stu_raw.csv"
SCHOOL_CSV = DATA_PROCESSED / "sch_raw.csv"

# Posterior draws and summaries (created by 02_fit_bayesian_model.py)
DRAWS_NPZ = RESULTS / "bayes_draws.npz"
NULL_ICC_NPY = RESULTS / "null_icc_draws.npy"
SUMMARY_CSV = RESULTS / "bayes_summary.csv"

for d in (DATA_RAW, DATA_PROCESSED, DATA_COMPARISON, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)
