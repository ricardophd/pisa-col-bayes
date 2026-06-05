"""data_prep.py
Build the analytic dataset: aggregate student-reported instructional indices to
school means, dummy-code the categorical predictors, and apply complete-case
selection. Importable by the model and figure scripts; also runnable directly to
print a sample summary that should reproduce 7,376 students in 260 schools.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import paths

PV = [f"PV{i}MATH" for i in range(1, 11)]
QUAL = ["DISCLIM", "TEACHSUP", "COGACRCO", "COGACMCO"]
SCH_QUAL = [f"SCH_{q}" for q in QUAL]


def build_analysis_data():
    stu = pd.read_csv(paths.STUDENT_CSV)
    sch = pd.read_csv(paths.SCHOOL_CSV)

    # school means of student-reported instructional quality
    sm = stu.groupby("CNTSCHID")[QUAL].mean()
    sm.columns = SCH_QUAL
    sm = sm.reset_index()

    df = stu.drop(columns=QUAL).merge(sm, on="CNTSCHID", how="left")
    df = df.merge(sch[["CNTSCHID", "W_SCHGRNRABWT"]], on="CNTSCHID", how="left")

    # Gender: 1 = Female (reference), 2 = Male
    df["Male"] = np.where(df["ST004D01T"] == 2, 1.0,
                          np.where(df["ST004D01T"] == 1, 0.0, np.nan))
    # Grade: reference = 7
    for g in [8, 9, 10, 11]:
        df[f"Grade{g}"] = np.where(df["ST001D01T"] == g, 1.0,
                                   np.where(df["ST001D01T"].isin([7, 8, 9, 10, 11]), 0.0, np.nan))
    # Repetition: 0 = No (reference), 1 = Yes
    df["RepeatYes"] = np.where(df["REPEAT"] == 1, 1.0,
                               np.where(df["REPEAT"] == 0, 0.0, np.nan))

    model_cols = (PV + ["ESCS", "Male", "Grade8", "Grade9", "Grade10", "Grade11",
                        "RepeatYes"] + SCH_QUAL + ["CNTSCHID", "W_SCHGRNRABWT"])
    df = df[df["ST001D01T"].isin([7, 8, 9, 10, 11])]
    df = df.dropna(subset=model_cols)
    return df, stu, sch


if __name__ == "__main__":
    df, stu, sch = build_analysis_data()
    print("Analytic sample:", len(df), "students in", df["CNTSCHID"].nunique(), "schools")
    print("Raw:", len(stu), "students in", stu["CNTSCHID"].nunique(), "schools")
    print("Grade distribution (%):")
    print((df["ST001D01T"].value_counts(normalize=True).sort_index() * 100).round(1))
    print("Repetition %:", round(df["RepeatYes"].mean() * 100, 1))
    print("Male %:", round(df["Male"].mean() * 100, 1))
    print("ESCS mean/sd:", round(df["ESCS"].mean(), 3), round(df["ESCS"].std(), 3))
