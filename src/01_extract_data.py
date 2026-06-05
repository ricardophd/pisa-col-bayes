"""01_extract_data.py
Read the raw PISA 2022 Colombia SPSS files and cache the modeled variables as CSV.

Run:
    python src/01_extract_data.py

Requires the official SPSS files in data/raw/:
    COL_STU_QQQ.SAV, COL_SCHOOL.sav
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sav_reader import read_sav
import paths

NEED_STU = (["CNT", "CNTRYID", "CNTSCHID", "CNTSTUID", "ESCS", "ST004D01T",
             "ST001D01T", "REPEAT", "DISCLIM", "TEACHSUP", "COGACRCO", "COGACMCO",
             "W_FSTUWT"] + [f"PV{i}MATH" for i in range(1, 11)])
NEED_SCH = ["CNT", "CNTRYID", "CNTSCHID", "W_SCHGRNRABWT"]


def main():
    if not paths.STUDENT_SAV.exists() or not paths.SCHOOL_SAV.exists():
        raise SystemExit(
            f"Raw SPSS files not found in {paths.DATA_RAW}. "
            "Place COL_STU_QQQ.SAV and COL_SCHOOL.sav there first.")

    print("Reading student file (this takes ~15-20 s)...")
    stu = read_sav(str(paths.STUDENT_SAV), usecols=NEED_STU)
    stu.to_csv(paths.STUDENT_CSV, index=False)
    print(f"  wrote {paths.STUDENT_CSV}  ({stu.shape[0]} rows)")

    print("Reading school file...")
    sch = read_sav(str(paths.SCHOOL_SAV), usecols=NEED_SCH)
    sch.to_csv(paths.SCHOOL_CSV, index=False)
    print(f"  wrote {paths.SCHOOL_CSV}  ({sch.shape[0]} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
