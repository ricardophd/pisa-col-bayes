# Raw data

Place the official PISA 2022 Colombia SPSS files here:

- `COL_STU_QQQ.SAV`  (student questionnaire file)
- `COL_SCHOOL.sav`   (school questionnaire file)

These files are distributed by the OECD and are not redistributed in this
repository (see `.gitignore`). They can be obtained from the PISA 2022 database:
https://www.oecd.org/pisa/data/2022database/

Once the files are in place, run `python src/01_extract_data.py` to regenerate the
cached CSV extracts in `data/processed/`. Those cached extracts are already included
in the repository, so the modeling and plotting steps run without the raw files.
