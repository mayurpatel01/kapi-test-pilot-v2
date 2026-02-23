import re
from pathlib import Path
import pandas as pd

DATA_MARTS = Path("data/marts")
RAW_FILE = Path("data/raw/form5500/f_5500_2024_latest.csv")

COLS = {
    "employer_name": "SPONSOR_DFE_NAME",
    "state": "SPONS_DFE_MAIL_US_STATE",
    "zip": "SPONS_DFE_MAIL_US_ZIP",
    "city": "SPONS_DFE_MAIL_US_CITY",
    "ein": "SPONS_DFE_EIN",
}

def norm_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    suffixes = r"\b(INC|INCORPORATED|LLC|L L C|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|HOLDINGS|GROUP)\b"
    s = re.sub(suffixes, "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mode_value(s):
    s = s.dropna()
    s = s[s.astype(str).str.strip() != ""]
    if len(s) == 0:
        return ""
    return s.value_counts().index[0]

def main():

    print("Loading employer universe from existing mart...")
    gaps = pd.read_parquet(DATA_MARTS / "employer_product_matrix.parquet")

    employer_universe = gaps[["Employer"]].drop_duplicates()
    employer_universe["employer_key"] = employer_universe["Employer"].apply(norm_name)

    print("Reading Form 5500 main file...")
    df = pd.read_csv(
        RAW_FILE,
        sep=",",
        dtype=str,
        usecols=list(COLS.values()),
        low_memory=False
    )

    df = df.rename(columns={v: k for k, v in COLS.items()})
    df["employer_key"] = df["employer_name"].apply(norm_name)

    df["state"] = df["state"].fillna("").str.strip().str.upper()
    df["zip"] = df["zip"].fillna("").str.replace(r"\D", "", regex=True).str.zfill(5).str[:5]
    df["city"] = df["city"].fillna("").str.strip()
    df["ein"] = df["ein"].fillna("").str.replace(r"\D", "", regex=True).str.zfill(9).str[-9:]

    grouped = df.groupby("employer_key", as_index=False).agg({
        "state": mode_value,
        "zip": mode_value,
        "city": mode_value,
        "ein": mode_value
    })

    geo = employer_universe.merge(grouped, on="employer_key", how="left")

    out = geo[["Employer", "state", "zip", "city", "ein"]].rename(columns={
        "state": "State",
        "zip": "ZIP",
        "city": "City",
        "ein": "EIN"
    })

    output_path = DATA_MARTS / "employer_geo.parquet"
    out.to_parquet(output_path, index=False)

    print("\nSUCCESS.")
    print("File written to:", output_path)
    print("Total employers:", len(out))
    print("State coverage %:", (out["State"] != "").mean())
    print("ZIP coverage %:", (out["ZIP"] != "").mean())

if __name__ == "__main__":
    main()