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
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    suffixes = r"\b(INC|INCORPORATED|LLC|L L C|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|HOLDINGS|GROUP)\b"
    s = re.sub(suffixes, "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_zip(z):
    if z is None:
        return ""
    z = re.sub(r"\D", "", str(z))
    z = (z + "00000")[:5]  # pad then cut
    return z

def clean_ein(e):
    if e is None:
        return ""
    e = re.sub(r"\D", "", str(e))
    e = ("000000000" + e)[-9:]
    return e

def main():
    print("Loading employer universe from existing mart...")
    gaps = pd.read_parquet(DATA_MARTS / "employer_product_matrix.parquet")
    employers = gaps[["Employer"]].drop_duplicates()
    employers["employer_key"] = employers["Employer"].apply(norm_name)

    employer_keys = set(employers["employer_key"].tolist())
    print(f"Employer universe: {len(employers):,}")

    usecols = list(COLS.values())
    rename_map = {v: k for k, v in COLS.items()}

    # We'll store best guess per employer_key as we stream
    # Using "first non-empty" (fast) — later we can switch to "mode" if needed
    best = {}

    print("Streaming Form 5500 main file in chunks...")
    chunk_size = 250_000
    matched_rows = 0
    chunk_idx = 0

    reader = pd.read_csv(
        RAW_FILE,
        sep=",",
        dtype=str,
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        chunk_idx += 1
        if chunk_idx % 1 == 0:
            print(f"  - chunk {chunk_idx:,} ...")

        chunk = chunk.rename(columns=rename_map)
        chunk["employer_key"] = chunk["employer_name"].apply(norm_name)

        # Keep only employers that exist in your marts (massive speed-up)
        chunk = chunk[chunk["employer_key"].isin(employer_keys)]
        if chunk.empty:
            continue

        matched_rows += len(chunk)

        # Clean geo fields
        chunk["state"] = chunk["state"].fillna("").astype(str).str.strip().str.upper()
        chunk["zip"] = chunk["zip"].apply(clean_zip)
        chunk["city"] = chunk["city"].fillna("").astype(str).str.strip()
        chunk["ein"] = chunk["ein"].apply(clean_ein)

        # Fill best dict (first non-empty values)
        for r in chunk.itertuples(index=False):
            k = r.employer_key
            if k not in best:
                best[k] = {"state": "", "zip": "", "city": "", "ein": ""}
            if not best[k]["state"] and r.state:
                best[k]["state"] = r.state
            if not best[k]["zip"] and r.zip and r.zip != "00000":
                best[k]["zip"] = r.zip
            if not best[k]["city"] and r.city:
                best[k]["city"] = r.city
            if not best[k]["ein"] and r.ein and r.ein != "000000000":
                best[k]["ein"] = r.ein

        # Optional early stop if we’ve resolved almost all employers
        if len(best) >= int(len(employers) * 0.98):
            print("  - Reached 98% employer coverage; stopping early.")
            break

    print(f"Chunks processed: {chunk_idx:,}")
    print(f"Matched sponsor rows: {matched_rows:,}")
    print(f"Employers with geo found: {len(best):,}")

    geo_rows = []
    for k, v in best.items():
        geo_rows.append((k, v["state"], v["zip"], v["city"], v["ein"]))

    geo = pd.DataFrame(geo_rows, columns=["employer_key","State","ZIP","City","EIN"])
    out = employers.merge(geo, on="employer_key", how="left")[["Employer","State","ZIP","City","EIN"]]

    out_path = DATA_MARTS / "employer_geo.parquet"
    out.to_parquet(out_path, index=False)

    print("\nSUCCESS.")
    print("Wrote:", out_path)
    print("State coverage %:", (out["State"].fillna("") != "").mean())
    print("ZIP coverage %:", (out["ZIP"].fillna("") != "").mean())

if __name__ == "__main__":
    main()