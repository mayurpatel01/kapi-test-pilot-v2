import glob, os
import pandas as pd

files = glob.glob("data/raw/form5500/*.txt") + glob.glob("data/raw/form5500/*.csv")
files = sorted(files, key=os.path.getsize, reverse=True)

print("raw files:", files[:3])
src = files[0]
print("using:", src)

df = pd.read_csv(src, sep="|", nrows=2000, dtype=str, low_memory=False)
print("num cols:", len(df.columns))

geo_like = [c for c in df.columns if any(k in c.upper() for k in [
    "SPONS", "SPONSOR", "NAME", "EIN", "STATE", "ZIP", "CITY", "ADDR", "ADDRESS"
])]
print("geo-like columns:")
for c in geo_like:
    print(" -", c)