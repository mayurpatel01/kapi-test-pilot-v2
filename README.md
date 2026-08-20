# Voluntary Benefits Intelligence (Form 5500)

Streamlit dashboard and Excel export built on DOL Form 5500 + Schedule A. Scope
is group benefits and the voluntary line — AON does not sell medical, dental or
vision, so those are excluded from the pipeline entirely.

**The data covers plan year 2024, filed during 2025.** The DOL release is named
"2024 Latest" and the two years are easy to confuse: 96.2% of filings have a
plan year beginning in 2024, while 97.9% were *received* by DOL during calendar
2025 (the rest by January 2026). So this is the most recent complete year of
filings — not a 2025 view of the market. "Latest" means the most recent version
of each filing, so amendments supersede originals.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the committed parquet marts in `data/marts/`, so it runs straight
from a clone. Streamlit Cloud deploys from `main`.

## Regenerating the data

The raw DOL files are **not** in the repo — they are ~74 MB of zips that nothing
reads at runtime, and only the ETL needs them. Download them from the DOL Form
5500 dataset page (<https://www.dol.gov/agencies/ebsa/about-ebsa/our-activities/public-disclosure/foia/form-5500-datasets>)
into `data/raw/`:

| File | Used for |
|---|---|
| `F_5500_2024_Latest.zip` | employer names and sponsor identifiers |
| `F_SCH_A_2024_Latest.zip` | products, covered lives, carriers, premium |
| `F_SCH_A_PART1_2024_Latest.zip` | broker names and commissions |

Then rebuild:

```bash
python etl/build_marts.py \
  --zip_a data/raw/F_5500_2024_Latest.zip \
  --zip_b data/raw/F_SCH_A_2024_Latest.zip \
  --zip_c data/raw/F_SCH_A_PART1_2024_Latest.zip \
  --out_dir data/marts
```

Marts in `data/marts/` **are** committed on purpose: Streamlit Cloud builds from
the repo and never runs the ETL, so an untracked mart does not exist in the
deployed app. Commit them after rebuilding.

## Excel export

```bash
python scripts/export_dataset.py --out exports/pilot_dataset.xlsx
python scripts/export_dataset.py --out exports/pilot_dataset_with_detail.xlsx --with-detail
```

Caps for filer keying errors are flags, not hardcoded: `--comm-cap`,
`--lives-cap`, `--premium-cap`. Every excluded row is listed on the workbook's
`Data_Quality_Flags` sheet rather than dropped silently.

## Layout

| Path | What it is |
|---|---|
| `app.py` | Streamlit dashboard |
| `etl/build_marts.py` | Raw DOL zips → parquet marts |
| `etl/benefits.py` | Product classification, incl. voluntary parsing from free text |
| `etl/brokers.py` | Broker name matching and tiering — shared by app and export |
| `scripts/export_dataset.py` | Excel workbook builder |
| `data/marts/*.parquet` | Committed marts the app reads |

## Two things that will bite you

**Voluntary products have no Schedule A checkbox.** Critical illness, accident
and hospital indemnity are filed under the OTHER checkbox as free text, which is
why OTHER is the most-used box on the form. `etl/benefits.py` parses that text.
AD&D is deliberately kept out of "accident" — it appears in ~46k filings, nearly
double the entire real voluntary universe, and folding it in roughly doubles
apparent market size.

**Premium must be summed over contracts, not products.** One Schedule A contract
can cover life + STD + LTD, appearing three times in the product-exploded table
while reporting a single premium. `data/marts/employer_contract.parquet` holds
one row per contract for exactly this reason; summing the product table instead
inflates premium ~2.5x. The same applies to covered lives, which is why those use
MAX rather than SUM.
