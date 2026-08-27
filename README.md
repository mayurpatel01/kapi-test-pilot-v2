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

Put each year's three zips in `data/raw/<year>/`, then build every year at once:

```bash
python etl/build_all_years.py                  # every year found under data/raw
python etl/build_all_years.py --years 2023 2024
python etl/build_all_years.py --skip-build     # just refresh trend_summary
```

That writes `data/marts/<plan_year>/` per year plus a small cross-year
`trend_summary.parquet`. The app loads only the selected year, so memory stays
bounded as years accumulate. A single year can still be built directly:

```bash
python etl/build_marts.py --plan-year 2024 \
  --zip_a data/raw/2024/F_5500_2024_Latest.zip \
  --zip_b data/raw/2024/F_SCH_A_2024_Latest.zip \
  --zip_c data/raw/2024/F_SCH_A_PART1_2024_Latest.zip \
  --out_dir data/marts/2024
```

`--plan-year` matters: a release is only ~96% the year it is named for, and the
late/amended tail also appears in its own release, so loading several years
unfiltered double-counts.

Marts in `data/marts/` **are** committed on purpose: Streamlit Cloud builds from
the repo and never runs the ETL, so an untracked mart does not exist in the
deployed app. Commit them after rebuilding.

**Re-download before rebuilding.** DOL keeps adding late and amended filings to
closed years. A stale 2024 snapshot held 193,986 filings against 216,769 in the
current release — 10.5% missing, and it understated commissions by 12.8%.

## Multi-year: what is available, and when

DOL names each release for the **plan year**, not the filing year, and filings
arrive over roughly two years. Verified across three releases:

| Plan year | Filings held | State | Received between |
|---|---:|---|---|
| 2023 | 223,028 | complete | 2024-01-01 .. 2026-07-25 |
| 2024 | 193,986 | ~87% | 2025-01-01 .. 2026-01-24 |
| 2025 | 57,161 | **~26%** | 2026-01-02 .. 2026-07-25 |

**Plan year 2025 is about a quarter of a year and must not be trended against a
complete year.** Only 35.6% of a year files by the on-time deadline; the
mid-October extension deadline is where ~37% of the year lands in a single
month. Worse, the missing filings are not a random sample — at this point in
the 2024 cycle the data held 40% of filings but only **34% of covered lives**,
because large complex plans take extensions (mean group size 1,062 lives for
early filers against 3,090 for late ones). Comparing a partial year to a
complete one shows a fake collapse concentrated in exactly the large accounts
that matter most.

Plan year 2025 becomes usable around **November 2026**, complete by roughly
February 2027. For trend analysis now, go backwards: 2021–2023 are complete.

If an early 2025 read is needed sooner, compare **cohorts at the same point in
the cycle** — 2024 filings received by 21 Aug 2025 against 2025 filings received
by 21 Aug 2026. Both carry the same bias, so the comparison is valid, but those
figures must never be mixed with full-year numbers.

Note also that DOL keeps adding late and amended filings to *closed* years, so
re-running the ETL on a fresh download will not reproduce existing marts
exactly. The 2023 file was still receiving filings in July 2026.

### Employers are keyed on EIN

Done — the pipeline keys on `SPONS_DFE_EIN`, with `Employer` carrying a
canonical display name resolved per EIN (the name that company filed most often)
and disambiguated with the EIN where two companies share a name. `Employer` and
`EIN` are strictly 1:1, so any groupby on either is correct.

Geo now comes from the filing row too, rather than the previous approach of
matching on names with `INC`/`LLC`/`CORP` stripped — which merged distinct
companies. State, city and ZIP coverage went from partial to 99.9%.

Why it mattered, measured on 2023 vs 2024:

- `SPONS_DFE_EIN` is populated on **100%** of filings in both years.
- EIN carries over better than name — 92.6% against 90.6%.
- **5.9% of carried-over companies (7,291) changed their filed name between
  years** — `TERUMO BCT` to `TERUMO BLOOD AND CELL TECHNOLOGIES INC`,
  `G W LISK COMPANY INC` to `G W LISK COMPANY`. Tracking by name reads every one
  of those as a lost account.
- 1,719 names map to more than one EIN *within a single year*, so a name is not
  a unique key even before any trend work.

Schema is stable across 2023/2024/2025 — every column the ETL depends on is
present in all three — and the `(ACK_ID, FORM_ID)` commission join holds at 100%
in every year, so per-product commission works throughout.

### What the trend actually shows

Comparing the two complete years, voluntary is growing while core is flat:

| Product | 2023 | 2024 | Change |
|---|---:|---:|---:|
| Hospital Indemnity | 11,044 | 14,235 | **+28.9%** |
| Critical Illness | 20,394 | 23,201 | +13.8% |
| Legal | 4,519 | 5,055 | +11.9% |
| Accident | 23,169 | 25,826 | +11.5% |
| Long Term Care | 1,514 | 1,487 | −1.8% |
| Cancer | 3,145 | 3,065 | −2.5% |
| *Life (core)* | *58,764* | *58,346* | *−0.7%* |
| *LTD (core)* | *49,052* | *48,966* | *−0.2%* |

Employers holding each product, counted as distinct EINs so a renamed company
is not double counted.

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
