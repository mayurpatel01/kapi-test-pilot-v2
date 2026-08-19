"""
Benefit classification for Form 5500 Schedule A.

Schedule A gives every welfare benefit its own checkbox EXCEPT the voluntary
products. Critical illness, accident and hospital indemnity are all filed under
WLFR_BNFT_OTHER_IND with a free-text label in WLFR_TYPE_BNFT_OTH_TEXT, which is
why OTHER is the most-used box on the form (107,271 of 287,585 rows).

This module turns that free text into products.

Two things make it non-trivial:

1. The text is a list, not a value: "ACCIDENT, CRITICAL ILLNESS, HOSPITAL" is
   one string describing three products. We split into atoms before matching so
   each product is counted once, rather than letting one string match every
   pattern.

2. AD&D contaminates everything. "ACCIDENTAL DEATH AND DISMEMBERMENT" appears in
   45,964 filings -- almost double the entire real voluntary universe -- and a
   naive /ACCIDENT/ match swallows all of it, roughly doubling apparent VB
   market size. AD&D is a life rider, not a worksite product, so it is matched
   FIRST and given its own product rather than being folded into Accident.
"""

import re

import pandas as pd

# Order matters: each atom takes the first rule that matches, so the most
# specific / most contaminating patterns have to come first.
BENEFIT_RULES = [
    # AD&D before anything else -- see module docstring.
    ("AD&D",               r"DISMEMB|ACCIDENTAL\s*DEATH|\bAD\s*&\s*D\b|\bAD&D\b|\bADD\b|\bA\.?D\.?&?D\.?\b"),
    ("Critical Illness",   r"CRITICAL\s*ILL|CRIT\s*ILL|SPECIFIED\s*DISEASE|DREAD\s*DISEASE"),
    ("Cancer",             r"\bCANCER\b"),
    ("Hospital Indemnity", r"HOSPITAL|HOSP\s*INDEM|MEDICAL\s*BRIDGE|MED\s*BRIDGE"),
    ("Accident",           r"\bACCIDENT\b|ACCIDENTAL\s*INJURY|\bACCDENT\b|\bACCIDNET\b"),
    ("Long Term Care",     r"LONG\s*TERM\s*CARE|\bLTC\b"),
    ("Legal",              r"\bLEGAL\b"),
    ("Identity Theft",     r"IDENTITY|\bID\s*THEFT\b"),
    ("Pet",                r"\bPET\b"),
]

# Which bucket each product reports into.
#
# AON does not sell medical, dental or vision, so those checkboxes are never
# read and the free-text classifier drops their atoms -- the dataset is group
# benefits plus the voluntary line, nothing else.
#
# Legal, long term care, identity theft and pet are voluntary products the
# worksite team quotes, so they count toward voluntary penetration. AD&D is
# NOT: it is a life rider that ~87% of groups already carry, and counting it
# pushes apparent penetration to 91% while collapsing the cross-sell list from
# ~26,000 employers to ~4,400. It stays adjacent and is reported separately.
CORE_PRODUCTS = ["Life", "STD", "LTD"]
VOLUNTARY_PRODUCTS = [
    "Accident", "Critical Illness", "Hospital Indemnity", "Cancer",
    "Legal", "Long Term Care", "Identity Theft", "Pet",
]
ADJACENT_PRODUCTS = ["AD&D"]

# The classic worksite trio, quoted together often enough to track on its own.
VB_TRIO = ["Accident", "Critical Illness", "Hospital Indemnity"]


def column_suffix(product: str) -> str:
    """Product name -> a safe column suffix, e.g. 'Critical Illness' -> 'CriticalIllness'."""
    return re.sub(r"[^A-Za-z0-9]", "", product)

PRODUCT_GROUP = (
    {p: "Core" for p in CORE_PRODUCTS}
    | {p: "Voluntary" for p in VOLUNTARY_PRODUCTS}
    | {p: "Adjacent" for p in ADJACENT_PRODUCTS}
)

ALL_PRODUCTS = CORE_PRODUCTS + VOLUNTARY_PRODUCTS + ADJACENT_PRODUCTS

_COMPILED = [(name, re.compile(pat)) for name, pat in BENEFIT_RULES]

# Splits the free-text list into individual benefit atoms. " AND " is included
# because filers write "ACCIDENT AND CRITICAL ILLNESS" as often as they use a
# comma; splitting "ACCIDENTAL DEATH AND DISMEMBERMENT" on it is harmless since
# both halves still match the AD&D rule.
_SPLIT = re.compile(r"[,;/]|\bAND\b|\+")


def classify_atom(atom: str) -> str | None:
    """Map one benefit phrase to a product name, or None if unrecognized."""
    for name, rx in _COMPILED:
        if rx.search(atom):
            return name
    return None


def explode_other_text(df: pd.DataFrame, text_col: str = "WLFR_TYPE_BNFT_OTH_TEXT",
                       keep_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Expand the free-text OTHER column into one row per (filing, product).

    Returns the keep_cols plus a 'Product' column. Rows whose text matches no
    rule are dropped -- they are overwhelmingly EAP, telehealth, supplemental
    life and wellness, none of which are voluntary products.
    """
    keep_cols = keep_cols or []
    txt = df[text_col].fillna("").astype(str).str.strip().str.upper()

    work = df.loc[txt != "", keep_cols].copy()
    work["_atoms"] = txt[txt != ""].str.split(_SPLIT)

    out = work.explode("_atoms")
    out["_atoms"] = out["_atoms"].str.strip()
    out = out[out["_atoms"] != ""]

    out["Product"] = out["_atoms"].map(classify_atom)
    out = out[out["Product"].notna()].drop(columns=["_atoms"])

    # One filing can list the same product twice ("ACCIDENT, VOLUNTARY ACCIDENT").
    return out.drop_duplicates()
