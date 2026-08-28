"""
Data-quality checks on filed Form 5500 figures.

Two tiers, deliberately separated:

EXCLUSIONS are values so large they destroy every total they touch - a single
row reporting $563 trillion in commissions, another putting $6.9 billion on a
170-life group. Those are removed from rollups by the caller.

FLAGS are values that are implausible but not destructive, so they stay in the
totals and are surfaced for a human to judge. Auto-dropping them would be worse
than keeping them: many are real money with one bad field, and silently removing
~$650M of premium because a filer left the headcount blank would be its own
kind of error.

Everything here operates on the contract-level table (one row per Schedule A
contract), which is the grain the money is actually filed at.
"""

import numpy as np
import pandas as pd

# A contract paying more than this per covered life is not credible for the
# group benefits lines in scope. The median is around $313 and the 99th
# percentile around $5,400, so $50k is roughly 160x the typical account.
PREMIUM_PER_LIFE_CAP = 50_000.0


def flag_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    """Return the contracts frame with one boolean column per check, plus
    QualityFlag naming the first check that fired."""
    c = contracts.copy()
    prem = pd.to_numeric(c.get("Premium", 0), errors="coerce").fillna(0)
    comm = pd.to_numeric(c.get("Commission", 0), errors="coerce").fillna(0)
    lives = pd.to_numeric(c.get("Covered_Lives", 0), errors="coerce").fillna(0)

    ppl = np.where(lives > 0, prem / lives.replace(0, np.nan), np.nan)

    c["Flag_NegativeAmount"] = (prem < 0) | (comm < 0)
    c["Flag_ZeroLivesWithMoney"] = (lives <= 0) & ((prem > 0) | (comm > 0))
    c["Flag_CommissionExceedsPremium"] = (prem > 0) & (comm > prem)
    c["Flag_CommissionNoPremium"] = (comm > 0) & (prem <= 0)
    c["Flag_PremiumPerLife"] = pd.Series(ppl, index=c.index) > PREMIUM_PER_LIFE_CAP

    c["PremiumPerLife"] = ppl
    flag_cols = [col for col in c.columns if col.startswith("Flag_")]
    c["AnyQualityFlag"] = c[flag_cols].any(axis=1)

    # Most severe first, so the label names the thing worth acting on.
    order = ["Flag_NegativeAmount", "Flag_CommissionExceedsPremium",
             "Flag_CommissionNoPremium", "Flag_PremiumPerLife",
             "Flag_ZeroLivesWithMoney"]
    label = {
        "Flag_NegativeAmount": "Negative premium or commission",
        "Flag_CommissionExceedsPremium": "Commission exceeds premium",
        "Flag_CommissionNoPremium": "Commission reported with no premium",
        "Flag_PremiumPerLife": f"Premium over ${PREMIUM_PER_LIFE_CAP:,.0f} per covered life",
        "Flag_ZeroLivesWithMoney": "No covered lives reported, but money flowing",
    }
    c["QualityFlag"] = ""
    for col in reversed(order):
        c.loc[c[col], "QualityFlag"] = label[col]
    return c


FLAG_EXPLANATIONS = {
    "Negative premium or commission":
        "A negative amount is not a premium. Usually a refund, rebate or prior-year "
        "adjustment entered without a sign convention. Kept because the offsetting "
        "positive is often on another row of the same filing.",
    "Commission exceeds premium":
        "The broker is shown earning more than the carrier was paid. Several land on "
        "exactly 200% of premium, which points at the same figure being keyed into both "
        "boxes. The commission is usually the wrong one.",
    "Commission reported with no premium":
        "A commission with no premium behind it. The commission is generally real - the "
        "premium box was left empty - so the money is kept but any rate calculated from "
        "it is meaningless.",
    f"Premium over ${PREMIUM_PER_LIFE_CAP:,.0f} per covered life":
        "Against a median around $313 per life. Almost always a headcount entered as a "
        "handful of people while the premium covers the whole group, so the premium is "
        "credible and the lives are not.",
    "No covered lives reported, but money flowing":
        "The largest group by far. Premium and commission are filed but the headcount is "
        "blank or zero, so these employers are invisible to anything measured per life "
        "and score no opportunity at all - they are missing from target lists rather than "
        "wrong in them.",
}


def summarise(flagged: pd.DataFrame) -> pd.DataFrame:
    """One row per check: contracts, employers, and money involved."""
    rows = []
    for col, label in [
        ("Flag_NegativeAmount", "Negative premium or commission"),
        ("Flag_CommissionExceedsPremium", "Commission exceeds premium"),
        ("Flag_CommissionNoPremium", "Commission reported with no premium"),
        ("Flag_PremiumPerLife", f"Premium over ${PREMIUM_PER_LIFE_CAP:,.0f} per covered life"),
        ("Flag_ZeroLivesWithMoney", "No covered lives reported, but money flowing"),
    ]:
        if col not in flagged.columns:
            continue
        sel = flagged[flagged[col]]
        rows.append({
            "Check": label,
            "Contracts": len(sel),
            "Employers": sel["Employer"].nunique() if "Employer" in sel.columns else 0,
            "Premium": float(sel["Premium"].sum()) if "Premium" in sel.columns else 0.0,
            "Commission": float(sel["Commission"].sum()) if "Commission" in sel.columns else 0.0,
            "What it means": FLAG_EXPLANATIONS.get(label, ""),
        })
    return pd.DataFrame(rows).sort_values("Contracts", ascending=False)
