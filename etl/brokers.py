"""
Broker company-name matching and tiering.

Single source of truth for app.py and scripts/export_dataset.py, which
previously each carried their own copy of these constants and had already
started to drift.

On AON matching
---------------
The original rule required a name to START WITH one of five full roots
("AON CORPORATION", "AON RISK SERVICES", ...). That was too narrow and scored
88 distinct AON entities as competitors -- $46.8M of commissions across 284
employer filings -- including plain "AON" (which is how IBM files), the whole
AON INSURANCE AGENCY family, every Puerto Rico entity, and misspellings the
filers themselves introduced ("AON CONSUTLING", "AON CONSULTNG").

The rule is now three tests, all anchored on AON as a WHOLE WORD. That word
boundary is what keeps out the real false positives, which contain the letters
"aon" inside another word rather than as a token: SAMMAONS COMPANY LP,
CHERYL LYNN GAONA, DATAONLINE.

Note that DOL truncates the filed broker name at 35 characters, so subsidiary
suffixes arrive mangled -- "AN AON COMPANY" shows up as "AN AON COMP", "AN AON",
or is cut off entirely. Rule B is written to tolerate that.
"""

import re

import pandas as pd


def norm(s) -> str:
    """Uppercase, strip punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Rule A -- the filer IS an AON entity: the name opens with the token AON.
_AON_PREFIX = re.compile(r"^AON\b")

# Rule B -- self-declared subsidiary: "... AN AON COMPANY" / "... (AON)".
# \s* after AN absorbs the "ANAON" typo; the trailing-AON alternative catches
# names truncated mid-suffix. All three alternatives require AON as a token.
_AON_SUBSIDIARY = re.compile(r"\bAN\s*AON\b|\bAON\s*CO|\bAON$")

# Rule C -- a named AON operating unit after a person or office prefix,
# e.g. "ANN BOYER - AON CONSULTING INC.".
_AON_UNIT = re.compile(
    r"\bAON (CONSULTING|CONSUTLING|CONSULTNG|CONSUTING|HEWITT|RISK|"
    r"CORPORATION|GROUP|SECURITIES|SOLUTIONS|SERVICES)\b"
)

TIER1_PATTERNS = {
    "MARSH": [r"\bMARSH\b", r"\bMARSH\s+MCLENNAN\b", r"\bMMC\b"],
    "WTW": [r"\bWILLIS\b", r"\bTOWERS\b", r"\bWTW\b", r"\bWILLIS\s+TOWERS\s+WATSON\b"],
    "GALLAGHER": [r"\bGALLAGHER\b", r"\bARTHUR\s+J\s+GALLAGHER\b"],
    "BROWN & BROWN": [r"\bBROWN\b.*\bBROWN\b", r"\bBROWN\s*&\s*BROWN\b"],
}

TIER_LABEL = {
    "Tier0": "Tier0 - AON (incumbent)",
    "Tier1": "Tier1 - Global major",
    "Tier2": "Tier2 - Large regional/other",
    "Tier3": "Tier3 - Small/local",
}


def is_aon_composite(broker_name: str) -> bool:
    n = norm(broker_name)
    return bool(_AON_PREFIX.search(n) or _AON_SUBSIDIARY.search(n) or _AON_UNIT.search(n))


def aon_match_rule(broker_name: str) -> str:
    """Which rule fired -- surfaced in the export's name-matching audit sheet."""
    n = norm(broker_name)
    if _AON_PREFIX.search(n):
        return "AON composite (name starts with AON)"
    if _AON_SUBSIDIARY.search(n):
        return "AON composite (declared AON subsidiary)"
    if _AON_UNIT.search(n):
        return "AON composite (named AON operating unit)"
    return ""


def match_any(patterns, text) -> bool:
    return any(re.search(p, text) for p in patterns)


def broker_family(broker_name: str) -> str:
    n = norm(broker_name)
    if not n:
        return "UNKNOWN"
    if is_aon_composite(n):
        return "AON"
    for fam, pats in TIER1_PATTERNS.items():
        if match_any(pats, n):
            return fam
    return "OTHER"


def match_rule(broker_name: str) -> str:
    """Human-readable explanation of how a name was classified."""
    rule = aon_match_rule(broker_name)
    if rule:
        return rule
    fam = broker_family(broker_name)
    return f"Tier1 major (keyword match: {fam})" if fam != "OTHER" else "No match - OTHER"


def assign_tiers(broker_agg: pd.DataFrame, tier2_pct: float = 0.10) -> pd.DataFrame:
    """Tier0 = AON, Tier1 = named majors, Tier2 = top X% of 'Other' by lives, else Tier3."""
    out = broker_agg.copy()
    out["Tier"] = "Tier3"
    out.loc[out["BrokerFamily"] == "AON", "Tier"] = "Tier0"
    out.loc[out["BrokerFamily"].isin(list(TIER1_PATTERNS.keys())), "Tier"] = "Tier1"

    mask_other = out["BrokerFamily"].eq("OTHER")
    other = out[mask_other]
    if len(other) > 0:
        thresh = other["CoveredLives"].quantile(1 - tier2_pct)
        out.loc[mask_other & (out["CoveredLives"] >= thresh), "Tier"] = "Tier2"
        out.loc[mask_other & (out["CoveredLives"] < thresh), "Tier"] = "Tier3"
    return out
