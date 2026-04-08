"""
Input Generator
───────────────
Generates valid input combinations for the pre-generation pipeline.

The ML model uses `location_tier` (High/Mid/Low) as its geographic feature,
NOT individual country codes. Countries were grouped into tiers during EDA
based on median salary quantiles (0.33 / 0.66 thresholds). So the Cartesian
product iterates over tiers, not over all 50 countries.

Business rules:
    - Only Full-Time (FT) employment is generated.
    - Executive (EX) roles at Small (S) companies are filtered out.
    - Cross-border workers (is_same_country=0) cannot have remote_ratio=0.
    - Dummy one-hot columns for employment_type are included so the ML
      model's joblib preprocessor never encounters missing columns.

Usage:
    from pipeline.generator import generate_combinations

    combos = generate_combinations()                         # all tiers
    combos = generate_combinations(tier_filter="High_Tier")  # single tier
"""

import logging
from itertools import product
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

# ── Load model feature mappings ───────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
_mappings = joblib.load(_BASE_DIR / "model" / "feature_mappings.joblib")

# ── Job categories → representative titles ────────────────────────────────────
# The model uses one-hot encoded job_category (4 categories), but the API
# expects a specific job_title string. We pick one representative title per
# category — preferring the title that matches the category name exactly.
_category_to_titles: dict[str, list[str]] = {}
for _title, _cat in _mappings["job_category_map"].items():
    _category_to_titles.setdefault(_cat, []).append(_title)

CATEGORY_REPRESENTATIVE: dict[str, str] = {}
for _cat, _titles in sorted(_category_to_titles.items()):
    CATEGORY_REPRESENTATIVE[_cat] = _cat if _cat in _titles else sorted(_titles)[0]

# ── Country map — code → {region, tier} (extracted from EDA) ─────────────────
# Countries were grouped into tiers during feature engineering using median
# salary quantiles: bottom 33% → Low_Tier, 33-66% → Mid_Tier, top 33% → High_Tier.
COUNTRY_MAP: dict[str, dict[str, str]] = {
    "AE": {"region": "Middle East",     "tier": "High_Tier"},
    "AS": {"region": "Asia",            "tier": "Low_Tier"},
    "AT": {"region": "Europe",          "tier": "High_Tier"},
    "AU": {"region": "Oceania",         "tier": "High_Tier"},
    "BE": {"region": "Europe",          "tier": "High_Tier"},
    "BR": {"region": "South America",   "tier": "Low_Tier"},
    "CA": {"region": "North America",   "tier": "High_Tier"},
    "CH": {"region": "Europe",          "tier": "Mid_Tier"},
    "CL": {"region": "South America",   "tier": "Mid_Tier"},
    "CN": {"region": "Asia",            "tier": "High_Tier"},
    "CO": {"region": "South America",   "tier": "Low_Tier"},
    "CZ": {"region": "Europe",          "tier": "Mid_Tier"},
    "DE": {"region": "Europe",          "tier": "High_Tier"},
    "DK": {"region": "Europe",          "tier": "Mid_Tier"},
    "DZ": {"region": "Africa",          "tier": "High_Tier"},
    "EE": {"region": "Europe",          "tier": "Low_Tier"},
    "ES": {"region": "Europe",          "tier": "Mid_Tier"},
    "FR": {"region": "Europe",          "tier": "Mid_Tier"},
    "GB": {"region": "Europe",          "tier": "High_Tier"},
    "GR": {"region": "Europe",          "tier": "Mid_Tier"},
    "HN": {"region": "Central America", "tier": "Low_Tier"},
    "HR": {"region": "Europe",          "tier": "Mid_Tier"},
    "HU": {"region": "Europe",          "tier": "Low_Tier"},
    "IE": {"region": "Europe",          "tier": "High_Tier"},
    "IL": {"region": "Middle East",     "tier": "High_Tier"},
    "IN": {"region": "Asia",            "tier": "Low_Tier"},
    "IQ": {"region": "Middle East",     "tier": "High_Tier"},
    "IR": {"region": "Middle East",     "tier": "Low_Tier"},
    "IT": {"region": "Europe",          "tier": "Mid_Tier"},
    "JP": {"region": "Asia",            "tier": "High_Tier"},
    "KE": {"region": "Africa",          "tier": "Low_Tier"},
    "LU": {"region": "Europe",          "tier": "Mid_Tier"},
    "MD": {"region": "Europe",          "tier": "Low_Tier"},
    "MT": {"region": "Europe",          "tier": "Low_Tier"},
    "MX": {"region": "North America",   "tier": "Low_Tier"},
    "MY": {"region": "Asia",            "tier": "Mid_Tier"},
    "NG": {"region": "Africa",          "tier": "Low_Tier"},
    "NL": {"region": "Europe",          "tier": "Mid_Tier"},
    "NZ": {"region": "Oceania",         "tier": "High_Tier"},
    "PK": {"region": "Asia",            "tier": "Low_Tier"},
    "PL": {"region": "Europe",          "tier": "Mid_Tier"},
    "PT": {"region": "Europe",          "tier": "Mid_Tier"},
    "RO": {"region": "Europe",          "tier": "Mid_Tier"},
    "RU": {"region": "Europe",          "tier": "High_Tier"},
    "SG": {"region": "Asia",            "tier": "High_Tier"},
    "SI": {"region": "Europe",          "tier": "Mid_Tier"},
    "TR": {"region": "Middle East",     "tier": "Low_Tier"},
    "UA": {"region": "Europe",          "tier": "Low_Tier"},
    "US": {"region": "North America",   "tier": "High_Tier"},
    "VN": {"region": "Asia",            "tier": "Low_Tier"},
}

# ── Tier → representative country (for FastAPI calls) ─────────────────────────
# The API expects a country code, but the model only uses the tier. We pick
# one representative country per tier so all tier-level predictions are correct.
TIER_REPRESENTATIVE: dict[str, str] = {
    "High_Tier": "US",
    "Mid_Tier": "FR",
    "Low_Tier": "IN",
}

# ── Tier → list of countries (for expanding to per-country Supabase rows) ─────
TIER_COUNTRIES: dict[str, list[str]] = {"High_Tier": [], "Mid_Tier": [], "Low_Tier": []}
for _code, _info in COUNTRY_MAP.items():
    TIER_COUNTRIES[_info["tier"]].append(_code)

# ── Cartesian product dimensions ──────────────────────────────────────────────
LOCATION_TIERS = ["High_Tier", "Mid_Tier", "Low_Tier"]
EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
COMPANY_SIZES = ["S", "M", "L"]
IS_SAME_COUNTRY_VALUES = [1, 0]

# ── API-only parameters (not stored in Supabase) ─────────────────────────────
# The FastAPI endpoint requires work_year and remote_ratio, but the Supabase
# schema does not store them. We fix sensible defaults for the batch run.
_WORK_YEAR = 2024
_REMOTE_RATIO = {1: 50, 0: 100}  # same-country → hybrid, cross-border → fully remote


# ── Validation ────────────────────────────────────────────────────────────────

def is_valid_combination(
    experience_level: str,
    company_size: str,
    is_same_country: int,
    remote_ratio: int,
) -> bool:
    """
    Returns True if this combination passes all business-rule filters.

    Rules applied:
        1. EX + S filtered out (rare/noisy in the dataset).
        2. Cross-border (is_same_country=0) + remote_ratio=0 filtered out.
    Employment pruning (FT only) is handled upstream — only FT combinations
    are generated, so no check is needed here.
    """
    if experience_level == "EX" and company_size == "S":
        return False

    if is_same_country == 0 and remote_ratio == 0:
        return False

    return True


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_combinations(tier_filter: str | None = None) -> list[dict]:
    """
    Generate all valid input combinations for the pre-generation pipeline.

    Iterates over (job_category × experience_level × company_size ×
    location_tier × is_same_country). Each combination includes the fields
    needed for the FastAPI call (using a representative country per tier)
    plus metadata and dummy one-hot columns.

    Parameters
    ----------
    tier_filter : str | None
        If set, only generate combinations for this tier.
        One of "High_Tier", "Mid_Tier", "Low_Tier".

    Returns
    -------
    list[dict]
        List of valid combination dicts.
    """
    tiers = LOCATION_TIERS
    if tier_filter:
        if tier_filter not in TIER_REPRESENTATIVE:
            raise ValueError(
                f"Unknown tier: '{tier_filter}'. "
                f"Valid: {LOCATION_TIERS}"
            )
        tiers = [tier_filter]

    job_categories = sorted(CATEGORY_REPRESENTATIVE.keys())
    combinations: list[dict] = []

    for job_cat, exp, size, tier, same in product(
        job_categories,
        EXPERIENCE_LEVELS,
        COMPANY_SIZES,
        tiers,
        IS_SAME_COUNTRY_VALUES,
    ):
        remote_ratio = _REMOTE_RATIO[same]

        if not is_valid_combination(exp, size, same, remote_ratio):
            continue

        rep_country = TIER_REPRESENTATIVE[tier]

        # For cross-border: employee_residence must differ from company_location
        if same == 1:
            employee_residence = rep_country
        else:
            employee_residence = "GB" if rep_country == "US" else "US"

        combo = {
            # ── Fields for FastAPI /predict ────────────────────────────
            "job_title": CATEGORY_REPRESENTATIVE[job_cat],
            "experience_level": exp,
            "employment_type": "FT",
            "company_location": rep_country,
            "employee_residence": employee_residence,
            "company_size": size,
            "work_year": _WORK_YEAR,
            "remote_ratio": remote_ratio,
            # ── Metadata (used by orchestrator for Supabase record) ───
            "is_same_country": same,
            "job_category": job_cat,
            "location_tier": tier,
            # ── Dummy one-hot columns for ML model compatibility ──────
            "employment_type_FT": 1,
            "employment_type_CT": 0,
            "employment_type_FL": 0,
            "employment_type_PT": 0,
        }
        combinations.append(combo)

    logger.info(
        "Generated %d valid combinations (tier_filter=%s)",
        len(combinations),
        tier_filter,
    )
    return combinations


# ── CLI preview ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    combos = generate_combinations()
    print(f"\nTotal combinations: {len(combos)}\n")
    for tier in LOCATION_TIERS:
        count = sum(1 for c in combos if c["location_tier"] == tier)
        print(f"  {tier}: {count}")
    print()
    for c in combos[:5]:
        print(
            f"  {c['job_category']:30s} | {c['experience_level']} | "
            f"{c['company_size']} | {c['location_tier']:10s} | "
            f"same={c['is_same_country']}"
        )
    if len(combos) > 5:
        print(f"  ... and {len(combos) - 5} more")
