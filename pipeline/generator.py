"""
Input Generator
───────────────
Generates valid input combinations for the pre-generation pipeline.

Iterates over every country in COUNTRY_MAP (not just tiers) so that each
supported country has dedicated rows in Supabase for the Streamlit dropdown.

Business rules:
    - Only Full-Time (FT) employment is generated.
    - Executive (EX) roles at Small (S) companies are filtered out.
    - Cross-border workers (is_same_country=0) cannot have remote_ratio=0.
    - Dummy one-hot columns for employment_type are included so the ML
      model's joblib preprocessor never encounters missing columns.

Usage:
    from pipeline.generator import generate_combinations

    combos = generate_combinations()                         # all countries
    combos = generate_combinations(country_filter="US")      # single country
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
# The API expects a job_title, but the DB stores job_category. We pick one
# representative title per category — preferring the title that matches the
# category name exactly (all 4 do in this dataset).
_category_to_titles: dict[str, list[str]] = {}
for _title, _cat in _mappings["job_category_map"].items():
    _category_to_titles.setdefault(_cat, []).append(_title)

CATEGORY_REPRESENTATIVE: dict[str, str] = {}
for _cat, _titles in sorted(_category_to_titles.items()):
    CATEGORY_REPRESENTATIVE[_cat] = _cat if _cat in _titles else sorted(_titles)[0]

# ── Country map — code → {region, tier} ───────────────────────────────────────
# Every country must have a dedicated row in Supabase for Streamlit dropdowns.
# Countries were grouped into tiers during EDA using median salary quantiles:
# bottom 33% → Low_Tier, 33-66% → Mid_Tier, top 33% → High_Tier.
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

# ── Pruned country list — top 4 per tier by dataset frequency ─────────────────
# Reduces from 50 countries (8,800 combos) to 12 countries (2,112 combos).
# Full COUNTRY_MAP is kept above for reference / future expansion.
ACTIVE_COUNTRIES: list[str] = [
    # High_Tier:  US (355), GB (47), CA (30), DE (28)
    "US", "GB", "CA", "DE",
    # Mid_Tier:   FR (15),  ES (14), GR (11), NL (4)
    "FR", "ES", "GR", "NL",
    # Low_Tier:   IN (24),  PK (3),  MX (3),  BR (3)
    "IN", "PK", "MX", "BR",
]

# ── Cartesian product dimensions ──────────────────────────────────────────────
EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
COMPANY_SIZES = ["S", "M", "L"]
REMOTE_RATIOS = [0, 50, 100]
IS_SAME_COUNTRY_VALUES = [1, 0]

# ── API-only parameter (required by FastAPI but not stored in Supabase) ───────
_WORK_YEAR = 2024

# For is_same_country=0, the API needs employee_residence ≠ company_location.
# We use a fixed dummy residence that is a valid code in the API's tier_map.
# "XX" is NOT valid — the API validates residence against its 50-code list.
_DUMMY_RESIDENCE = "GB"
_DUMMY_RESIDENCE_FOR_GB = "US"  # when company is already in GB


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
        2. Cross-border (is_same_country=0) + fully on-site (remote=0) filtered out.
    Employment pruning (FT only) is handled upstream.
    """
    if experience_level == "EX" and company_size == "S":
        return False

    if is_same_country == 0 and remote_ratio == 0:
        return False

    return True


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_combinations(country_filter: str | None = None) -> list[dict]:
    """
    Generate all valid input combinations for the pre-generation pipeline.

    Iterates over every country in COUNTRY_MAP so each country has dedicated
    rows in Supabase for the Streamlit dropdown filter.

    Parameters
    ----------
    country_filter : str | None
        If set, only generate combinations for this ISO-2 country code.

    Returns
    -------
    list[dict]
        List of valid combination dicts.
    """
    countries = list(ACTIVE_COUNTRIES)
    if country_filter:
        country_filter = country_filter.upper()
        if country_filter not in COUNTRY_MAP:
            raise ValueError(
                f"Unknown country code: '{country_filter}'. "
                f"Valid: {sorted(COUNTRY_MAP.keys())}"
            )
        countries = [country_filter]

    job_categories = sorted(CATEGORY_REPRESENTATIVE.keys())
    combinations: list[dict] = []

    for job_cat, exp, size, country, remote_ratio, same in product(
        job_categories,
        EXPERIENCE_LEVELS,
        COMPANY_SIZES,
        countries,
        REMOTE_RATIOS,
        IS_SAME_COUNTRY_VALUES,
    ):
        if not is_valid_combination(exp, size, same, remote_ratio):
            continue

        # For cross-border: employee_residence must differ from company_location
        # and must be a valid code in the API's country_tier_map.
        if same == 1:
            employee_residence = country
        elif country == "GB":
            employee_residence = _DUMMY_RESIDENCE_FOR_GB
        else:
            employee_residence = _DUMMY_RESIDENCE

        combo = {
            # ── Fields for FastAPI /predict ────────────────────────────
            "job_title": CATEGORY_REPRESENTATIVE[job_cat],
            "experience_level": exp,
            "employment_type": "FT",
            "company_location": country,
            "employee_residence": employee_residence,
            "company_size": size,
            "work_year": _WORK_YEAR,
            "remote_ratio": remote_ratio,
            # ── Metadata (used by orchestrator for Supabase record) ───
            "is_same_country": same,
            "job_category": job_cat,
            "region": COUNTRY_MAP[country]["region"],
            "location_tier": COUNTRY_MAP[country]["tier"],
            # ── Dummy one-hot columns for ML model compatibility ──────
            "employment_type_FT": 1,
            "employment_type_CT": 0,
            "employment_type_FL": 0,
            "employment_type_PT": 0,
        }
        combinations.append(combo)

    logger.info(
        "Generated %d valid combinations (country_filter=%s)",
        len(combinations),
        country_filter,
    )
    return combinations


# ── CLI preview ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Single country preview
    us_combos = generate_combinations(country_filter="US")
    print(f"\nUS combinations: {len(us_combos)}\n")
    for c in us_combos[:8]:
        print(
            f"  {c['job_category']:30s} | {c['experience_level']} | "
            f"{c['company_size']} | remote={c['remote_ratio']:3d} | "
            f"same={c['is_same_country']} | res={c['employee_residence']}"
        )

    # Full count
    all_combos = generate_combinations()
    print(f"\nTotal combinations ({len(ACTIVE_COUNTRIES)} active countries): {len(all_combos)}")

    # Per-tier breakdown
    for tier in ["High_Tier", "Mid_Tier", "Low_Tier"]:
        count = sum(1 for c in all_combos if c["location_tier"] == tier)
        tier_countries = [c for c in ACTIVE_COUNTRIES if COUNTRY_MAP[c]["tier"] == tier]
        print(f"  {tier}: {count} combos ({tier_countries})")
