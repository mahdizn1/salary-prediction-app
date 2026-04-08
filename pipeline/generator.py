"""
Input Generator
───────────────
Generates valid input combinations for the pre-generation pipeline.

Business rules:
    - Only Full-Time (FT) employment is generated.
    - Executive (EX) roles at Small (S) companies are filtered out.
    - Cross-border workers (is_same_country=0) cannot have remote_ratio=0.
    - Dummy one-hot columns for employment_type are included so the ML
      model's joblib preprocessor never encounters missing columns.

The generator outputs API-ready dicts: each contains the 8 fields that
FastAPI /predict expects, plus metadata (is_same_country, region, etc.)
and dummy employment columns.

Usage:
    from pipeline.generator import generate_combinations

    # All valid combinations
    combos = generate_combinations()

    # Filter to a single country for testing
    combos = generate_combinations(country_filter="US")
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

# ── Fixed generation parameters ───────────────────────────────────────────────
EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
COMPANY_SIZES = ["S", "M", "L"]
WORK_YEAR = 2024  # fixed — not stored in Supabase, only needed for the API
IS_SAME_COUNTRY_VALUES = [1, 0]

# Remote ratio defaults (not stored in Supabase, only needed for the API):
#   same country   → 50 (hybrid, sensible middle ground)
#   cross-border   → 100 (fully remote — cannot be 0 per business rules)
_REMOTE_RATIO = {1: 50, 0: 100}

# For cross-border (is_same_country=0), we need a different employee_residence.
# Pick a common default that differs from company_location.
_CROSS_BORDER_RESIDENCE = {"US": "GB"}  # US workers → GB residence
_CROSS_BORDER_DEFAULT = "US"            # everyone else → US residence


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
    # Executive roles at small companies are rare/noisy
    if experience_level == "EX" and company_size == "S":
        return False

    # Cross-border workers must have some remote component
    if is_same_country == 0 and remote_ratio == 0:
        return False

    return True


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_combinations(country_filter: str | None = None) -> list[dict]:
    """
    Generate all valid input combinations for the pre-generation pipeline.

    Each combination is a dict ready to be sent to FastAPI /predict, with
    additional metadata fields and dummy one-hot columns.

    Parameters
    ----------
    country_filter : str | None
        If set, only generate combinations for this ISO-2 country code.
        Useful for testing (e.g. country_filter="US" for a quick run).

    Returns
    -------
    list[dict]
        List of valid combination dicts.
    """
    countries = COUNTRY_MAP.keys()
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

    for job_cat, exp, size, country, same in product(
        job_categories,
        EXPERIENCE_LEVELS,
        COMPANY_SIZES,
        countries,
        IS_SAME_COUNTRY_VALUES,
    ):
        remote_ratio = _REMOTE_RATIO[same]

        if not is_valid_combination(exp, size, same, remote_ratio):
            continue

        # Resolve employee_residence for the API call
        if same == 1:
            employee_residence = country
        else:
            employee_residence = _CROSS_BORDER_RESIDENCE.get(
                country, _CROSS_BORDER_DEFAULT
            )

        combo = {
            # ── Fields for FastAPI /predict ────────────────────────────
            "job_title": CATEGORY_REPRESENTATIVE[job_cat],
            "experience_level": exp,
            "employment_type": "FT",
            "company_location": country,
            "employee_residence": employee_residence,
            "company_size": size,
            "work_year": WORK_YEAR,
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

    # Quick preview: US-only combinations
    combos = generate_combinations(country_filter="US")
    print(f"\nUS combinations: {len(combos)}\n")
    for c in combos[:5]:
        print(
            f"  {c['job_category']:30s} | {c['experience_level']} | "
            f"  {c['company_size']} | same={c['is_same_country']} | "
            f"remote={c['remote_ratio']}"
        )
    if len(combos) > 5:
        print(f"  ... and {len(combos) - 5} more")

    # Full generation count
    all_combos = generate_combinations()
    print(f"\nTotal combinations (all countries): {len(all_combos)}")
