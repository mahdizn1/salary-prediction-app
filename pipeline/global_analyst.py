"""
Global Analyst Module (Gemini 2.5 Flash)
──────────────────────────────────────
Pure data processing + LLM module. No database logic, no execution flow.

Layer 1 — Pandas Aggregation:
    calculate_market_stats(df) pre-computes every metric the dashboard needs.
    Nothing is left for the LLM to calculate — it only interprets.

Layer 2 — Gemini Structured JSON:
    generate_summary(stats_dict) feeds the full stats payload to Gemini 2.5 Flash
    and forces a structured JSON response containing the executive summary,
    chart captions, and a Data Transparency (XAI) note.

Public API:
    get_global_insights_payload() → dict
        Reads CSV, enriches, aggregates, calls Gemini, returns the final payload.
        The Orchestrator calls this and handles persistence.

Micro-narratives (per-record) remain on the local Ollama instance in llm_analyst.py.
"""

import json
import logging
import os
from pathlib import Path

import google.generativeai as genai
import joblib
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Load .env from the pipeline directory ────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Configure Gemini ─────────────────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ── Data paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = _PROJECT_ROOT / "data" / "ds_salaries.csv"
MAPPINGS_PATH = _PROJECT_ROOT / "model" / "feature_mappings.joblib"
GEMINI_CACHE_FILE = _PROJECT_ROOT / "data" / "gemini_dev_cache.json"

# ── Country → Region mapping (mirrors orchestrator.py) ──────────────────────
COUNTRY_REGION: dict[str, str] = {
    "AE": "Middle East", "AS": "Asia", "AT": "Europe", "AU": "Oceania",
    "BE": "Europe", "BR": "South America", "CA": "North America",
    "CH": "Europe", "CL": "South America", "CN": "Asia", "CO": "South America",
    "CZ": "Europe", "DE": "Europe", "DK": "Europe", "DZ": "Africa",
    "EE": "Europe", "ES": "Europe", "FR": "Europe", "GB": "Europe",
    "GR": "Europe", "HN": "Central America", "HR": "Europe", "HU": "Europe",
    "IE": "Europe", "IL": "Middle East", "IN": "Asia", "IQ": "Middle East",
    "IR": "Middle East", "IT": "Europe", "JP": "Asia", "KE": "Africa",
    "LU": "Europe", "MD": "Europe", "MT": "Europe", "MX": "North America",
    "MY": "Asia", "NG": "Africa", "NL": "Europe", "NZ": "Oceania",
    "PK": "Asia", "PL": "Europe", "PT": "Europe", "RO": "Europe",
    "RU": "Europe", "SG": "Asia", "SI": "Europe", "TR": "Middle East",
    "UA": "Europe", "US": "North America", "VN": "Asia",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: Pandas Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive job_category and region from the raw ds_salaries.csv columns
    using the saved feature mappings and the country→region lookup.
    """
    mappings = joblib.load(MAPPINGS_PATH)
    job_category_map = mappings["job_category_map"]

    df = df.copy()
    df["job_category"] = df["job_title"].map(job_category_map)
    df["region"] = df["company_location"].map(COUNTRY_REGION).fillna("Other")
    # Drop rows with unmapped job titles (shouldn't happen, but safety)
    df = df.dropna(subset=["job_category"])
    return df


def calculate_market_stats(df: pd.DataFrame) -> dict:
    """
    Pre-compute every aggregate metric the dashboard and LLM need.

    Every metric includes its sample size (n) so the LLM can reason
    about statistical weight and flag low-confidence segments.

    Parameters
    ----------
    df : pd.DataFrame
        The original ds_salaries.csv enriched with job_category and region.
        Uses salary_in_usd as the salary column.

    Returns
    -------
    dict
        Multi-level dictionary ready to be serialised as JSON for the LLM.
    """
    total = len(df)
    global_median = float(df["salary_in_usd"].median())

    # ── Seniority ladder ─────────────────────────────────────────────────
    exp_order = ["EN", "MI", "SE", "EX"]
    exp_labels = {"EN": "Entry-Level", "MI": "Mid-Level", "SE": "Senior", "EX": "Executive"}
    seniority = (
        df.groupby("experience_level")["salary_in_usd"]
        .agg(["median", "count"])
        .reindex(exp_order)
    )
    seniority_ladder = {
        exp_labels[lvl]: {"median_salary": round(float(row["median"])), "n": int(row["count"])}
        for lvl, row in seniority.iterrows()
    }

    # ── Regional comparison ──────────────────────────────────────────────
    regional = (
        df.groupby("region")["salary_in_usd"]
        .agg(["median", "count"])
        .sort_values("median", ascending=False)
    )
    regional_comparison = {
        region: {"median_salary": round(float(row["median"])), "n": int(row["count"])}
        for region, row in regional.iterrows()
    }

    # ── Role distribution ────────────────────────────────────────────────
    role_counts = df["job_category"].value_counts()
    role_distribution = {role: int(n) for role, n in role_counts.items()}

    # ── Category medians ─────────────────────────────────────────────────
    category_medians = {
        cat: round(float(med))
        for cat, med in df.groupby("job_category")["salary_in_usd"].median().items()
    }

    # ── Remote vs on-site ────────────────────────────────────────────────
    remote_full = df[df["remote_ratio"] == 100]["salary_in_usd"].median()
    onsite_full = df[df["remote_ratio"] == 0]["salary_in_usd"].median()

    # Executive remote premium
    ex_remote = df[(df["experience_level"] == "EX") & (df["remote_ratio"] == 100)]
    ex_onsite = df[(df["experience_level"] == "EX") & (df["remote_ratio"] == 0)]

    remote_vs_onsite = {
        "fully_remote": {
            "median_salary": round(float(remote_full)),
            "n": int((df["remote_ratio"] == 100).sum()),
        },
        "fully_onsite": {
            "median_salary": round(float(onsite_full)),
            "n": int((df["remote_ratio"] == 0).sum()),
        },
        "executive_remote": {
            "median_salary": round(float(ex_remote["salary_in_usd"].median())) if len(ex_remote) else 0,
            "n": len(ex_remote),
        },
        "executive_onsite": {
            "median_salary": round(float(ex_onsite["salary_in_usd"].median())) if len(ex_onsite) else 0,
            "n": len(ex_onsite),
        },
    }

    # ── Company size dynamics (experience × size) ────────────────────────
    size_order = ["S", "M", "L"]
    size_labels = {"S": "Small", "M": "Medium", "L": "Large"}
    cs_pivot = (
        df.groupby(["experience_level", "company_size"])["salary_in_usd"]
        .agg(["median", "count"])
    )
    company_size_dynamics = {}
    for exp in exp_order:
        company_size_dynamics[exp_labels[exp]] = {}
        for size in size_order:
            if (exp, size) in cs_pivot.index:
                row = cs_pivot.loc[(exp, size)]
                company_size_dynamics[exp_labels[exp]][size_labels[size]] = {
                    "median_salary": round(float(row["median"])),
                    "n": int(row["count"]),
                }

    # ── US market deep dive ──────────────────────────────────────────────
    us = df[df["company_location"] == "US"]
    us_by_exp = us.groupby("experience_level")["salary_in_usd"].agg(["median", "count"])
    us_by_size = us.groupby("company_size")["salary_in_usd"].agg(["median", "count"])
    us_by_cat = us.groupby("job_category")["salary_in_usd"].agg(["median", "count"])

    us_market = {
        "total_us_rows": len(us),
        "us_median_salary": round(float(us["salary_in_usd"].median())) if len(us) else 0,
        "by_experience": {
            exp_labels.get(lvl, lvl): {"median_salary": round(float(row["median"])), "n": int(row["count"])}
            for lvl, row in us_by_exp.iterrows()
        },
        "by_company_size": {
            size_labels.get(s, s): {"median_salary": round(float(row["median"])), "n": int(row["count"])}
            for s, row in us_by_size.iterrows()
        },
        "by_job_category": {
            cat: {"median_salary": round(float(row["median"])), "n": int(row["count"])}
            for cat, row in us_by_cat.iterrows()
        },
    }

    # ── Dataset demographics (XAI transparency) ─────────────────────────
    region_demo = df["region"].value_counts()
    dataset_demographics = {
        region: {"n": int(n), "pct_of_total": round(n / total * 100, 1)}
        for region, n in region_demo.items()
    }

    return {
        "global_metrics": {
            "total_rows": total,
            "global_median_salary": round(global_median),
        },
        "seniority_ladder": seniority_ladder,
        "regional_comparison": regional_comparison,
        "role_distribution": role_distribution,
        "category_medians": category_medians,
        "remote_vs_onsite": remote_vs_onsite,
        "company_size_dynamics": company_size_dynamics,
        "us_market_deep_dive": us_market,
        "dataset_demographics": dataset_demographics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: Gemini 2.5 Flash — Structured JSON Narrative
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an elite Senior Labor Economist. You will receive a JSON payload "
    "of pre-calculated salary aggregates for the Data Science market.\n\n"
    "CONTEXT & METHODOLOGY:\n"
    "- Experience strongly predicts salary. Large companies pay a premium.\n"
    "- Dataset is heavily skewed towards North America and Europe.\n"
    "- Countries were grouped into Location Tiers as a variance-reduction "
    "technique to stabilize predictions for emerging markets with sparse data.\n\n"
    "WRITING RULES (STRICT):\n"
    "1. NEVER use JSON key names, variable names, or snake_case identifiers "
    "in any text. Write in natural, professional English only.\n"
    "2. Executive summary: Tell a STORY, not a data dump. Focus on the "
    "narrative of climbing the career ladder, the massive US market premium, "
    "and the startup penalty. Weave numbers in naturally — do NOT list every "
    "single median. Lead with insight, not statistics.\n"
    "3. Captions: State the mathematical INSIGHT, not a description. "
    "NEVER start with 'This chart shows...' or 'This chart illustrates...'. "
    "Example: 'Executives earn nearly 3x more than Entry-Level roles.' "
    "NOT 'This chart shows salary progression across experience levels.'\n"
    "4. XAI note: Project CONFIDENCE. US and European predictions carry the "
    "highest statistical confidence. The Location Tier system ensures global "
    "robustness by stabilizing predictions for underrepresented markets. "
    "Do NOT use the words 'high variance', 'bias', or 'limitation'.\n\n"
    "Return a valid JSON object with EXACTLY this schema:\n"
    "{\n"
    '  "executive_summary": "3-paragraph narrative (career ladder → '
    'US premium → company size dynamics).",\n'
    '  "data_transparency_note": "1-paragraph confidence-projecting XAI note.",\n'
    '  "captions": {\n'
    '    "seniority_ladder": "1 sentence insight.",\n'
    '    "regional_comparison": "1 sentence insight.",\n'
    '    "role_distribution": "1 sentence insight.",\n'
    '    "remote_premium": "1 sentence insight.",\n'
    '    "heatmap_job_region": "1 sentence insight.",\n'
    '    "regional_representation": "1 sentence insight.",\n'
    '    "us_deep_dive": "1 sentence insight."\n'
    "  }\n"
    "}"
)


def generate_summary(stats_dict: dict, force_refresh: bool = False) -> dict | None:
    """
    Sends the full aggregated stats payload to Gemini 2.5 Flash and returns
    a structured JSON response with the executive summary, XAI note, and
    chart captions.

    Implements a local dev cache to avoid burning API quota during development.
    Pass force_refresh=True to bypass the cache and hit the API.

    Parameters
    ----------
    stats_dict : dict
        The output of calculate_market_stats().
    force_refresh : bool
        If True, skip the cache and call Gemini.

    Returns
    -------
    dict | None
        Parsed JSON with keys: executive_summary, data_transparency_note, captions.
        None on failure.
    """
    # ── Dev cache check ──────────────────────────────────────────────────
    if not force_refresh and GEMINI_CACHE_FILE.exists():
        logger.info("Using local cached Gemini response to save API quota.")
        with open(GEMINI_CACHE_FILE, "r") as f:
            return json.load(f)

    # ── API call ─────────────────────────────────────────────────────────
    prompt = (
        "Here is the aggregated market data payload:\n"
        f"{json.dumps(stats_dict, indent=2)}"
    )

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        logger.info("Gemini returned structured JSON with keys: %s", list(result.keys()))

        # ── Save to dev cache ────────────────────────────────────────────
        GEMINI_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GEMINI_CACHE_FILE, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Gemini response cached to %s", GEMINI_CACHE_FILE)

        return result

    except json.JSONDecodeError as e:
        logger.error("Gemini returned invalid JSON: %s", e)
        return None

    except Exception as e:
        logger.error("Gemini API failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Called by the Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def get_global_insights_payload() -> dict | None:
    """
    End-to-end data processing: CSV → enrich → aggregate → Gemini → payload.

    Returns the final dictionary ready for the Orchestrator to persist,
    or None on any failure.

    Returns
    -------
    dict | None
        On success: {
            "executive_summary": str,
            "data_transparency_note": str,
            "chart_captions": dict,
            "category_medians": dict,
            "market_stats_json": dict,
        }
        On failure: None
    """
    # ── Load and enrich ──────────────────────────────────────────────────
    if not DATASET_CSV.exists():
        logger.error("ds_salaries.csv not found at %s", DATASET_CSV)
        return None

    df = _enrich_dataset(pd.read_csv(DATASET_CSV))
    logger.info("Loaded %d rows from ds_salaries.csv", len(df))

    # ── Aggregate ────────────────────────────────────────────────────────
    stats = calculate_market_stats(df)
    logger.info(
        "Aggregation complete — global median: $%s, %d regions, %d roles",
        f"{stats['global_metrics']['global_median_salary']:,}",
        len(stats["regional_comparison"]),
        len(stats["role_distribution"]),
    )

    # ── Generate narrative via Gemini ────────────────────────────────────
    response_json = generate_summary(stats)
    if response_json is None:
        logger.error("Gemini failed to produce a valid summary")
        return None

    summary = response_json.get("executive_summary", "")
    logger.info("Executive summary generated (%d chars)", len(summary))

    return {
        "executive_summary": response_json.get("executive_summary", ""),
        "data_transparency_note": response_json.get("data_transparency_note", ""),
        "chart_captions": response_json.get("captions", {}),
        "category_medians": stats["category_medians"],
        "market_stats_json": stats,
    }
