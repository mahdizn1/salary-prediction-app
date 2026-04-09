"""
Global Analyst Module (Gemini 2.5 Flash)
──────────────────────────────────────
Two-Layer Narrative Architecture + Explainable AI (XAI) Transparency Layer.

Layer 1 — Pandas Aggregation:
    calculate_market_stats(df) pre-computes every metric the dashboard needs.
    Nothing is left for the LLM to calculate — it only interprets.

Layer 2 — Gemini Structured JSON:
    generate_summary(stats_dict) feeds the full stats payload to Gemini 2.5 Flash
    and forces a structured JSON response containing the executive summary,
    chart captions, and a Data Transparency (XAI) note.

Persistence:
    run_global_analysis() orchestrates aggregation → LLM → Supabase upsert.

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
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ── Load .env from the pipeline directory ────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Configure Gemini ─────────────────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ── Supabase credentials ────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

# ── Data paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = _PROJECT_ROOT / "data" / "ds_salaries.csv"
MAPPINGS_PATH = _PROJECT_ROOT / "model" / "feature_mappings.joblib"

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
# LAYER 2: Gemini 2.0 Flash — Structured JSON Narrative
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an elite Senior Labor Economist and Data Scientist. "
    "You will receive a JSON payload containing precise, pre-calculated salary "
    "aggregates for the global Data Science market (~600 rows).\n\n"
    "CONTEXT FROM PREVIOUS EDA & METHODOLOGY:\n"
    "- Experience is the strongest predictor of salary, with a massive jump at "
    "the Executive level.\n"
    "- Large companies pay a premium, especially to senior talent.\n"
    "- The dataset has a significant geographic skew towards North America "
    "and Europe.\n"
    "- Methodology Note: During feature engineering, countries were grouped into "
    "Location Tiers (High/Mid/Low) using statistical quantiles (33rd/66th "
    "percentiles) of their median salaries. Because of data sparsity in some "
    "regions, countries with a sample size of 1 may be disproportionately "
    "classified based on that single outlier.\n\n"
    "Your task is to write a cohesive data narrative and a professional "
    "Data Transparency (XAI) note.\n"
    "Do NOT hallucinate numbers. ONLY use the numbers provided in the payload.\n\n"
    "Return a valid JSON object with EXACTLY this schema:\n"
    "{\n"
    '  "executive_summary": "A 3-paragraph cohesive narrative telling the '
    "global market story (Geography -> Experience -> Company Size). "
    'Explicitly reference the charts below.",\n'
    '  "data_transparency_note": "A 1-paragraph Explainable AI (XAI) note '
    "explaining the dataset's geographic skew (North America/Europe) and openly "
    "acknowledging that using statistical quantiles for Location Tiers on sparse "
    "data introduces potential high-variance classifications for underrepresented "
    "countries. Frame this professionally as 'Known Technical Limitations' or "
    "'Methodology Caveats'. This will serve as a transition into the US-specific "
    'market deep dive.",\n'
    '  "captions": {\n'
    '    "seniority_ladder": "One sentence caption pointing out the steepest jump.",\n'
    '    "regional_comparison": "One sentence caption highlighting the top vs bottom region.",\n'
    '    "role_distribution": "One sentence caption highlighting the most in-demand role.",\n'
    '    "remote_premium": "One sentence caption on remote vs on-site pay.",\n'
    '    "heatmap_job_region": "One sentence caption on the highest paying role/region combo.",\n'
    '    "regional_representation": "One sentence caption highlighting the dominance '
    'of North America and Europe in the training data.",\n'
    '    "us_deep_dive": "One sentence caption summarizing the uniqueness of the '
    'US market based on the data."\n'
    "  }\n"
    "}"
)

FALLBACK_RESPONSE = {
    "executive_summary": "Market analysis temporarily unavailable.",
    "data_transparency_note": "",
    "captions": {},
}


def generate_summary(stats_dict: dict) -> dict:
    """
    Sends the full aggregated stats payload to Gemini 2.5 Flash and returns
    a structured JSON response with the executive summary, XAI note, and
    chart captions.

    Parameters
    ----------
    stats_dict : dict
        The output of calculate_market_stats().

    Returns
    -------
    dict
        Parsed JSON with keys: executive_summary, data_transparency_note, captions.
    """
    prompt = (
        "Here is the aggregated market data payload:\n"
        f"{json.dumps(stats_dict, indent=2)}"
    )

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        parsed = json.loads(response.text)
        logger.info("Gemini returned structured JSON with keys: %s", list(parsed.keys()))
        return parsed

    except json.JSONDecodeError as e:
        logger.error("Gemini returned invalid JSON: %s", e)
        return FALLBACK_RESPONSE

    except Exception as e:
        logger.error("Gemini API failed: %s", e)
        return FALLBACK_RESPONSE


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: Orchestration — Aggregate → LLM → Supabase
# ═══════════════════════════════════════════════════════════════════════════════

def run_global_analysis() -> bool:
    """
    End-to-end global analysis pipeline:
    1. Load ds_salaries.csv (original dataset)
    2. Enrich with derived features (job_category, region)
    3. Aggregate with Pandas
    4. Generate narrative with Gemini
    5. Upsert to Supabase global_insights table

    Returns
    -------
    bool
        True on success, False on any failure.
    """
    # ── Step 1: Load and enrich data ─────────────────────────────────────
    if not DATASET_CSV.exists():
        logger.error("ds_salaries.csv not found at %s", DATASET_CSV)
        return False

    df = _enrich_dataset(pd.read_csv(DATASET_CSV))
    logger.info("Loaded %d rows from ds_salaries.csv", len(df))

    # ── Step 2: Aggregate ────────────────────────────────────────────────
    stats = calculate_market_stats(df)
    logger.info(
        "Aggregation complete — global median: $%s, %d regions, %d roles",
        f"{stats['global_metrics']['global_median_salary']:,}",
        len(stats["regional_comparison"]),
        len(stats["role_distribution"]),
    )

    # ── Step 3: Generate narrative via Gemini ────────────────────────────
    response_json = generate_summary(stats)
    summary = response_json.get("executive_summary", "")
    if not summary or summary == FALLBACK_RESPONSE["executive_summary"]:
        logger.error("Gemini failed to produce a valid summary")
        return False

    logger.info("Executive summary generated (%.0f chars)", len(summary))

    # ── Step 4: Upsert to Supabase ──────────────────────────────────────
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase credentials not set — skipping DB upsert")
        # Still print the result for local testing
        print("\n── Executive Summary ──────────────────────────────────────")
        print(summary)
        print("\n── Data Transparency Note ────────────────────────────────")
        print(response_json.get("data_transparency_note", ""))
        print("\n── Chart Captions ────────────────────────────────────────")
        for key, caption in response_json.get("captions", {}).items():
            print(f"  {key}: {caption}")
        print("───────────────────────────────────────────────────────────")
        return False

    data = {
        "id": 1,  # singleton row — always upsert the same record
        "executive_summary": response_json.get("executive_summary", ""),
        "data_transparency_note": response_json.get("data_transparency_note", ""),
        "chart_captions": response_json.get("captions", {}),
        "category_medians": stats["category_medians"],
        "market_stats_json": stats,
    }

    try:
        client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        client.table("global_insights").upsert(data).execute()
        logger.info("Supabase upsert OK → global_insights (id=1)")
        return True

    except Exception as exc:
        logger.error("Supabase upsert failed: %s", exc)
        return False


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    success = run_global_analysis()
    raise SystemExit(0 if success else 1)
