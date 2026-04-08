"""
Orchestrator Script
───────────────────
The "brain" of the pre-generation pipeline. Coordinates the FastAPI prediction
server, the Ollama LLM analyst, and Supabase persistence.

Usage:
    python pipeline/orchestrator.py --step predict
    python pipeline/orchestrator.py --step analyze
    python pipeline/orchestrator.py --step push_db
    python pipeline/orchestrator.py --step full_pipeline

Steps:
    predict       → call FastAPI for predictions only (no LLM, no DB)
    analyze       → prediction + LLM narrative (no DB)
    push_db       → full pipeline: predict → analyze → insert into Supabase
    full_pipeline → alias for push_db

Error contract:
    A failure on any single record is logged and skipped. The loop always
    continues. A summary of successes and failures is printed at the end.
"""

import argparse
import csv
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import requests
from supabase import create_client, Client

from pipeline.generator import generate_combinations
from pipeline.llm_analyst import generate_micro_narrative, generate_global_summary

# ── Load .env from the pipeline directory ─────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── External service URLs ──────────────────────────────────────────────────────
FASTAPI_URL = "http://localhost:8000/predict"

# ── Supabase credentials — loaded from environment, never hardcoded ────────────
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

# ── Country-code → region mapping ─────────────────────────────────────────────
# Covers all 50 ISO-2 codes in the model's country_tier_map.
COUNTRY_REGION: dict[str, str] = {
    "AE": "Middle East",
    "AS": "Asia",
    "AT": "Europe",
    "AU": "Oceania",
    "BE": "Europe",
    "BR": "South America",
    "CA": "North America",
    "CH": "Europe",
    "CL": "South America",
    "CN": "Asia",
    "CO": "South America",
    "CZ": "Europe",
    "DE": "Europe",
    "DK": "Europe",
    "DZ": "Africa",
    "EE": "Europe",
    "ES": "Europe",
    "FR": "Europe",
    "GB": "Europe",
    "GR": "Europe",
    "HN": "Central America",
    "HR": "Europe",
    "HU": "Europe",
    "IE": "Europe",
    "IL": "Middle East",
    "IN": "Asia",
    "IQ": "Middle East",
    "IR": "Middle East",
    "IT": "Europe",
    "JP": "Asia",
    "KE": "Africa",
    "LU": "Europe",
    "MD": "Europe",
    "MT": "Europe",
    "MX": "North America",
    "MY": "Asia",
    "NG": "Africa",
    "NL": "Europe",
    "NZ": "Oceania",
    "PK": "Asia",
    "PL": "Europe",
    "PT": "Europe",
    "RO": "Europe",
    "RU": "Europe",
    "SG": "Asia",
    "SI": "Europe",
    "TR": "Middle East",
    "UA": "Europe",
    "US": "North America",
    "VN": "Asia",
}

# ── Reference statistics for the LLM analyst ──────────────────────────────────
# These are approximate market medians computed during EDA.
# The LLM uses them to contextualise the predicted salary.
GLOBAL_MEDIANS: dict = {
    "overall_median": 110_000,
    "median_entry": 67_000,
    "median_mid": 100_000,
    "median_senior": 140_000,
    "median_executive": 180_000,
}

# ── Sample input combinations (hardcoded for integration testing) ──────────────
# These 3 combinations exercise the most common input dimensions.
# The full pipeline script will replace this with the generated combination grid.
SAMPLE_COMBINATIONS: list[dict] = [
    {
        "job_title": "Data Scientist",
        "experience_level": "SE",
        "employment_type": "FT",
        "company_location": "US",
        "employee_residence": "US",
        "company_size": "L",
        "work_year": 2024,
        "remote_ratio": 100,
    },
    {
        "job_title": "ML Engineer",
        "experience_level": "MI",
        "employment_type": "FT",
        "company_location": "GB",
        "employee_residence": "GB",
        "company_size": "M",
        "work_year": 2023,
        "remote_ratio": 50,
    },
    {
        "job_title": "Data Analyst",
        "experience_level": "EN",
        "employment_type": "CT",
        "company_location": "DE",
        "employee_residence": "DE",
        "company_size": "S",
        "work_year": 2022,
        "remote_ratio": 0,
    },
]


# ── Network functions ──────────────────────────────────────────────────────────

def call_fastapi(payload: dict) -> dict | None:
    """
    Calls GET /predict on the local FastAPI server.

    Parameters
    ----------
    payload : dict
        All 8 query parameters the endpoint expects.

    Returns
    -------
    dict | None
        Full API response dict (predicted_salary_usd + derived inputs),
        or None on any failure.
    """
    try:
        response = requests.get(FASTAPI_URL, params=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        logger.error(
            "FastAPI is not running at %s. "
            "Start it with: uvicorn api.main:app --reload",
            FASTAPI_URL,
        )
        return None

    except requests.exceptions.Timeout:
        logger.error("FastAPI request timed out for job: %s", payload.get("job_title"))
        return None

    except requests.exceptions.HTTPError as exc:
        logger.error(
            "FastAPI HTTP %s for '%s': %s",
            exc.response.status_code,
            payload.get("job_title"),
            exc.response.text,
        )
        return None

    except (KeyError, ValueError, TypeError) as exc:
        logger.error(
            "Unexpected FastAPI response format for '%s': %s",
            payload.get("job_title"),
            exc,
        )
        return None

    except Exception as exc:
        logger.error(
            "Unexpected error calling FastAPI for '%s': %s",
            payload.get("job_title"),
            exc,
            exc_info=True,
        )
        return None


def call_llm(payload: dict, prediction: float) -> str:
    """
    Calls the LLM analyst module to generate a narrative for this record.

    This function never returns None — the LLM module always provides a
    fallback string on failure. We wrap it here for symmetry with the
    other network functions and to add orchestrator-level logging.

    Parameters
    ----------
    payload : dict
        Original input combination (same dict sent to FastAPI).
    prediction : float
        Predicted salary in USD.

    Returns
    -------
    str
        LLM-generated narrative, or the fallback string if Ollama failed.
    """
    narrative = generate_micro_narrative(payload, prediction, GLOBAL_MEDIANS)
    logger.info(
        "Narrative generated for '%s' (%s): %.80s…",
        payload.get("job_title"),
        payload.get("experience_level"),
        narrative,
    )
    return narrative


def push_to_supabase(final_record: dict) -> bool:
    """
    Inserts one prediction record into the Supabase `precomputed_salaries` table.

    Parameters
    ----------
    final_record : dict
        The fully assembled record matching the precomputed_salaries schema.

    Returns
    -------
    bool
        True on success, False on any failure.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error(
            "Supabase credentials not set. "
            "Export SUPABASE_URL and SUPABASE_KEY environment variables."
        )
        return False

    try:
        client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        client.table("precomputed_salaries").insert(final_record).execute()
        logger.info(
            "Supabase insert OK: '%s' (%s, %s) → $%s",
            final_record.get("job_category"),
            final_record.get("experience_level"),
            final_record.get("country_code"),
            f"{final_record.get('predicted_salary', 0):,.0f}",
        )
        return True

    except Exception as exc:
        logger.error(
            "Supabase insert failed for '%s': %s",
            final_record.get("job_category"),
            exc,
        )
        return False


# ── Pipeline step functions ────────────────────────────────────────────────────

def run_predict(combinations: list[dict]) -> None:
    """
    Runs only the prediction step — calls FastAPI for each combination
    and prints the result. No LLM call, no database write.
    """
    print(f"\nRunning PREDICT step for {len(combinations)} combination(s)…\n")
    for combo in combinations:
        api_resp = call_fastapi(combo)
        if api_resp is None:
            print(f"  SKIP  {combo.get('job_title')} ({combo.get('experience_level')}) — prediction failed")
            continue
        salary = api_resp["predicted_salary_usd"]
        inputs = api_resp["inputs"]
        print(
            f"  OK    {inputs['job_category']} ({inputs['experience_level']}, "
            f"{inputs['company_location']}) → ${salary:,.0f}"
        )


def run_analyze(combinations: list[dict]) -> None:
    """
    Runs prediction + LLM analysis — calls FastAPI then Ollama for each
    combination and prints the narrative. No database write.
    """
    print(f"\nRunning ANALYZE step for {len(combinations)} combination(s)…\n")
    for combo in combinations:
        api_resp = call_fastapi(combo)
        if api_resp is None:
            print(f"  SKIP  {combo.get('job_title')} — prediction failed, skipping LLM\n")
            continue

        salary = api_resp["predicted_salary_usd"]
        inputs = api_resp["inputs"]
        narrative = call_llm(combo, salary)
        print(
            f"  [{inputs['job_category']} | {inputs['experience_level']} | "
            f"{inputs['company_location']}]\n"
            f"  Salary: ${salary:,.0f}\n"
            f"  Narrative: {narrative}\n"
        )


def run_full_pipeline(combinations: list[dict], push: bool = True) -> None:
    """
    Runs the complete pipeline: predict → analyze → persist.

    A failure at any step logs a warning and moves on to the next record.
    This is critical resilience behaviour — the loop must never crash.

    Parameters
    ----------
    combinations : list[dict]
        Input combinations to process.
    push : bool
        If True, inserts each assembled record into Supabase.
        Set to False for a dry-run that skips the database write.
    """
    total = len(combinations)
    print(f"\nRunning FULL PIPELINE for {total} combination(s)…\n")

    success_count = 0
    fail_count = 0

    for i, combo in enumerate(combinations, start=1):
        job_title = combo.get("job_title", "Unknown")
        exp = combo.get("experience_level", "?")
        loc = combo.get("company_location", "?")
        logger.info("Processing [%d/%d]: %s | %s | %s", i, total, job_title, exp, loc)

        # ── Step 1: Predict ────────────────────────────────────────────────
        api_resp = call_fastapi(combo)
        if api_resp is None:
            logger.warning("SKIP [%d/%d]: prediction failed for %s", i, total, job_title)
            fail_count += 1
            continue

        salary = api_resp["predicted_salary_usd"]
        inputs = api_resp["inputs"]

        # ── Step 2: LLM narrative (never None) ────────────────────────────
        narrative = call_llm(combo, salary)

        # ── Step 3: Assemble the final record for precomputed_salaries ────
        country_code = inputs["company_location"]
        final_record = {
            "job_category": inputs["job_category"],
            "experience_level": inputs["experience_level"],
            "company_size": inputs["company_size"],
            "employment_type": inputs["employment_type"],
            "is_same_country": inputs["is_same_country"],
            "country_code": country_code,
            "region": COUNTRY_REGION.get(country_code, "Other"),
            "location_tier": inputs["location_tier"],
            "remote_ratio": inputs["remote_ratio"],
            "predicted_salary": salary,
            "narrative": narrative,
            "chart_data": None,
        }

        # ── Step 4: Persist to Supabase ───────────────────────────────────
        if push:
            inserted = push_to_supabase(final_record)
            if not inserted:
                logger.warning(
                    "SKIP DB [%d/%d]: Supabase insert failed for %s",
                    i, total, job_title,
                )
                fail_count += 1
                continue

        success_count += 1
        status = "OK (no DB)" if not push else "OK"
        print(f"  {status}  {inputs['job_category']} ({exp}, {country_code}) → ${salary:,.0f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Total: {total} | Success: {success_count} | Failed: {fail_count}")
    print(f"{'─' * 60}\n")


def run_batch_predict(combinations: list[dict], output_path: str = "data/predictions.csv") -> None:
    """
    Runs predictions for all combinations and saves results to a CSV file.
    No LLM narrative, no Supabase — predictions only, for EDA.
    """
    total = len(combinations)
    print(f"\nRunning BATCH PREDICT for {total} combination(s) → {output_path}\n")

    csv_columns = [
        "job_category", "experience_level", "company_size", "employment_type",
        "is_same_country", "country_code", "region", "location_tier",
        "remote_ratio", "predicted_salary",
    ]

    success_count = 0
    fail_count = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        for i, combo in enumerate(combinations, start=1):
            api_resp = call_fastapi(combo)
            if api_resp is None:
                logger.warning("SKIP [%d/%d]: prediction failed", i, total)
                fail_count += 1
                continue

            salary = api_resp["predicted_salary_usd"]
            inputs = api_resp["inputs"]
            country_code = inputs["company_location"]

            row = {
                "job_category": inputs["job_category"],
                "experience_level": inputs["experience_level"],
                "company_size": inputs["company_size"],
                "employment_type": inputs["employment_type"],
                "is_same_country": inputs["is_same_country"],
                "country_code": country_code,
                "region": COUNTRY_REGION.get(country_code, "Other"),
                "location_tier": inputs["location_tier"],
                "remote_ratio": inputs["remote_ratio"],
                "predicted_salary": salary,
            }
            writer.writerow(row)
            success_count += 1

            if i % 200 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({success_count} OK, {fail_count} failed)")

    print(f"\n{'─' * 60}")
    print(f"  Total: {total} | Success: {success_count} | Failed: {fail_count}")
    print(f"  Saved to: {output_path}")
    print(f"{'─' * 60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _resolve_combinations(args) -> list[dict]:
    """Pick the right combination source based on CLI flags."""
    if args.country:
        combos = generate_combinations(country_filter=args.country)
        if args.limit:
            combos = combos[: args.limit]
        return combos
    if args.generate:
        combos = generate_combinations()
        if args.limit:
            combos = combos[: args.limit]
        return combos
    return SAMPLE_COMBINATIONS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salary Prediction Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Steps:\n"
            "  predict       — FastAPI predictions only, print to console\n"
            "  batch_csv     — predictions only, save to CSV (no LLM, no DB)\n"
            "  analyze       — predict + LLM narrative (no DB)\n"
            "  push_db       — full pipeline including Supabase insert\n"
            "  full_pipeline — alias for push_db\n"
        ),
    )
    parser.add_argument(
        "--step",
        choices=["predict", "batch_csv", "analyze", "push_db", "full_pipeline"],
        required=True,
        help="Pipeline step to execute",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Use the generator instead of sample combinations",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="ISO-2 country code to generate combinations for (e.g. US)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of generated combinations to process",
    )
    args = parser.parse_args()
    combos = _resolve_combinations(args)

    if args.step == "predict":
        run_predict(combos)
    elif args.step == "batch_csv":
        run_batch_predict(combos)
    elif args.step == "analyze":
        run_analyze(combos)
    elif args.step in ("push_db", "full_pipeline"):
        run_full_pipeline(combos, push=True)


if __name__ == "__main__":
    main()
