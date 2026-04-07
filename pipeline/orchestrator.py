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
import logging
import os

import requests
from supabase import create_client, Client

from pipeline.llm_analyst import generate_micro_narrative

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

def call_fastapi(payload: dict) -> float | None:
    """
    Calls GET /predict on the local FastAPI server.

    Parameters
    ----------
    payload : dict
        All 8 query parameters the endpoint expects.

    Returns
    -------
    float | None
        Predicted salary in USD, or None on any failure.
        None signals the caller to skip this record — it does NOT raise.
    """
    try:
        response = requests.get(FASTAPI_URL, params=payload, timeout=10)
        response.raise_for_status()
        return float(response.json()["predicted_salary_usd"])

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
    Inserts one prediction record into the Supabase `predictions` table.

    Parameters
    ----------
    final_record : dict
        The fully assembled record ready for persistence.

    Returns
    -------
    bool
        True on success, False on any failure.
        False signals the caller to skip this record — it does NOT raise.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error(
            "Supabase credentials not set. "
            "Export SUPABASE_URL and SUPABASE_KEY environment variables."
        )
        return False

    try:
        client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        client.table("predictions").insert(final_record).execute()
        logger.info(
            "Supabase insert OK: '%s' (%s) → $%s",
            final_record.get("job_title"),
            final_record.get("experience_level"),
            f"{final_record.get('predicted_salary_usd', 0):,.0f}",
        )
        return True

    except Exception as exc:
        logger.error(
            "Supabase insert failed for '%s': %s",
            final_record.get("job_title"),
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
        prediction = call_fastapi(combo)
        if prediction is None:
            print(f"  SKIP  {combo.get('job_title')} ({combo.get('experience_level')}) — prediction failed")
            continue
        print(
            f"  OK    {combo['job_title']} ({combo['experience_level']}, "
            f"{combo['company_location']}) → ${prediction:,.0f}"
        )


def run_analyze(combinations: list[dict]) -> None:
    """
    Runs prediction + LLM analysis — calls FastAPI then Ollama for each
    combination and prints the narrative. No database write.
    """
    print(f"\nRunning ANALYZE step for {len(combinations)} combination(s)…\n")
    for combo in combinations:
        prediction = call_fastapi(combo)
        if prediction is None:
            print(f"  SKIP  {combo.get('job_title')} — prediction failed, skipping LLM\n")
            continue

        narrative = call_llm(combo, prediction)
        print(
            f"  [{combo['job_title']} | {combo['experience_level']} | "
            f"{combo['company_location']}]\n"
            f"  Salary: ${prediction:,.0f}\n"
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
        prediction = call_fastapi(combo)
        if prediction is None:
            logger.warning("SKIP [%d/%d]: prediction failed for %s", i, total, job_title)
            fail_count += 1
            continue

        # ── Step 2: LLM narrative (never None) ────────────────────────────
        narrative = call_llm(combo, prediction)

        # ── Step 3: Assemble the final record ─────────────────────────────
        final_record = {
            "job_title": job_title,
            "experience_level": exp,
            "employment_type": combo.get("employment_type"),
            "company_location": loc,
            "employee_residence": combo.get("employee_residence"),
            "company_size": combo.get("company_size"),
            "work_year": combo.get("work_year"),
            "remote_ratio": combo.get("remote_ratio"),
            "predicted_salary_usd": prediction,
            "narrative": narrative,
        }

        # ── Step 4: Persist to Supabase ───────────────────────────────────
        if push:
            inserted = push_to_supabase(final_record)
            if not inserted:
                logger.warning(
                    "SKIP DB [%d/%d]: Supabase insert failed for %s",
                    i, total, job_title,
                )
                # Count as partial success — prediction and narrative are fine,
                # only the DB write failed.
                fail_count += 1
                continue

        success_count += 1
        status = "OK (no DB)" if not push else "OK"
        print(f"  {status}  {job_title} ({exp}, {loc}) → ${prediction:,.0f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Total: {total} | Success: {success_count} | Failed: {fail_count}")
    print(f"{'─' * 60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salary Prediction Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Steps:\n"
            "  predict       — FastAPI predictions only (no LLM, no DB)\n"
            "  analyze       — predict + LLM narrative (no DB)\n"
            "  push_db       — full pipeline including Supabase insert\n"
            "  full_pipeline — alias for push_db\n"
        ),
    )
    parser.add_argument(
        "--step",
        choices=["predict", "analyze", "push_db", "full_pipeline"],
        required=True,
        help="Pipeline step to execute",
    )
    args = parser.parse_args()

    if args.step == "predict":
        run_predict(SAMPLE_COMBINATIONS)
    elif args.step == "analyze":
        run_analyze(SAMPLE_COMBINATIONS)
    elif args.step in ("push_db", "full_pipeline"):
        run_full_pipeline(SAMPLE_COMBINATIONS, push=True)


if __name__ == "__main__":
    main()
