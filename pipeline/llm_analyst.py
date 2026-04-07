"""
LLM Analyst Module
──────────────────
Sends a salary prediction record to a local Ollama instance (llama3.2)
and returns a short narrative insight about the predicted salary.

Failure modes handled gracefully:
- Ollama not running            → ConnectionError       → fallback narrative
- Ollama request too slow       → Timeout               → fallback narrative
- LLM returns non-JSON text     → JSONDecodeError       → fallback narrative
- LLM JSON missing 'narrative'  → KeyError              → fallback narrative
- JSON valid but wrong shape    → ValidationError       → fallback narrative
- Any other unexpected error    → Exception             → fallback narrative
"""

import json
import logging

import requests
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 60  # seconds — LLM inference can be slow on CPU

FALLBACK_NARRATIVE = (
    "Salary prediction generated successfully. "
    "LLM narrative unavailable at this time."
)

# Key names llama3.2 has been observed to use instead of "narrative"
_CANDIDATE_KEYS = ("narrative", "response", "insight", "analysis", "text", "content", "message")


# ── Output schema ──────────────────────────────────────────────────────────────
class LLMResponse(BaseModel):
    """
    The exact JSON structure we expect the LLM to return.
    Pydantic enforces this — any deviation triggers a ValidationError,
    which we catch and replace with the fallback narrative.
    """

    narrative: str


def _extract_narrative(parsed: dict) -> str:
    """
    Pull a narrative string out of the LLM's parsed JSON dict.

    llama3.2 sometimes ignores the requested key name and returns things like
    {"response": "..."} or {"insight": "..."} instead of {"narrative": "..."}.
    This function tries a list of common candidate keys first, then falls back
    to the first string-valued entry it finds anywhere in the dict.

    Raises ValueError if no usable string value can be found at all.
    """
    # 1. Try each known candidate key in priority order
    for key in _CANDIDATE_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            if key != "narrative":
                logger.warning("LLM used key '%s' instead of 'narrative' — extracted anyway.", key)
            return value.strip()

    # 2. Fallback: scan all top-level values for the first non-empty string
    for key, value in parsed.items():
        if isinstance(value, str) and value.strip():
            logger.warning("LLM used unexpected key '%s' — extracted first string value.", key)
            return value.strip()

    # 3. Nothing usable found
    raise ValueError(f"No string value found in LLM JSON. Keys present: {list(parsed.keys())}")


# ── Core function ──────────────────────────────────────────────────────────────
def generate_micro_narrative(
    payload: dict,
    predicted_salary: float,
    global_medians: dict,
) -> str:
    """
    Calls the local Ollama LLM to generate a micro-narrative for a salary
    prediction record.

    Parameters
    ----------
    payload : dict
        The original input combination sent to the FastAPI (job_title,
        experience_level, employment_type, company_location,
        employee_residence, company_size, work_year, remote_ratio).
    predicted_salary : float
        The predicted salary in USD returned by the Decision Tree model.
    global_medians : dict
        Reference statistics (e.g. overall_median, median_by_experience)
        used to give the LLM context for comparison.

    Returns
    -------
    str
        A narrative string. Never raises — returns FALLBACK_NARRATIVE on
        any error so the orchestrator loop can always continue.
    """
    job_title = payload.get("job_title", "Unknown")
    experience_level = payload.get("experience_level", "Unknown")
    employment_type = payload.get("employment_type", "Unknown")
    company_location = payload.get("company_location", "Unknown")
    company_size = payload.get("company_size", "Unknown")
    remote_ratio = payload.get("remote_ratio", "Unknown")
    work_year = payload.get("work_year", "Unknown")

    overall_median = global_medians.get("overall_median", 0)
    delta = predicted_salary - overall_median
    delta_pct = (delta / overall_median * 100) if overall_median else 0
    direction = "above" if delta >= 0 else "below"

    # ── System prompt ─────────────────────────────────────────────────────────
    # We tell the model exactly what JSON shape to produce.
    # The "format": "json" field in the request body enforces JSON mode on
    # Ollama's side, but we still validate the shape ourselves via Pydantic.
    system_prompt = (
        "You are a concise data analyst. "
        "Given a salary prediction record, write a 2-3 sentence narrative that "
        "explains what the predicted salary means in context — compare it to the "
        "market median, comment on the role/experience/location combination, and "
        "give one actionable insight. "
        "Respond ONLY with a valid JSON object using exactly this schema:\n"
        '{"narrative": "<your 2-3 sentence insight here>"}\n'
        "No markdown, no extra keys, no text outside the JSON object."
    )

    user_message = (
        f"Job Title: {job_title}\n"
        f"Experience Level: {experience_level}\n"
        f"Employment Type: {employment_type}\n"
        f"Company Location: {company_location}\n"
        f"Company Size: {company_size}\n"
        f"Remote Ratio: {remote_ratio}%\n"
        f"Work Year: {work_year}\n"
        f"Predicted Salary: ${predicted_salary:,.0f} USD\n"
        f"Market Median Salary: ${overall_median:,.0f} USD\n"
        f"Delta: ${abs(delta):,.0f} ({abs(delta_pct):.1f}%) {direction} median\n"
    )

    request_body = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_message,
        "format": "json",   # Ollama JSON mode — forces the model to output valid JSON
        "stream": False,    # We want the full response in one shot, not streamed tokens
    }

    # ── Network call + error handling ─────────────────────────────────────────
    try:
        response = requests.post(OLLAMA_URL, json=request_body, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()

        # Ollama wraps the model's output in {"response": "<text>", ...}
        raw_text: str = response.json().get("response", "")

        # Parse the LLM's output as JSON, then extract the narrative string.
        # _extract_narrative handles the case where llama3.2 uses a different
        # key name (e.g. "response", "insight") instead of "narrative".
        # LLMResponse then validates that the extracted value is a non-empty string.
        parsed: dict = json.loads(raw_text)
        narrative_text: str = _extract_narrative(parsed)
        validated = LLMResponse(narrative=narrative_text)
        return validated.narrative

    except requests.exceptions.ConnectionError:
        logger.error(
            "Ollama is not running. Could not connect to %s. "
            "Start it with: ollama serve",
            OLLAMA_URL,
        )
        return FALLBACK_NARRATIVE

    except requests.exceptions.Timeout:
        logger.error(
            "Ollama request timed out after %ds for job: %s",
            OLLAMA_TIMEOUT,
            job_title,
        )
        return FALLBACK_NARRATIVE

    except requests.exceptions.HTTPError as exc:
        logger.error(
            "Ollama HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        return FALLBACK_NARRATIVE

    except json.JSONDecodeError as exc:
        logger.error(
            "LLM returned non-JSON text for '%s': %s. Raw: %.200s",
            job_title,
            exc,
            raw_text if "raw_text" in dir() else "(no response captured)",
        )
        return FALLBACK_NARRATIVE

    except (KeyError, TypeError) as exc:
        logger.error(
            "Unexpected structure in LLM response for '%s': %s",
            job_title,
            exc,
        )
        return FALLBACK_NARRATIVE

    except ValidationError as exc:
        logger.error(
            "LLM JSON failed Pydantic validation for '%s': %s",
            job_title,
            exc,
        )
        return FALLBACK_NARRATIVE

    except Exception as exc:
        logger.error(
            "Unexpected error calling Ollama for '%s': %s",
            job_title,
            exc,
            exc_info=True,
        )
        return FALLBACK_NARRATIVE


# ── Manual smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    dummy_payload = {
        "job_title": "Data Scientist",
        "experience_level": "SE",
        "employment_type": "FT",
        "company_location": "US",
        "employee_residence": "US",
        "company_size": "L",
        "work_year": 2024,
        "remote_ratio": 100,
    }

    dummy_medians = {
        "overall_median": 110_000,
        "median_senior": 140_000,
    }

    predicted = 155_000.0

    print("Calling Ollama…")
    narrative = generate_micro_narrative(dummy_payload, predicted, dummy_medians)
    print("\n── Narrative ──────────────────────────────────────────────────────")
    print(narrative)
    print("───────────────────────────────────────────────────────────────────")
