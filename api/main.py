"""
Salary Prediction API
─────────────────────
GET /predict  → returns predicted salary in USD given job details
GET /valid-inputs → returns all accepted values for categorical params
GET /health   → liveness probe
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Artifact paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "randomforest_v1.joblib"
MAPPINGS_PATH = BASE_DIR / "model" / "feature_mappings.joblib"

# ── Load artifacts at startup (once, at import time) ──────────────────────────
# If either file is missing the API still starts; requests will receive a 503
# until the files are restored and the server is restarted.
_model = None
_mappings = None

try:
    _model = joblib.load(MODEL_PATH)
    _mappings = joblib.load(MAPPINGS_PATH)
    logger.info(
        "Artifacts loaded — model: %s | features: %d | mapping keys: %s",
        type(_model).__name__,
        getattr(_model, "n_features_in_", "?"),
        list(_mappings.keys()),
    )
except FileNotFoundError as exc:
    logger.error("Artifact file not found at startup: %s", exc)
except Exception as exc:
    logger.error("Unexpected error loading artifacts: %s", exc)


# ── Enums — FastAPI auto-validates and documents these ─────────────────────────
class ExperienceLevel(str, Enum):
    EN = "EN"  # Entry-level / Junior
    MI = "MI"  # Mid-level
    SE = "SE"  # Senior
    EX = "EX"  # Executive / Director


class EmploymentType(str, Enum):
    FT = "FT"  # Full-time
    PT = "PT"  # Part-time
    CT = "CT"  # Contract
    FL = "FL"  # Freelance


class CompanySize(str, Enum):
    S = "S"  # < 50 employees
    M = "M"  # 50–250 employees
    L = "L"  # > 250 employees


# ── Preprocessing ──────────────────────────────────────────────────────────────
def build_feature_vector(
    job_title: str,
    experience_level: ExperienceLevel,
    employment_type: EmploymentType,
    company_location: str,
    employee_residence: str,
    company_size: CompanySize,
    work_year: int,
    remote_ratio: int,
) -> pd.DataFrame:
    """
    Translate raw human-readable inputs into the 16-column DataFrame
    that decision_tree_v2 was trained on.

    Raises ValueError with a descriptive message on any unknown categorical
    value so callers can surface a clean 400 to the client.
    """
    job_map: dict = _mappings["job_category_map"]
    tier_map: dict = _mappings["country_tier_map"]
    exp_map: dict = _mappings["experience_level_map"]
    size_map: dict = _mappings["company_size_map"]

    # Normalise country codes to uppercase
    company_location = company_location.upper()
    employee_residence = employee_residence.upper()

    # -- Validate and map categorical values -----------------------------------
    if job_title not in job_map:
        raise ValueError(
            f"Unknown job_title: '{job_title}'. "
            f"Valid titles: {sorted(job_map.keys())}"
        )
    if company_location not in tier_map:
        raise ValueError(
            f"Unknown company_location: '{company_location}'. "
            f"Valid ISO codes: {sorted(tier_map.keys())}"
        )
    if employee_residence not in tier_map:
        raise ValueError(
            f"Unknown employee_residence: '{employee_residence}'. "
            f"Valid ISO codes: {sorted(tier_map.keys())}"
        )

    job_category: str = job_map[job_title]
    location_tier: str = tier_map[company_location]
    is_same_country: int = int(company_location == employee_residence)

    # Ordinal encodings match the training notebook
    exp_encoded: int = exp_map[experience_level.value]
    size_encoded: int = size_map[company_size.value]

    # -- Build a single-row DataFrame with the raw categorical columns ---------
    row = {
        "work_year": work_year,
        "experience_level": exp_encoded,
        "remote_ratio": remote_ratio,
        "company_size": size_encoded,
        "is_same_country": is_same_country,
        "job_category": job_category,
        "employment_type": employment_type.value,
        "location_tier": location_tier,
    }
    df = pd.DataFrame([row])

    # -- One-hot encode the nominal columns, mirroring training ----------------
    df = pd.get_dummies(df, columns=["job_category", "employment_type", "location_tier"])

    # -- Align columns to the exact set the model expects ----------------------
    # Any dummy column absent in this single row (e.g. unseen category combo)
    # is filled with 0; extra columns are dropped.
    expected_features: list[str] = list(_model.feature_names_in_)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features].astype(float)

    return df


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Salary Prediction API",
    description=(
        "Predicts data-science job salaries (USD) using a Decision Tree "
        "trained on the Kaggle Data Science Job Salaries dataset."
    ),
    version="1.0.0",
)


# ── Health probe ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    """Liveness check. Also reports whether model artifacts are loaded."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "mappings_loaded": _mappings is not None,
    }


# ── Valid inputs catalogue ─────────────────────────────────────────────────────
@app.get("/valid-inputs", tags=["meta"])
def valid_inputs():
    """
    Returns all accepted values for every categorical parameter.
    Useful for the orchestrator script that generates the input combinations.
    """
    if _mappings is None:
        raise HTTPException(
            status_code=503,
            detail="Mappings not loaded. Check server logs.",
        )
    return {
        "job_titles": sorted(_mappings["job_category_map"].keys()),
        "experience_levels": list(_mappings["experience_level_map"].keys()),
        "employment_types": ["FT", "PT", "CT", "FL"],
        "country_codes": sorted(_mappings["country_tier_map"].keys()),
        "company_sizes": list(_mappings["company_size_map"].keys()),
        "remote_ratios": [0, 50, 100],
    }


# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.get("/predict", tags=["prediction"])
def predict(
    job_title: Annotated[
        str,
        Query(description="Specific job title, e.g. 'Data Scientist'"),
    ],
    experience_level: Annotated[
        ExperienceLevel,
        Query(description="EN = Entry, MI = Mid, SE = Senior, EX = Executive"),
    ],
    employment_type: Annotated[
        EmploymentType,
        Query(description="FT = Full-time, PT = Part-time, CT = Contract, FL = Freelance"),
    ],
    company_location: Annotated[
        str,
        Query(min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code, e.g. 'US'"),
    ],
    employee_residence: Annotated[
        str,
        Query(min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code, e.g. 'US'"),
    ],
    company_size: Annotated[
        CompanySize,
        Query(description="S = <50 employees, M = 50–250, L = >250"),
    ],
    work_year: Annotated[
        int,
        Query(ge=2020, le=2025, description="Calendar year of employment (2020–2025)"),
    ],
    remote_ratio: Annotated[
        int,
        Query(ge=0, le=100, description="Percentage of remote work: 0, 50, or 100"),
    ],
):
    """
    Predict the expected salary in USD for a data-science role.

    **Error contract**
    | Code | Cause |
    |------|-------|
    | 400  | Unknown categorical value (job title, country code) or invalid remote_ratio |
    | 422  | Missing / wrong-type query parameter (handled by FastAPI/Pydantic) |
    | 503  | Model artifacts not loaded (server-side file issue) |
    | 500  | Unexpected model inference failure |
    """

    # 1. Artifact availability ─────────────────────────────────────────────────
    if _model is None or _mappings is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts unavailable. Please check server logs.",
        )

    # 2. Business-level value validation (not expressible as a type) ───────────
    if remote_ratio not in {0, 50, 100}:
        raise HTTPException(
            status_code=400,
            detail=f"remote_ratio must be 0, 50, or 100. Received: {remote_ratio}.",
        )

    # 3. Preprocessing ─────────────────────────────────────────────────────────
    try:
        X = build_feature_vector(
            job_title=job_title,
            experience_level=experience_level,
            employment_type=employment_type,
            company_location=company_location,
            employee_residence=employee_residence,
            company_size=company_size,
            work_year=work_year,
            remote_ratio=remote_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 4. Inference ─────────────────────────────────────────────────────────────
    try:
        salary_usd = float(_model.predict(X)[0])
    except Exception as exc:
        logger.error("Model inference failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal model error.",
        )

    # 5. Response ──────────────────────────────────────────────────────────────
    return {
        "predicted_salary_usd": round(salary_usd, 2),
        "inputs": {
            "job_title": job_title,
            "job_category": _mappings["job_category_map"][job_title],
            "experience_level": experience_level.value,
            "employment_type": employment_type.value,
            "company_location": company_location.upper(),
            "employee_residence": employee_residence.upper(),
            "location_tier": _mappings["country_tier_map"][company_location.upper()],
            "company_size": company_size.value,
            "work_year": work_year,
            "remote_ratio": remote_ratio,
            "is_same_country": int(company_location.upper() == employee_residence.upper()),
        },
    }
