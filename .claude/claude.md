# Project Memory — Salary Prediction Application

## Project Overview Context

**End-to-End ML Pipeline: From Data to Deployment.**

Build a salary prediction app that:
1. Takes data-science job details as inputs.
2. Predicts salary via a trained Random Forest model served through a FastAPI.
3. Passes results to a local LLM (Ollama / llama3.2) that generates narrative insights.
4. Persists everything to Supabase.
5. Displays results on a live Streamlit dashboard (reads from Supabase only).

The **deployed FastAPI** is a separate deliverable — same model, independently accessible, not part of the local pipeline flow.

---

## Architecture (Pre-Generation Pipeline)

```
Generator  →  FastAPI  →  Ollama LLM  →  Supabase  →  Streamlit
(combos)     (predict)   (narrative)    (persist)     (consume)
```

**Key principle:** This is a *pre-generation* architecture. The orchestrator runs offline,
generates all prediction + narrative records, and stores them in Supabase. The Streamlit
dashboard is a pure read layer — it never calls the model or Ollama directly.

### Three-Layer Data Flow

| Layer              | Role                   | Key Constraint                                    |
|--------------------|------------------------|---------------------------------------------------|
| **FastAPI (Data)** | ML prediction          | Expects `job_title`, `company_location`, `employee_residence`, raw inputs |
| **Generator (Orchestrator)** | Bridge / mapper | Iterates countries, maps to categories, formats payloads for API and DB |
| **Supabase (Product)** | Streamlit storage  | Flattened schema with `job_category`, `country_code`, `region`, `location_tier` |

---

## Architectural Constraints (Strict Rules)

### A. The Geographic Mapping Rule

The generator **MUST** iterate over a `COUNTRY_MAP` dictionary that maps **every country code**
to its region and tier:
```python
"US": {"region": "North America", "tier": "High_Tier"}
"FR": {"region": "Europe",        "tier": "Mid_Tier"}
"IN": {"region": "Asia",          "tier": "Low_Tier"}
```

**It must NEVER iterate over tiers and sample representative countries.** Every supported
country must have a dedicated row in Supabase so the Streamlit dropdown can filter by
individual country.

### B. The International/Dummy API Rule

When generating `is_same_country = 0` records:
- The generator must pass a `employee_residence` that **differs** from `company_location`
  to trigger the API's internal `is_same_country == 0` calculation.
- A fixed dummy residence is used (e.g. `"GB"` when company is `"US"`, `"US"` otherwise).
  The API validates `employee_residence` against its `country_tier_map`, so invalid codes
  like `"XX"` will be rejected with a 400.

The generator must also send dummy one-hot columns to prevent the joblib preprocessor from
failing on missing columns:
```python
"employment_type_FT": 1
"employment_type_CT": 0
"employment_type_FL": 0
"employment_type_PT": 0
```

### C. The Job Title vs Category Map Rule

- The **API** requires `job_title` as input (e.g. `"Data Scientist"`).
- The **database** stores `job_category` (the engineered feature).
- The generator uses `CATEGORY_REPRESENTATIVE` to map each of the 4 categories to a
  valid job title for the API call, then stores the category in the Supabase record.
- The 4 categories: `Data Analyst`, `Data Engineer`, `Data Scientist`, `Machine Learning Engineer`.

### D. Employment Pruning

Only `employment_type = "FT"` (Full-Time) is generated. The dummy one-hot columns
(CT=0, FL=0, PT=0) ensure the ML model still receives a complete feature vector.

### E. Business Validation Rules

- **EX + S** filtered out: Executive roles at Small companies are rare/noisy in the dataset.
- **Cross-border + remote=0** filtered out: `is_same_country=0` with `remote_ratio=0` is illogical.

---

## Repository Structure

```
salary-prediction-app/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI — GET /predict, /valid-inputs, /health
├── model/
│   ├── randomforest_v1.joblib
│   └── feature_mappings.joblib
├── pipeline/
│   ├── __init__.py
│   ├── generator.py         # Input combination generator (per-country)
│   ├── llm_analyst.py       # Ollama interface (micro + global prompts)
│   └── orchestrator.py      # Pipeline brain (CLI)
├── data/
│   └── ds_salaries.csv      # Original dataset
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Model & API Facts

- **Model:** `RandomForestRegressor` — `model/randomforest_v1.joblib`
- **Preprocessor:** ordinal + one-hot encoding in `api/main.py` via `feature_mappings.joblib`
- **API base URL (local):** `http://localhost:8000`
- **Prediction endpoint:** `GET /predict` (all params as query strings)
- **Valid inputs endpoint:** `GET /valid-inputs` (returns all accepted categorical values)

### Accepted Input Values

| Parameter          | Valid Values |
|--------------------|--------------|
| experience_level   | EN, MI, SE, EX |
| employment_type    | FT, PT, CT, FL |
| company_size       | S, M, L |
| remote_ratio       | 0, 50, 100 |
| work_year          | 2020–2025 |
| job_title          | 50 titles (see `/valid-inputs`) |
| company_location   | 50 ISO-2 country codes (see `/valid-inputs`) |
| employee_residence | same set as company_location |

---

## Supabase Schema

### Table: `precomputed_salaries`

| Column           | Type    | Notes                                  |
|------------------|---------|----------------------------------------|
| id               | uuid    | PK                                     |
| created_at       | tz      |                                        |
| job_category     | text    | e.g., Data Scientist                   |
| experience_level | text    | EN/MI/SE/EX                            |
| company_size     | text    | S/M/L                                  |
| employment_type  | text    | FT (Hardcoded for generation)          |
| is_same_country  | int     | 1 or 0                                 |
| country_code     | text    | e.g., US, FR                           |
| region           | text    | e.g., North America, Europe            |
| location_tier    | text    | Hidden ML mapping                      |
| remote_ratio     | int     | 0 / 50 / 100                           |
| predicted_salary | numeric |                                        |
| narrative        | text    | LLM generated text                     |
| chart_data       | jsonb   | Optional JSON for Streamlit rendering  |

### Table: `global_insights`

| Column            | Type    | Notes                                  |
|-------------------|---------|----------------------------------------|
| id                | uuid    | PK                                     |
| created_at        | tz      |                                        |
| executive_summary | text    | 3-paragraph LLM global analysis        |

---

## Git Branching Strategy — Feature Branch Workflow

### Rules (Non-Negotiable)

1. **Never commit directly to `master`.** All development happens on short-lived feature branches.
2. **Branch naming prefixes:**
   - `feat/`  — new functionality
   - `fix/`   — bug fix on already-merged code
   - `docs/`  — documentation only
   - `chore/` — tooling, deps, config (no production code change)
3. **Test on the branch** before requesting merge approval.
4. **Wait for user approval** of test results before merging to `master`.
5. **Post-merge fixes:** Never reuse a merged branch — create a new `fix/` branch.

### Commit Message Format

```
<type>: <short imperative description>

# Examples
feat: implement LLM analyst module with strict JSON validation
fix: handle None return from FastAPI in orchestrator loop
docs: update memory with final architecture and fix generator logic
```

---

## Error Handling Best Practices

**Rule: "No unhandled errors."** (Assignment requirement — will be checked.)

### Validation Layer

- Rely on **Pydantic** (FastAPI) for type and enum validation.
- HTTP status codes: `400` (bad value), `422` (missing param), `503` (artifacts), `500` (inference).

### Network Call Layer

Wrap **every** external network call in `try/except` with specific exception types.
Return `None` on failure — never raise from network functions.

### LLM Layer

- Catch `json.JSONDecodeError` and `pydantic.ValidationError`.
- Always return a **fallback narrative string** (never `None`, never raise).

### Orchestrator Loop Resilience

- A failure on one record **must not** crash the loop.
- If any step returns `None`, log a warning and `continue`.
- Track `success_count` and `fail_count`; print summary at the end.

---

## Environment Variables (.env — never committed)

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<service_role key>
```

---

## Deliverables Checklist

- [x] `api/main.py` — FastAPI with GET /predict
- [x] `pipeline/llm_analyst.py` — Ollama interface with dual prompts (micro + global)
- [x] `pipeline/orchestrator.py` — CLI with `--step`, `--generate`, `--tier`, `--limit`
- [x] `pipeline/generator.py` — Input combination generator with validation
- [ ] Supabase `precomputed_salaries` table fully populated
- [ ] Supabase `global_insights` table populated
- [ ] `streamlit/dashboard.py` — reads from Supabase only
- [ ] Deployed FastAPI endpoint URL
- [ ] Live Streamlit app URL
- [ ] `README.md`
