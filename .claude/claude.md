# Project Memory — Salary Prediction Application

## Assignment Context

**Week 1 — AIE Program.** End-to-End ML Pipeline: From Data to Deployment.
**Deadline:** Friday 06:00 AM.

Build a salary prediction app that:
1. Takes data-science job details as inputs.
2. Predicts salary via a trained Decision Tree model served through a FastAPI.
3. Passes results to a local LLM (Ollama / llama3.2) that generates narrative insights + at least one visualization.
4. Persists everything to Supabase.
5. Displays results on a live Streamlit dashboard (reads from Supabase only).

The **deployed FastAPI** is a separate deliverable — same model, independently accessible, not part of the local pipeline flow.

---

## Architecture (Pre-Generation Pipeline)

```
Local FastAPI  →  Local LLM (Ollama)  →  Supabase  →  Streamlit Dashboard
   (predict)        (narrative)          (persist)       (consume)
```

**Key principle:** This is a *pre-generation* architecture. The orchestrator script runs offline,
generates all prediction + narrative records, and stores them in Supabase. The Streamlit dashboard
is a pure read layer — it never calls the model or Ollama directly.

The deployed FastAPI is a separate, standalone deliverable.

---

## Repository Structure

```
salary-prediction-app/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app — GET /predict, /valid-inputs, /health
├── model/
│   ├── decision_tree_v2.joblib
│   └── feature_mappings.joblib
├── pipeline/
│   ├── __init__.py
│   ├── llm_analyst.py       # Step 3 — Ollama interface
│   └── orchestrator.py      # Step 4 — pipeline brain (CLI)
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Model & API Facts

- **Model:** `DecisionTreeRegressor` — `model/decision_tree_v2.joblib`
- **Preprocessor:** ordinal + one-hot encoding baked into `api/main.py` via `feature_mappings.joblib`
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

## Git Branching Strategy — Feature Branch Workflow

### Rules (Non-Negotiable)

1. **Never commit directly to `master`.** All development happens on short-lived feature branches.
2. **Branch naming prefixes:**
   - `feat/`  — new functionality
   - `fix/`   — bug fix on already-merged code
   - `docs/`  — documentation only
   - `chore/` — tooling, deps, config (no production code change)
3. **Local integration testing:** Test end-to-end functionality *on the feature branch* before
   merging. Do not merge to `master` just to test.
4. **Post-merge fixes:** Never reuse a merged branch.
   - Bug found in merged code → branch off `master` to a new `fix/...` branch
   - Fix it, merge to `master`, then merge `master` back into any active feature branch

### Typical Workflow

```bash
git checkout master
git checkout -b feat/my-feature   # branch off master
# ... write code, commit ...
# run local integration tests on this branch
git checkout master
git merge feat/my-feature         # fast-forward or --no-ff
```

### Commit Message Format

```
<type>: <short imperative description>

# Examples
feat: implement LLM analyst module with strict JSON validation
fix: handle None return from FastAPI in orchestrator loop
docs: add project context, architecture, and coding standards
chore: add supabase and requests to requirements
```

---

## Error Handling Best Practices

**Rule: "No unhandled errors."** (Assignment requirement — will be checked.)

### Validation Layer

- Rely on **Pydantic** (FastAPI) for type and enum validation — never manually re-validate what
  Pydantic already covers.
- Use standard HTTP status codes:
  - `400` — bad categorical value (unknown job title, country code, invalid remote_ratio)
  - `422` — missing/wrong-type parameter (FastAPI/Pydantic handles automatically)
  - `503` — model artifacts not loaded
  - `500` — unexpected model inference failure

### Network Call Layer (orchestrator & LLM analyst)

Wrap **every** external network call in `try/except`:

```python
# Pattern for any external call
try:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()["key"]
except requests.exceptions.ConnectionError:
    logger.error("Service not running at %s", url)
    return None
except requests.exceptions.Timeout:
    logger.error("Request timed out")
    return None
except requests.exceptions.HTTPError as exc:
    logger.error("HTTP %s: %s", exc.response.status_code, exc.response.text)
    return None
except (KeyError, ValueError) as exc:
    logger.error("Unexpected response format: %s", exc)
    return None
```

### LLM Layer

- Catch `json.JSONDecodeError` — LLM may not honour `format: json`.
- Catch `pydantic.ValidationError` — LLM may return structurally wrong JSON.
- Always return a **fallback narrative string** (never `None`, never raise).

### Orchestrator Loop Resilience

- The pipeline runs 1 000+ iterations. A failure on one record **must not** crash the loop.
- Pattern: if any step returns `None`, log a warning and `continue` to the next record.
- Track `success_count` and `fail_count`; print a summary at the end.

---

## Supabase Schema (predictions table)

| Column                | Type    | Notes                          |
|-----------------------|---------|--------------------------------|
| id                    | uuid    | auto-generated PK              |
| job_title             | text    |                                |
| experience_level      | text    | EN / MI / SE / EX              |
| employment_type       | text    | FT / PT / CT / FL              |
| company_location      | text    | ISO-2                          |
| employee_residence    | text    | ISO-2                          |
| company_size          | text    | S / M / L                      |
| work_year             | integer |                                |
| remote_ratio          | integer | 0 / 50 / 100                   |
| predicted_salary_usd  | numeric |                                |
| narrative             | text    | LLM-generated insight          |
| created_at            | timestamptz | auto-set by Supabase       |

---

## Environment Variables (.env — never committed)

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<anon or service_role key>
```

---

## Deliverables Checklist

- [ ] `api/main.py` — FastAPI with GET /predict (validated inputs, clean error codes)
- [ ] `pipeline/llm_analyst.py` — Ollama interface with JSON mode + Pydantic validation
- [ ] `pipeline/orchestrator.py` — CLI (`--step predict|analyze|push_db|full_pipeline`)
- [ ] Supabase `predictions` table populated
- [ ] `streamlit/dashboard.py` — reads from Supabase only, handles missing data gracefully
- [ ] Deployed FastAPI endpoint URL
- [ ] Live Streamlit app URL
- [ ] `README.md` — well-presented project documentation
