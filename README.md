# **Salario**

**Premium End-to-End ML Salary Analyst with "Glass Box" Explainable AI**

> From raw data to AI-narrated salary intelligence — pre-generated, instantly served, fully explainable.

<!-- Replace with an actual screenshot of the Market Landscape tab -->
![Market Landscape](docs/screenshot-market-landscape.png)

---

## The "Glass Box" Philosophy

Traditional salary predictors are **black boxes** — they output a number and expect you to trust it. Salario takes the opposite approach: every prediction is accompanied by a human-readable explanation of *why* the model positioned you where it did, backed by evidence charts drawn from the original dataset.

This is achieved through a **Two-Layer Narrative Architecture**:

| Layer | Scope | Engine | Purpose |
|-------|-------|--------|---------|
| **Macro** — Global Insights | Market-wide trends | Gemini 2.5 Flash (Cloud) | Executive summary, chart captions, and a Data Transparency note explaining geographic skew |
| **Micro** — Per-Prediction | Individual salary profiles | Ollama / Llama 3.2 (Local) | Personalized narratives with feature attribution ("Your salary is *above* the Data Scientist median, primarily driven by…") |

The LLMs **never calculate** — all statistics are pre-computed by Pandas. The models only *interpret* and *narrate*, ensuring numerical accuracy while delivering fluid, economist-style prose.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE (offline)                        │
│                                                             │
│  ds_salaries.csv                                            │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ FastAPI   │───▶│ Orchestrator │───▶│ Supabase          │  │
│  │ /predict  │    │ 1,000+ combos│    │ precomputed_salaries│ │
│  └──────────┘    └──────┬───────┘    │ global_insights    │  │
│                         │            └───────────────────┘  │
│              ┌──────────┴──────────┐                        │
│              ▼                     ▼                        │
│     ┌──────────────┐    ┌────────────────┐                  │
│     │ Ollama Local │    │ Gemini Cloud   │                  │
│     │ Llama 3.2    │    │ 2.5 Flash      │                  │
│     │ micro-narr.  │    │ exec. summary  │                  │
│     └──────────────┘    └────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (online)                          │
│                                                             │
│  Streamlit Dashboard ──▶ Supabase (read-only)               │
│  • Market Landscape tab (Gemini narratives + Plotly charts)  │
│  • Salary Predictor tab (instant lookup + Glass Box engine)  │
│  • Stripe-inspired CSS with Inter typeface                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| **FastAPI Server** | `api/main.py` | Hosts the Random Forest model, exposes `/predict`, `/valid-inputs`, `/health` |
| **Orchestrator** | `pipeline/orchestrator.py` | Central entry point — batch-generates predictions, triggers LLM narratives, upserts to Supabase |
| **Generator** | `pipeline/generator.py` | Produces all valid filter combinations (job category × experience × country × size × remote) |
| **LLM Analyst** | `pipeline/llm_analyst.py` | Crafts per-prediction micro-narratives via Ollama (Llama 3.2) |
| **Global Analyst** | `pipeline/global_analyst.py` | Aggregates market stats with Pandas, sends to Gemini for executive summary |
| **Dashboard** | `streamlit/dashboard.py` | Stripe-themed Streamlit UI — reads exclusively from Supabase |

### ML Model

- **Algorithm**: Random Forest Regressor
- **Tuning**: GridSearchCV with cross-validation
- **Features**: `job_category`, `experience_level`, `company_size`, `remote_ratio`, `country_code`, `employment_type`, `is_same_country`
- **Target**: `salary_in_usd`
- **Artifact**: `model/randomforest_v1.joblib`

---

## Setup & Deployment

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) running locally with `llama3.2` pulled
- Supabase project with `precomputed_salaries` and `global_insights` tables
- Gemini API key (Google AI Studio)

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example pipeline/.env
# Edit pipeline/.env with your actual credentials
```

### 3. Start the FastAPI Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API documentation is auto-generated:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 4. Run the Pipeline

```bash
# Generate all predictions and micro-narratives
python -m pipeline.orchestrator --step all

# Generate only the global executive summary (Gemini)
python -m pipeline.orchestrator --step global_summary
```

### 5. Launch the Dashboard

```bash
streamlit run streamlit/dashboard.py
```

---

## Project Structure

```
salary-prediction-app/
├── api/
│   └── main.py                 # FastAPI prediction server
├── data/
│   └── ds_salaries.csv         # Original 607-row dataset
├── model/
│   ├── randomforest_v1.joblib  # Trained model artifact
│   └── feature_mappings.joblib # job_category & location_tier maps
├── notebooks/
│   └── ...                     # EDA & experimentation
├── pipeline/
│   ├── orchestrator.py         # Central pipeline entry point
│   ├── generator.py            # Filter combination generator
│   ├── llm_analyst.py          # Ollama micro-narrative engine
│   └── global_analyst.py       # Pandas stats + Gemini summary
├── streamlit/
│   └── dashboard.py            # Salario dashboard (Stripe UI)
├── .env.example                # Required environment variables
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

---

## License

This project was developed as part of the AIE program curriculum.
