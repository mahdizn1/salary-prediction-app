"""
Data Salary AI Dashboard — Stripe SaaS Aesthetic
─────────────────────────────────────────────────
Experimental UI overhaul: edge-to-edge top navigation, Inter typeface,
high-contrast boxed cards, no emojis, horizontal radio navbar.

Branch: experiment/stripe-ui — DO NOT MERGE without visual review.

Connects to:
  - Supabase (precomputed_salaries, global_insights tables)
  - Local CSV (data/ds_salaries.csv) for evidence charts
  - model/feature_mappings.joblib for job_title -> job_category mapping
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
import joblib

# ── Environment & Config ──────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / "pipeline" / ".env")
st.set_page_config(
    page_title="AI Salary Analyst",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Stripe-Style CSS Injection ────────────────────────────────────────────────
# Applied immediately after set_page_config per task spec.
# Principles: Inter font, edge-to-edge layout, high-contrast white cards on
# #f8fafc background, hidden Streamlit chrome, no emojis (material icons).
st.markdown("""
<style>
    /* ── Inter typeface (global override) ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], [class*="st-"],
    button, input, select, textarea, .stMarkdown, .stText {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── App background ───────────────────────────────────────────────── */
    .stApp {
        background-color: #f8fafc;
    }

    /* ── Edge-to-edge layout (remove default padding) ─────────────────── */
    .block-container {
        padding-top: 1rem !important;
        max-width: 95% !important;
    }

    /* ── Hide Streamlit chrome ────────────────────────────────────────── */
    [data-testid="stHeader"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

    /* ── Typography ───────────────────────────────────────────────────── */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        letter-spacing: -0.025em !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    h2 {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    p, li, span, div { color: #1e293b; }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
        font-size: 0.85rem !important;
    }

    /* ── Horizontal radio navbar ──────────────────────────────────────── */
    [data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 0 !important;
    }
    /* Hide radio circles */
    [data-testid="stRadio"] > div [data-testid="stMarkdownContainer"] {
        display: flex;
        align-items: center;
    }
    [data-testid="stRadio"] > div > label {
        padding: 8px 24px !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: #64748b !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        border: 1px solid transparent !important;
        background: transparent !important;
    }
    [data-testid="stRadio"] > div > label:hover {
        color: #1e293b !important;
        background: #f1f5f9 !important;
    }
    [data-testid="stRadio"] > div > label[data-checked="true"],
    [data-testid="stRadio"] > div > label:has(input:checked) {
        color: #ffffff !important;
        background: #6366f1 !important;
        font-weight: 600 !important;
        border-color: #6366f1 !important;
    }
    /* Hide the radio dot indicator */
    [data-testid="stRadio"] > div > label > div:first-child {
        display: none !important;
    }

    /* ── Top bar container ────────────────────────────────────────────── */
    .top-bar {
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 12px 0 12px 0;
        margin: -1rem -2rem 24px -2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ── High-contrast card styling ───────────────────────────────────── */
    .stripe-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                    0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }

    /* ── Metric cards ─────────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                    0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    div[data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* ── Form container ───────────────────────────────────────────────── */
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                    0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }

    /* ── Info box ─────────────────────────────────────────────────────── */
    .stInfo {
        border-left: 4px solid #6366f1;
        background: #eef2ff;
        border-radius: 0 8px 8px 0;
    }

    /* ── Primary button ───────────────────────────────────────────────── */
    button[kind="primaryFormSubmit"],
    .stButton > button[kind="primary"] {
        background: #6366f1 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 10px 24px !important;
        transition: all 0.15s ease !important;
    }
    button[kind="primaryFormSubmit"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: #4f46e5 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
    }

    /* ── Filter pills ─────────────────────────────────────────────────── */
    .filter-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 16px 0 24px 0;
        align-items: center;
    }
    .filter-pills .pill-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .filter-pills .pill {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #c7d2fe;
    }

    /* ── Dividers ─────────────────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid #e2e8f0 !important;
        margin: 20px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Human-readable label maps ────────────────────────────────────────────────
EXP_LABELS = {
    "EN": "Entry-Level", "MI": "Mid-Level", "SE": "Senior", "EX": "Executive",
}
EXP_ORDER = ["EN", "MI", "SE", "EX"]

SIZE_LABELS = {"S": "Small", "M": "Medium", "L": "Large"}
SIZE_ORDER = ["S", "M", "L"]

TIER_LABELS = {
    "Low_Tier": "Emerging Market",
    "Mid_Tier": "Developing Market",
    "High_Tier": "Premium Market",
}

REMOTE_LABELS = {0: "On-site", 50: "Hybrid", 100: "Fully Remote"}

COUNTRY_NAMES = {
    "US": "United States", "GB": "United Kingdom", "CA": "Canada",
    "DE": "Germany", "FR": "France", "ES": "Spain", "GR": "Greece",
    "NL": "Netherlands", "IN": "India", "PK": "Pakistan",
    "MX": "Mexico", "BR": "Brazil",
}

ACTIVE_COUNTRIES = ["US", "GB", "CA", "DE", "FR", "ES", "GR", "NL",
                    "IN", "PK", "MX", "BR"]

COUNTRY_TIER = {
    "US": "High_Tier", "GB": "High_Tier", "CA": "High_Tier", "DE": "High_Tier",
    "FR": "Mid_Tier", "ES": "Mid_Tier", "GR": "Mid_Tier", "NL": "Mid_Tier",
    "IN": "Low_Tier", "PK": "Low_Tier", "MX": "Low_Tier", "BR": "Low_Tier",
}

COUNTRY_REGION = {
    "US": "North America", "GB": "Europe", "CA": "North America",
    "DE": "Europe", "FR": "Europe", "ES": "Europe", "GR": "Europe",
    "NL": "Europe", "IN": "Asia", "PK": "Asia",
    "MX": "North America", "BR": "South America",
}

CATEGORY_MEDIANS = {
    "Data Analyst": 92_000,
    "Data Engineer": 111_888,
    "Data Scientist": 110_000,
    "Machine Learning Engineer": 81_872,
}
OVERALL_MEDIAN = 101_570

CATEGORY_THRESHOLDS = {
    "Data Analyst": (-60, -10, 14, 63),
    "Data Engineer": (-58, -12, 18, 79),
    "Data Scientist": (-65, -15, 19, 91),
    "Machine Learning Engineer": (-74, -15, 20, 153),
}
_DEFAULT_THRESHOLDS = (-64, -13, 18, 96)

_FULL_REGION_MAP = {
    "AE": "Middle East", "AS": "Asia", "AT": "Europe", "AU": "Oceania",
    "BE": "Europe", "BR": "South America", "CA": "North America",
    "CH": "Europe", "CL": "South America", "CN": "Asia",
    "CO": "South America", "CZ": "Europe", "DE": "Europe", "DK": "Europe",
    "DZ": "Africa", "EE": "Europe", "ES": "Europe", "FR": "Europe",
    "GB": "Europe", "GR": "Europe", "HN": "Central America", "HR": "Europe",
    "HU": "Europe", "IE": "Europe", "IL": "Middle East", "IN": "Asia",
    "IQ": "Middle East", "IR": "Middle East", "IT": "Europe", "JP": "Asia",
    "KE": "Africa", "LU": "Europe", "MD": "Europe", "MT": "Europe",
    "MX": "North America", "MY": "Asia", "NG": "Africa", "NL": "Europe",
    "NZ": "Oceania", "PK": "Asia", "PL": "Europe", "PT": "Europe",
    "RO": "Europe", "RU": "Europe", "SG": "Asia", "SI": "Europe",
    "TR": "Middle East", "UA": "Europe", "US": "North America", "VN": "Asia",
}

_INDIGO = "#6366f1"
_SLATE = "#cbd5e1"

_PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", color="#1e293b", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


# ── Database Connection ───────────────────────────────────────────────────────
@st.cache_resource
def _get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        try:
            url = url or st.secrets["SUPABASE_URL"]
            key = key or st.secrets["SUPABASE_KEY"]
        except (KeyError, FileNotFoundError):
            pass
    if not url or not key:
        st.error("Supabase credentials not found. "
                 "Set SUPABASE_URL and SUPABASE_KEY in your .env.")
        st.stop()
    return create_client(url, key)


supabase: Client = _get_supabase()


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_csv() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parent.parent / "data" / "ds_salaries.csv"
    if not csv_path.exists():
        st.error(f"Dataset not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    mappings_path = (
        Path(__file__).resolve().parent.parent / "model" / "feature_mappings.joblib"
    )
    if mappings_path.exists():
        cat_map = joblib.load(mappings_path).get("job_category_map", {})
        df["job_category"] = df["job_title"].map(cat_map).fillna("Other")
    else:
        df["job_category"] = df["job_title"]

    df["region"] = df["company_location"].map(_FULL_REGION_MAP).fillna("Other")
    return df


@st.cache_data(ttl=3600)
def fetch_global_insights() -> dict | None:
    try:
        res = supabase.table("global_insights").select("*").eq("id", 1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None


def query_prediction(filters: dict) -> dict | None:
    try:
        q = supabase.table("precomputed_salaries").select("*")
        for k, v in filters.items():
            q = q.eq(k, v)
        res = q.execute()
        return res.data[0] if res.data else None
    except Exception as e:
        st.error(f"Database query error: {e}")
        return None


# ── Glass Box helpers ─────────────────────────────────────────────────────────

def _granular_status(predicted: float, job_category: str) -> str:
    median = CATEGORY_MEDIANS.get(job_category, OVERALL_MEDIAN)
    pct = ((predicted - median) / median * 100) if median else 0
    lo_ext, lo_std, hi_std, hi_ext = CATEGORY_THRESHOLDS.get(
        job_category, _DEFAULT_THRESHOLDS
    )
    if pct < lo_ext:
        return "exceptionally below"
    if pct < lo_std:
        return "below"
    if pct <= hi_std:
        return "on par with"
    if pct <= hi_ext:
        return "above"
    return "exceptionally above"


def _primary_driver(
    exp: str, tier: str, size: str, status: str
) -> tuple[str, str]:
    very_below = "exceptionally below" in status
    below = "below" in status and not very_below
    very_above = "exceptionally above" in status
    above = "above" in status and not very_above

    if very_below and exp in ("EN", "MI"):
        if tier == "Low_Tier":
            return ("the heavily discounted baseline of "
                    f"{TIER_LABELS['Low_Tier']} locations"), "Geography"
        return "extreme outlier macroeconomic factors", "Market Overview"

    if very_above and exp in ("SE", "EX"):
        if exp == "EX":
            return "the extreme premium commanded by Executive leadership", "Experience"
        if exp == "SE" and tier == "High_Tier" and size == "L":
            return ("large-enterprise budgets combined with "
                    f"{TIER_LABELS['High_Tier']} rates"), "Company Size"
        return "highly specialized, niche skill demands", "Market Overview"

    if exp in ("SE", "EX") and above:
        return f"premium compensation for {EXP_LABELS[exp]} expertise", "Experience"
    if exp in ("EN", "MI") and below:
        return f"baseline compensation typical for {EXP_LABELS[exp]} roles", "Experience"

    if exp in ("EN", "MI") and above and tier == "High_Tier":
        return ("competitive Premium Market rates offsetting "
                "lower experience"), "Geography"
    if exp in ("SE", "EX") and below and tier in ("Low_Tier", "Mid_Tier"):
        return (f"regional constraints of "
                f"{TIER_LABELS.get(tier, tier)} locations"), "Geography"

    if exp in ("SE", "EX") and below and size == "S":
        return "budgetary constraints of smaller organisations", "Company Size"
    if exp in ("EN", "MI") and above and size == "L":
        return "premium compensation banding of large enterprises", "Company Size"

    if above or very_above:
        return "specialised skill demands creating a unique premium", "Market Overview"
    if below or very_below:
        return "macroeconomic conditions suppressing this rate", "Market Overview"

    return "standard market equilibrium for this profile", "Market Overview"


# ── Card wrapper ──────────────────────────────────────────────────────────────
def _card_open():
    st.markdown('<div class="stripe-card">', unsafe_allow_html=True)

def _card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_csv()
    insights = fetch_global_insights()

    # ── Edge-to-edge top bar ─────────────────────────────────────────────
    col_logo, col_nav = st.columns([1, 3])

    with col_logo:
        st.markdown("### :material/analytics: AI Salary Analyst")

    with col_nav:
        nav = st.radio(
            "nav",
            ["Market Landscape", "Salary Predictor"],
            horizontal=True,
            label_visibility="collapsed",
        )

    st.divider()

    if nav == "Market Landscape":
        _render_market(df, insights)
    else:
        _render_predictor(df)


# ── Tab 1: Market Landscape ──────────────────────────────────────────────────

def _render_market(df: pd.DataFrame, insights: dict | None):
    st.caption("Macro-level trends from the original training dataset")

    if df.empty:
        st.error("Dataset unavailable. Ensure ds_salaries.csv is in data/.")
        return

    if insights and insights.get("executive_summary"):
        st.info(insights["executive_summary"])
    else:
        st.info(
            f"The global data-science salary market has an overall median of "
            f"**${OVERALL_MEDIAN:,}** across **{len(df):,}** records spanning "
            f"**{df['company_location'].nunique()}** countries. Compensation "
            f"varies significantly by role, seniority, and geography."
        )

    st.divider()

    # ── Row 1 ────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        _card_open()
        st.subheader("Role Distribution")
        counts = df["job_category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.pie(
            counts, names="Category", values="Count",
            hole=0.5,
            color_discrete_sequence=[_INDIGO, "#818cf8", "#a5b4fc", "#c7d2fe"],
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), **_PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        _card_close()

    with c2:
        _card_open()
        st.subheader("The Seniority Ladder")
        trend = (
            df.groupby("experience_level")["salary_in_usd"]
            .median().reindex(EXP_ORDER).reset_index()
        )
        trend["label"] = trend["experience_level"].map(EXP_LABELS)
        fig = px.line(
            trend, x="label", y="salary_in_usd",
            markers=True, color_discrete_sequence=[_INDIGO],
        )
        fig.update_layout(
            xaxis_title="Experience Level",
            yaxis_title="Median Salary (USD)",
            margin=dict(t=10, b=10),
            **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        _card_close()

    # ── Row 2 ────────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        _card_open()
        st.subheader("Regional Salary Comparison")
        reg = (
            df.groupby("region")["salary_in_usd"]
            .median().sort_values().reset_index()
        )
        fig = px.bar(
            reg, x="salary_in_usd", y="region",
            orientation="h", color="salary_in_usd",
            color_continuous_scale="Purp",
        )
        fig.update_layout(
            xaxis_title="Median Salary (USD)", yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        _card_close()

    with c4:
        _card_open()
        st.subheader("Salary Ranges by Role")
        fig = px.box(
            df, x="job_category", y="salary_in_usd",
            color="job_category",
            color_discrete_sequence=[_INDIGO, "#818cf8", "#a5b4fc", "#c7d2fe"],
        )
        fig.update_layout(
            xaxis_title="", yaxis_title="Salary (USD)",
            showlegend=False,
            margin=dict(t=10, b=10),
            **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        _card_close()


# ── Tab 2: Salary Predictor ─────────────────────────────────────────────────

def _render_predictor(df: pd.DataFrame):
    st.caption("AI-powered predictions with Glass Box explanations")

    sorted_countries = sorted(
        ACTIVE_COUNTRIES, key=lambda c: COUNTRY_NAMES.get(c, c)
    )

    # ── Stable search form (prevents flickering) ─────────────────────────
    with st.form("predictor_form"):
        f1, f2, f3, f4, f5 = st.columns(5)

        job = f1.selectbox("Job Category", sorted(CATEGORY_MEDIANS))
        exp = f2.selectbox(
            "Experience Level", EXP_ORDER,
            format_func=lambda x: EXP_LABELS[x],
        )
        country = f3.selectbox(
            "Country", sorted_countries,
            format_func=lambda c: COUNTRY_NAMES.get(c, c),
            index=sorted_countries.index("US"),
        )
        size = f4.selectbox(
            "Company Size", SIZE_ORDER,
            format_func=lambda s: SIZE_LABELS[s],
            index=1,
        )
        remote = f5.selectbox(
            "Work Mode", [0, 50, 100],
            format_func=lambda r: REMOTE_LABELS[r],
        )

        submitted = st.form_submit_button(
            "Get Results", type="primary", use_container_width=True,
        )

    if not submitted:
        st.info("Select your profile above and click **Get Results**.")
        return

    # ── Active filter pills ──────────────────────────────────────────────
    pills_html = (
        '<div class="filter-pills">'
        '<span class="pill-label">Showing results for</span>'
        f'<span class="pill">{job}</span>'
        f'<span class="pill">{EXP_LABELS[exp]}</span>'
        f'<span class="pill">{COUNTRY_NAMES.get(country, country)}</span>'
        f'<span class="pill">{SIZE_LABELS[size]}</span>'
        f'<span class="pill">{REMOTE_LABELS[remote]}</span>'
        '</div>'
    )
    st.markdown(pills_html, unsafe_allow_html=True)

    # ── Query ────────────────────────────────────────────────────────────
    with st.spinner("Querying predictions..."):
        data = query_prediction({
            "job_category": job,
            "experience_level": exp,
            "country_code": country,
            "company_size": size,
            "remote_ratio": remote,
            "is_same_country": 1,
            "employment_type": "FT",
        })

    if not data:
        st.warning(
            "No precomputed prediction found for this combination. "
            "Try adjusting your filters."
        )
        return

    pred = data["predicted_salary"]
    median = CATEGORY_MEDIANS.get(job, OVERALL_MEDIAN)
    delta_pct = ((pred - median) / median * 100) if median else 0

    status = _granular_status(pred, job)
    tier = COUNTRY_TIER.get(country, "Mid_Tier")
    driver_text, chart_type = _primary_driver(exp, tier, size, status)

    # ── Metric cards ─────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Salary", f"${pred:,.0f}",
              delta=f"{delta_pct:+.1f}% vs. {job} Median")
    m2.metric("Category Median", f"${median:,.0f}")
    m3.metric("Market Position", status.title())

    st.divider()

    # ── Gauge + Narrative ────────────────────────────────────────────────
    col_g, col_n = st.columns([1.2, 1])

    with col_g:
        _card_open()
        lo_ext, lo_std, hi_std, hi_ext = CATEGORY_THRESHOLDS.get(
            job, _DEFAULT_THRESHOLDS
        )
        band_excep_lo = median * (1 + lo_ext / 100)
        band_below    = median * (1 + lo_std / 100)
        band_above    = median * (1 + hi_std / 100)
        band_excep_hi = median * (1 + hi_ext / 100)
        gauge_max     = max(band_excep_hi * 1.3, pred * 1.2)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            delta={"reference": median, "valueformat": "$,.0f"},
            number={"valueformat": "$,.0f"},
            title={"text": f"{job} — Market Position",
                    "font": {"size": 15, "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [0, gauge_max], "tickformat": "$,.0f"},
                "bar": {"color": _INDIGO},
                "steps": [
                    {"range": [0, band_excep_lo],           "color": "#fee2e2"},
                    {"range": [band_excep_lo, band_below],  "color": "#fef3c7"},
                    {"range": [band_below, band_above],     "color": "#d1fae5"},
                    {"range": [band_above, band_excep_hi],  "color": "#dbeafe"},
                    {"range": [band_excep_hi, gauge_max],   "color": "#e0e7ff"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "value": median,
                },
            },
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=60, b=20, l=30, r=30),
            **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        _card_close()

    with col_n:
        _card_open()
        st.subheader("AI Narrative")
        st.info(data.get("narrative", "Narrative unavailable."))
        _card_close()

    # ── Evidence chart (Glass Box) ───────────────────────────────────────
    st.divider()
    st.subheader(f"Evidence: {chart_type}")
    st.caption(
        f"Why the model positioned you as **{status}** "
        f"the {job} median"
    )

    if df.empty:
        st.warning("Original dataset unavailable for evidence charts.")
        return

    job_df = df[df["job_category"] == job]

    _card_open()

    very_below = "exceptionally below" in status
    very_above = "exceptionally above" in status

    if very_below and tier == "Low_Tier":
        _chart_region_vs_global(job_df, job, country)
    elif very_above and exp == "EX":
        _chart_exec_premium(job_df, job)
    elif chart_type == "Experience":
        _chart_experience(job_df, job, exp)
    elif chart_type == "Geography":
        _chart_geography(job_df, job, country)
    elif chart_type == "Company Size":
        _chart_company_size(job_df, job, size)
    else:
        _chart_market_overview(job_df, job, pred)

    _card_close()


# ── Evidence chart renderers ─────────────────────────────────────────────────

def _chart_experience(job_df: pd.DataFrame, job: str, current_exp: str):
    """Line chart showing salary progression slope for this role."""
    agg = (
        job_df.groupby("experience_level")["salary_in_usd"]
        .median().reindex(EXP_ORDER).reset_index()
    )
    agg["label"] = agg["experience_level"].map(EXP_LABELS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["label"], y=agg["salary_in_usd"],
        mode="lines+markers+text",
        line=dict(color=_SLATE, width=2.5),
        marker=dict(
            size=[16 if e == current_exp else 8
                  for e in agg["experience_level"]],
            color=[_INDIGO if e == current_exp else _SLATE
                   for e in agg["experience_level"]],
        ),
        text=[f"${v:,.0f}" for v in agg["salary_in_usd"]],
        textposition="top center",
        textfont=dict(size=11, family="Inter"),
    ))
    current_row = agg[agg["experience_level"] == current_exp]
    if not current_row.empty:
        fig.add_annotation(
            x=EXP_LABELS[current_exp],
            y=current_row["salary_in_usd"].iloc[0],
            text=f"You: {EXP_LABELS[current_exp]}",
            showarrow=True, arrowhead=2, arrowcolor=_INDIGO,
            font=dict(color=_INDIGO, size=11, family="Inter"),
            bgcolor="#eef2ff", bordercolor=_INDIGO, borderwidth=1,
            borderpad=4,
        )
    fig.update_layout(
        title=f"Experience Progression — {job}",
        xaxis_title="Experience Level",
        yaxis_title="Median Salary (USD)",
        showlegend=False,
        margin=dict(t=50, b=20),
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_geography(job_df: pd.DataFrame, job: str, current_country: str):
    """Horizontal bar: user's region highlighted in indigo, others in slate."""
    agg = (
        job_df.groupby("region")["salary_in_usd"]
        .median().sort_values().reset_index()
    )
    current_region = COUNTRY_REGION.get(current_country, "")
    colours = [_INDIGO if r == current_region else _SLATE
               for r in agg["region"]]

    fig = go.Figure(go.Bar(
        y=agg["region"], x=agg["salary_in_usd"],
        orientation="h", marker_color=colours,
        text=[f"${v:,.0f}" for v in agg["salary_in_usd"]],
        textposition="outside",
        textfont=dict(size=11, family="Inter"),
    ))
    fig.update_layout(
        title=f"Regional Pay Distribution — {job}",
        xaxis_title="Median Salary (USD)", yaxis_title="",
        showlegend=False,
        margin=dict(t=50, b=20),
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_company_size(job_df: pd.DataFrame, job: str, current_size: str):
    """Grouped bar: salary by company size x experience."""
    agg = (
        job_df.groupby(["company_size", "experience_level"])["salary_in_usd"]
        .median().reset_index()
    )
    agg["size_label"] = agg["company_size"].map(SIZE_LABELS)
    agg["exp_label"] = agg["experience_level"].map(EXP_LABELS)
    agg["size_label"] = pd.Categorical(
        agg["size_label"],
        categories=["Small", "Medium", "Large"],
        ordered=True,
    )
    agg = agg.sort_values("size_label")

    fig = px.bar(
        agg, x="size_label", y="salary_in_usd",
        color="exp_label", barmode="group",
        color_discrete_sequence=[_INDIGO, "#818cf8", "#a5b4fc", "#c7d2fe"],
        title=f"Company Size x Experience — {job}",
    )
    fig.update_layout(
        xaxis_title="Company Size",
        yaxis_title="Median Salary (USD)",
        legend_title="Experience",
        margin=dict(t=50, b=20),
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_market_overview(
    job_df: pd.DataFrame, job: str, pred_salary: float
):
    """Histogram with prediction + median vertical lines."""
    median = CATEGORY_MEDIANS.get(job, OVERALL_MEDIAN)

    fig = px.histogram(
        job_df, x="salary_in_usd", nbins=30,
        color_discrete_sequence=["#a5b4fc"],
        title=f"Salary Distribution — {job}",
    )
    fig.add_vline(
        x=pred_salary, line_dash="dash", line_color=_INDIGO, line_width=2,
        annotation_text=f"Your Prediction: ${pred_salary:,.0f}",
        annotation_position="top right",
        annotation_font=dict(family="Inter", size=11),
    )
    fig.add_vline(
        x=median, line_dash="dot", line_color="#ef4444", line_width=2,
        annotation_text=f"Median: ${median:,.0f}",
        annotation_position="top left",
        annotation_font=dict(family="Inter", size=11),
    )
    fig.update_layout(
        xaxis_title="Salary (USD)", yaxis_title="Count",
        margin=dict(t=50, b=20),
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Specific extreme-condition charts ────────────────────────────────────────

def _chart_region_vs_global(
    job_df: pd.DataFrame, job: str, current_country: str,
):
    """Exceptionally Below + Low_Tier: region vs global median."""
    current_region = COUNTRY_REGION.get(current_country, "")
    global_median = CATEGORY_MEDIANS.get(job, OVERALL_MEDIAN)

    region_median = (
        job_df[job_df["region"] == current_region]["salary_in_usd"].median()
    )
    if pd.isna(region_median):
        region_median = 0

    labels = [f"{current_region} Median", "Global Median"]
    values = [region_median, global_median]
    colours = [_INDIGO, _SLATE]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h", marker_color=colours,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=12, family="Inter"),
    ))
    fig.update_layout(
        title=f"Geographic Discount — {job} in {current_region}",
        xaxis_title="Median Salary (USD)", yaxis_title="",
        margin=dict(t=50, b=20),
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_exec_premium(job_df: pd.DataFrame, job: str):
    """Exceptionally Above + EX: executive salary across company sizes."""
    ex_data = job_df[job_df["experience_level"] == "EX"]
    agg = (
        ex_data.groupby("company_size")["salary_in_usd"]
        .median().reindex(SIZE_ORDER).reset_index()
    )
    agg["size_label"] = agg["company_size"].map(SIZE_LABELS)
    colours = [_INDIGO if s == "L" else _SLATE for s in agg["company_size"]]

    fig = go.Figure(go.Bar(
        x=agg["size_label"], y=agg["salary_in_usd"],
        marker_color=colours,
        text=[f"${v:,.0f}" if pd.notna(v) else "N/A"
              for v in agg["salary_in_usd"]],
        textposition="outside",
        textfont=dict(size=12, family="Inter"),
    ))
    fig.update_layout(
        title=f"Executive Leadership Premium — {job}",
        xaxis_title="Company Size",
        yaxis_title="Median Executive Salary (USD)",
        margin=dict(t=50, b=20),
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
