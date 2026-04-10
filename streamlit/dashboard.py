"""
Salario — AI Salary Intelligence Platform
──────────────────────────────────────────
Premium SaaS dashboard with Stripe-inspired design language.
Edge-to-edge navigation, Inter typeface, isolated high-contrast cards.

Branch: experiment/stripe-ui — visual review only, do not merge.

Data sources:
  - Supabase (precomputed_salaries, global_insights)
  - Local CSV (data/ds_salaries.csv) for evidence charts
  - model/feature_mappings.joblib for job_title -> job_category
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / "pipeline" / ".env")
st.set_page_config(
    page_title="Salario — AI Salary Analyst",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Stripe SaaS Aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Inter typeface — maximum specificity ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [class*="st-"], p, span, div,
    button, input, select, textarea, label,
    .stMarkdown, .stText, .stSelectbox, .stRadio {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.05rem !important;
        color: #475569;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.5rem !important; }

    /* ── App background — cool slate-gray ─────────────────────────────── */
    [data-testid="stAppViewContainer"],
    .stApp {
        background-color: #f7f9fc !important;
    }

    /* ── Edge-to-edge layout ──────────────────────────────────────────── */
    .block-container {
        padding-top: 0 !important;
        max-width: 95% !important;
    }

    /* ── Hide all Streamlit chrome ────────────────────────────────────── */
    [data-testid="stHeader"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
    }

    /* ── Top navbar bar (pure white + bottom line) ────────────────────── */
    .salario-navbar {
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 16px 0;
        margin: 0 -3% 28px -3%;
        padding-left: 3%;
        padding-right: 3%;
        display: flex;
        align-items: center;
    }

    /* ── Salario brand header ─────────────────────────────────────────── */
    .salario-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .salario-logo {
        width: 38px;
        height: 38px;
        position: relative;
    }
    .salario-logo-shape {
        position: absolute;
        border-radius: 8px;
    }
    .salario-logo-shape-1 {
        width: 26px; height: 26px;
        background: #635BFF;
        top: 0; left: 0;
        opacity: 0.9;
    }
    .salario-logo-shape-2 {
        width: 26px; height: 26px;
        background: #80E9FF;
        bottom: 0; right: 0;
        opacity: 0.85;
    }
    .salario-title {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        color: #0f172a !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* ── Horizontal radio as navbar items ──────────────────────────────── */
    [data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 0 !important;
        align-items: center !important;
    }
    [data-testid="stRadio"] > div {
        gap: 10px !important;
    }
    [data-testid="stRadio"] > div > label {
        padding: 10px 20px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: #475569 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04) !important;
    }
    [data-testid="stRadio"] > div > label:hover {
        color: #0f172a !important;
        background: #f1f5f9 !important;
    }
    [data-testid="stRadio"] > div > label[data-checked="true"],
    [data-testid="stRadio"] > div > label:has(input:checked) {
        color: #ffffff !important;
        background: #635BFF !important;
        border-color: #635BFF !important;
        box-shadow: 0 6px 14px rgba(99, 91, 255, 0.25) !important;
    }
    [data-testid="stRadio"] > div > label > div:first-child {
        display: none !important;
    }

    /* ── Isolated high-contrast cards — target Streamlit data-testids ── */
    [data-testid="stPlotlyChart"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05),
                    0 2px 4px -1px rgba(0,0,0,0.03) !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05),
                    0 2px 4px -1px rgba(0,0,0,0.03) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    div[data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05),
                    0 2px 4px -1px rgba(0,0,0,0.03) !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* ── Info box ─────────────────────────────────────────────────────── */
    .stInfo, [data-testid="stAlert"] {
        background: #eef2ff !important;
        border-left: 4px solid #635BFF !important;
        border-radius: 0 12px 12px 0 !important;
        font-size: 1rem !important;
    }

    /* ── Primary button ───────────────────────────────────────────────── */
    button[kind="primaryFormSubmit"],
    .stButton > button[kind="primary"] {
        background: #635BFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 12px 28px !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
        color: #ffffff !important;
    }
    button[kind="primaryFormSubmit"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: #5046E5 !important;
        box-shadow: 0 6px 20px rgba(99, 91, 255, 0.35) !important;
        transform: translateY(-1px);
    }

    /* ── Filter pills ─────────────────────────────────────────────────── */
    .filter-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0 28px 0;
        align-items: center;
    }
    .filter-pills .pill-label {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .filter-pills .pill {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #c7d2fe;
    }

    /* ── Section card (manual wrapper) ────────────────────────────────── */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05),
                    0 2px 4px -1px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }

    /* ── Dividers ─────────────────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid #e2e8f0 !important;
        margin: 24px 0 !important;
    }

    /* ── Select boxes — larger text ───────────────────────────────────── */
    [data-testid="stSelectbox"] label {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Label maps ────────────────────────────────────────────────────────────────
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

_INDIGO = "#635BFF"
_SLATE = "#cbd5e1"

# ── Plotly defaults — transparent bg, larger fonts ────────────────────────────
_PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", color="#475569", size=14),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title_font=dict(size=18, color="#0f172a", family="Inter, sans-serif"),
    xaxis=dict(tickfont=dict(size=13), title_font=dict(size=14)),
    yaxis=dict(tickfont=dict(size=13), title_font=dict(size=14)),
)
_PLOTLY_CONFIG = {"displayModeBar": False, "staticPlot": True, "responsive": True}


# ── Database ──────────────────────────────────────────────────────────────────
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


# ── Data ──────────────────────────────────────────────────────────────────────
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
def get_global_insights() -> dict | None:
    """Fetch the latest global insights payload from Supabase."""
    try:
        res = supabase.table("global_insights").select("*").limit(1).execute()
        if res.data:
            return res.data[0]
    except Exception:
        st.warning("Unable to load Supabase global insights. Showing defaults.")
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


def _primary_driver(exp, tier, size, status):
    vb = "exceptionally below" in status
    b = "below" in status and not vb
    va = "exceptionally above" in status
    a = "above" in status and not va

    if vb and exp in ("EN", "MI"):
        if tier == "Low_Tier":
            return (f"the heavily discounted baseline of "
                    f"{TIER_LABELS['Low_Tier']} locations"), "Geography"
        return "extreme outlier macroeconomic factors", "Market Overview"
    if va and exp in ("SE", "EX"):
        if exp == "EX":
            return "the extreme premium commanded by Executive leadership", "Experience"
        if exp == "SE" and tier == "High_Tier" and size == "L":
            return (f"large-enterprise budgets combined with "
                    f"{TIER_LABELS['High_Tier']} rates"), "Company Size"
        return "highly specialized, niche skill demands", "Market Overview"
    if exp in ("SE", "EX") and a:
        return f"premium compensation for {EXP_LABELS[exp]} expertise", "Experience"
    if exp in ("EN", "MI") and b:
        return f"baseline compensation typical for {EXP_LABELS[exp]} roles", "Experience"
    if exp in ("EN", "MI") and a and tier == "High_Tier":
        return "competitive Premium Market rates offsetting lower experience", "Geography"
    if exp in ("SE", "EX") and b and tier in ("Low_Tier", "Mid_Tier"):
        return f"regional constraints of {TIER_LABELS.get(tier, tier)} locations", "Geography"
    if exp in ("SE", "EX") and b and size == "S":
        return "budgetary constraints of smaller organisations", "Company Size"
    if exp in ("EN", "MI") and a and size == "L":
        return "premium compensation banding of large enterprises", "Company Size"
    if a or va:
        return "specialised skill demands creating a unique premium", "Market Overview"
    if b or vb:
        return "macroeconomic conditions suppressing this rate", "Market Overview"
    return "standard market equilibrium for this profile", "Market Overview"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df = load_csv()
    insights = get_global_insights()

    # ── Salario branded top navbar (raw HTML) ────────────────────────────
    st.markdown("""
    <div class="salario-navbar">
        <div class="salario-brand">
            <div class="salario-logo">
                <div class="salario-logo-shape salario-logo-shape-1"></div>
                <div class="salario-logo-shape salario-logo-shape-2"></div>
            </div>
            <span class="salario-title">Salario</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation radio (styled as navbar pills) ────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# MARKET LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════
def _render_market(df: pd.DataFrame, insights: dict | None):
    st.caption("Macro-level trends from the original training dataset")

    if df.empty:
        st.error("Dataset unavailable. Ensure ds_salaries.csv is in data/.")
        return

    captions = insights.get("chart_captions", {}) if insights else {}
    exec_summary = insights.get("executive_summary") if insights else None
    transparency_note = insights.get("data_transparency_note") if insights else None

    def _caption(key: str, fallback: str) -> str:
        if isinstance(captions, str):
            try:
                parsed = json.loads(captions)
            except json.JSONDecodeError:
                parsed = {}
            return parsed.get(key, fallback)
        return captions.get(key, fallback) if captions else fallback

    summary_text = exec_summary or (
        f"The global data-science salary market has an overall median of "
        f"**${OVERALL_MEDIAN:,}** across **{len(df):,}** records spanning "
        f"**{df['company_location'].nunique()}** countries. Compensation "
        f"varies significantly by role, seniority, and geography."
    )

    st.markdown(
        f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                    padding:18px 22px;margin-bottom:18px;font-size:1.15rem;
                    line-height:1.65;color:#0f172a;">
            {summary_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Drivers: 2-column grid ─────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        counts = df["job_category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.pie(
            counts, names="Category", values="Count", hole=0.5,
            color_discrete_sequence=[_INDIGO, "#818cf8", "#a5b4fc", "#c7d2fe"],
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(font=dict(size=13)),
            **_PLOTLY_LAYOUT,
        )
        _chart_card(
            "Role Distribution",
            _caption("role_distribution", "Role mix across the dataset."),
            fig,
        )

    with c2:
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
        _chart_card(
            "The Seniority Ladder",
            _caption("seniority_ladder", "Median salary climbs with experience."),
            fig,
        )

    # ── Additional drivers ─────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
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
        _chart_card(
            "Regional Salary Comparison",
            _caption("regional_comparison", "Geographic pay spread across regions."),
            fig,
        )

    with c4:
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
        _chart_card(
            "Salary Ranges by Role",
            _caption("salary_ranges_by_role", "Distribution spread within each role."),
            fig,
        )

    # ── Featured: Remote Premium (full width) ──────────────────────────
    remote_caption = _caption(
        "remote_premium",
        "Remote flexibility shifts median compensation versus on-site roles.",
    )
    remote_fig = _remote_premium_chart(df)
    _chart_card("Remote Premium", remote_caption, remote_fig)

    # ── XAI footer ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Data Methodology & Transparency")
    footer_note = transparency_note or (
        "Dataset skewed toward North America/Europe; location tiers were grouped "
        "to stabilise estimates in emerging markets."
    )
    st.markdown(
        f'<p style="font-size:0.9rem; color:#94a3b8; line-height:1.6;">{footer_note}</p>',
        unsafe_allow_html=True,
    )


def _chart_card(title: str, caption: str, fig):
    """Uniform card wrapper: title → caption → chart."""
    with st.container(border=True):
        st.subheader(title)
        st.markdown(
            f'<p style="color:#64748b;margin:4px 0 12px 0;'
            f'font-size:1rem;line-height:1.5;">{caption}</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


def _remote_premium_chart(df: pd.DataFrame) -> go.Figure:
    """Featured remote vs on-site premium chart."""
    if df.empty or "remote_ratio" not in df.columns:
        return go.Figure()

    order = [0, 50, 100]
    agg = (
        df[df["remote_ratio"].isin(order)]
        .groupby("remote_ratio")["salary_in_usd"]
        .median()
        .reindex(order)
        .reset_index()
    )
    agg["label"] = agg["remote_ratio"].map(REMOTE_LABELS)

    fig = go.Figure(go.Bar(
        x=agg["label"],
        y=agg["salary_in_usd"],
        marker_color=[_SLATE, "#818cf8", _INDIGO],
        text=[f"${v:,.0f}" if pd.notna(v) else "N/A" for v in agg["salary_in_usd"]],
        textposition="outside",
        textfont=dict(size=14, family="Inter"),
    ))
    fig.update_layout(
        xaxis_title="Work Mode",
        yaxis_title="Median Salary (USD)",
        margin=dict(t=30, b=20),
        showlegend=False,
        **_PLOTLY_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SALARY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
def _render_predictor(df: pd.DataFrame):
    st.caption("AI-powered predictions with Glass Box explanations")

    sorted_countries = sorted(
        ACTIVE_COUNTRIES, key=lambda c: COUNTRY_NAMES.get(c, c)
    )

    with st.form("predictor_form"):
        f1, f2, f3, f4, f5 = st.columns(5)
        job = f1.selectbox("Job Category", sorted(CATEGORY_MEDIANS))
        exp = f2.selectbox("Experience Level", EXP_ORDER,
                           format_func=lambda x: EXP_LABELS[x])
        country = f3.selectbox("Country", sorted_countries,
                               format_func=lambda c: COUNTRY_NAMES.get(c, c),
                               index=sorted_countries.index("US"))
        size = f4.selectbox("Company Size", SIZE_ORDER,
                            format_func=lambda s: SIZE_LABELS[s], index=1)
        remote = f5.selectbox("Work Mode", [0, 50, 100],
                              format_func=lambda r: REMOTE_LABELS[r])
        submitted = st.form_submit_button(
            "Get Results", type="primary", use_container_width=True)

    if not submitted:
        st.info("Select your profile above and click **Get Results**.")
        return

    # ── Filter pills ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="filter-pills">'
        '<span class="pill-label">Showing results for</span>'
        f'<span class="pill">{job}</span>'
        f'<span class="pill">{EXP_LABELS[exp]}</span>'
        f'<span class="pill">{COUNTRY_NAMES.get(country, country)}</span>'
        f'<span class="pill">{SIZE_LABELS[size]}</span>'
        f'<span class="pill">{REMOTE_LABELS[remote]}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Querying predictions..."):
        data = query_prediction({
            "job_category": job, "experience_level": exp,
            "country_code": country, "company_size": size,
            "remote_ratio": remote, "is_same_country": 1,
            "employment_type": "FT",
        })

    if not data:
        st.warning("No precomputed prediction found. Try adjusting your filters.")
        return

    pred = data["predicted_salary"]
    median = CATEGORY_MEDIANS.get(job, OVERALL_MEDIAN)
    delta_pct = ((pred - median) / median * 100) if median else 0
    status = _granular_status(pred, job)
    tier = COUNTRY_TIER.get(country, "Mid_Tier")
    driver_text, chart_type = _primary_driver(exp, tier, size, status)

    # ── Metrics ──────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Salary", f"${pred:,.0f}",
              delta=f"{delta_pct:+.1f}% vs. {job} Median")
    m2.metric("Category Median", f"${median:,.0f}")
    m3.metric("Market Position", status.title())

    st.divider()

    # ── Gauge + Narrative ────────────────────────────────────────────────
    col_g, col_n = st.columns([1.2, 1])

    with col_g:
        lo_ext, lo_std, hi_std, hi_ext = CATEGORY_THRESHOLDS.get(
            job, _DEFAULT_THRESHOLDS)
        be = median * (1 + lo_ext / 100)
        bb = median * (1 + lo_std / 100)
        ba = median * (1 + hi_std / 100)
        bh = median * (1 + hi_ext / 100)
        gmax = max(bh * 1.3, pred * 1.2)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred,
            delta={"reference": median, "valueformat": "$,.0f"},
            number={"valueformat": "$,.0f",
                     "font": {"size": 36, "family": "Inter", "color": "#0f172a"}},
            title={"text": f"{job} — Market Position",
                    "font": {"size": 16, "family": "Inter", "color": "#475569"}},
            gauge={
                "axis": {"range": [0, gmax], "tickformat": "$,.0f",
                         "tickfont": {"size": 11}},
                "bar": {"color": _INDIGO},
                "steps": [
                    {"range": [0, be],    "color": "#fee2e2"},
                    {"range": [be, bb],   "color": "#fef3c7"},
                    {"range": [bb, ba],   "color": "#d1fae5"},
                    {"range": [ba, bh],   "color": "#dbeafe"},
                    {"range": [bh, gmax], "color": "#e0e7ff"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 3},
                              "value": median},
            },
        ))
    fig.update_layout(
        height=320, margin=dict(t=70, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)

    with col_n:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("AI Narrative")
        st.info(data.get("narrative", "Narrative unavailable."))
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Evidence (Glass Box) ─────────────────────────────────────────────
    st.divider()
    st.subheader(f"Evidence: {chart_type}")
    st.caption(
        f"Why the model positioned you as **{status}** the {job} median")

    if df.empty:
        st.warning("Original dataset unavailable for evidence charts.")
        return

    job_df = df[df["job_category"] == job]
    vb = "exceptionally below" in status
    va = "exceptionally above" in status

    if vb and tier == "Low_Tier":
        _chart_region_vs_global(job_df, job, country)
    elif va and exp == "EX":
        _chart_exec_premium(job_df, job)
    elif chart_type == "Experience":
        _chart_experience(job_df, job, exp)
    elif chart_type == "Geography":
        _chart_geography(job_df, job, country)
    elif chart_type == "Company Size":
        _chart_company_size(job_df, job, size)
    else:
        _chart_market_overview(job_df, job, pred)


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _chart_experience(job_df, job, current_exp):
    agg = (job_df.groupby("experience_level")["salary_in_usd"]
           .median().reindex(EXP_ORDER).reset_index())
    agg["label"] = agg["experience_level"].map(EXP_LABELS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["label"], y=agg["salary_in_usd"],
        mode="lines+markers+text",
        line=dict(color=_SLATE, width=2.5),
        marker=dict(
            size=[18 if e == current_exp else 8 for e in agg["experience_level"]],
            color=[_INDIGO if e == current_exp else _SLATE
                   for e in agg["experience_level"]],
        ),
        text=[f"${v:,.0f}" for v in agg["salary_in_usd"]],
        textposition="top center",
        textfont=dict(size=13, family="Inter"),
    ))
    cur = agg[agg["experience_level"] == current_exp]
    if not cur.empty:
        fig.add_annotation(
            x=EXP_LABELS[current_exp], y=cur["salary_in_usd"].iloc[0],
            text=f"You: {EXP_LABELS[current_exp]}",
            showarrow=True, arrowhead=2, arrowcolor=_INDIGO,
            font=dict(color=_INDIGO, size=13, family="Inter"),
            bgcolor="#eef2ff", bordercolor=_INDIGO, borderwidth=1, borderpad=5,
        )
    fig.update_layout(
        title=f"Experience Progression — {job}",
        xaxis_title="Experience Level", yaxis_title="Median Salary (USD)",
        showlegend=False, margin=dict(t=50, b=20), **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


def _chart_geography(job_df, job, current_country):
    agg = (job_df.groupby("region")["salary_in_usd"]
           .median().sort_values().reset_index())
    cur_region = COUNTRY_REGION.get(current_country, "")
    colours = [_INDIGO if r == cur_region else _SLATE for r in agg["region"]]

    fig = go.Figure(go.Bar(
        y=agg["region"], x=agg["salary_in_usd"],
        orientation="h", marker_color=colours,
        text=[f"${v:,.0f}" for v in agg["salary_in_usd"]],
        textposition="outside", textfont=dict(size=13, family="Inter"),
    ))
    fig.update_layout(
        title=f"Regional Pay Distribution — {job}",
        xaxis_title="Median Salary (USD)", yaxis_title="",
        showlegend=False, margin=dict(t=50, b=20), **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


def _chart_company_size(job_df, job, current_size):
    agg = (job_df.groupby(["company_size", "experience_level"])["salary_in_usd"]
           .median().reset_index())
    agg["size_label"] = agg["company_size"].map(SIZE_LABELS)
    agg["exp_label"] = agg["experience_level"].map(EXP_LABELS)
    agg["size_label"] = pd.Categorical(
        agg["size_label"], categories=["Small", "Medium", "Large"], ordered=True)
    agg = agg.sort_values("size_label")

    fig = px.bar(
        agg, x="size_label", y="salary_in_usd",
        color="exp_label", barmode="group",
        color_discrete_sequence=[_INDIGO, "#818cf8", "#a5b4fc", "#c7d2fe"],
        title=f"Company Size x Experience — {job}",
    )
    fig.update_layout(
        xaxis_title="Company Size", yaxis_title="Median Salary (USD)",
        legend_title="Experience", margin=dict(t=50, b=20), **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


def _chart_market_overview(job_df, job, pred_salary):
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
        annotation_font=dict(family="Inter", size=13),
    )
    fig.add_vline(
        x=median, line_dash="dot", line_color="#ef4444", line_width=2,
        annotation_text=f"Median: ${median:,.0f}",
        annotation_position="top left",
        annotation_font=dict(family="Inter", size=13),
    )
    fig.update_layout(
        xaxis_title="Salary (USD)", yaxis_title="Count",
        margin=dict(t=50, b=20), **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


def _chart_region_vs_global(job_df, job, current_country):
    cur_region = COUNTRY_REGION.get(current_country, "")
    global_med = CATEGORY_MEDIANS.get(job, OVERALL_MEDIAN)
    region_med = job_df[job_df["region"] == cur_region]["salary_in_usd"].median()
    if pd.isna(region_med):
        region_med = 0

    fig = go.Figure(go.Bar(
        x=[region_med, global_med],
        y=[f"{cur_region} Median", "Global Median"],
        orientation="h", marker_color=[_INDIGO, _SLATE],
        text=[f"${v:,.0f}" for v in [region_med, global_med]],
        textposition="outside", textfont=dict(size=14, family="Inter"),
    ))
    fig.update_layout(
        title=f"Geographic Discount — {job} in {cur_region}",
        xaxis_title="Median Salary (USD)", yaxis_title="",
        margin=dict(t=50, b=20), **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


def _chart_exec_premium(job_df, job):
    ex = job_df[job_df["experience_level"] == "EX"]
    agg = ex.groupby("company_size")["salary_in_usd"].median().reindex(SIZE_ORDER).reset_index()
    agg["size_label"] = agg["company_size"].map(SIZE_LABELS)
    colours = [_INDIGO if s == "L" else _SLATE for s in agg["company_size"]]

    fig = go.Figure(go.Bar(
        x=agg["size_label"], y=agg["salary_in_usd"],
        marker_color=colours,
        text=[f"${v:,.0f}" if pd.notna(v) else "N/A" for v in agg["salary_in_usd"]],
        textposition="outside", textfont=dict(size=14, family="Inter"),
    ))
    fig.update_layout(
        title=f"Executive Leadership Premium — {job}",
        xaxis_title="Company Size", yaxis_title="Median Executive Salary (USD)",
        margin=dict(t=50, b=20), **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)


if __name__ == "__main__":
    main()
