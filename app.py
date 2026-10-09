"""
Phase 10 — Dashboard (redesigned UI)
Intelligent E-Commerce Customer Analytics Platform

Run with:  streamlit run app.py   (from inside the dashboard/ folder)

Modules
- Executive Dashboard   : Total Revenue, Total Customers, Orders, Avg Order Value
- Customer Analytics    : Segment, Purchase Probability, Predicted CLV, Profile
- Recommendation Engine : Top 5 recommended products (optional)
- Business Insights     : Sales Trends, Category / Seller Performance, Geographic Sales
- Explainability        : SHAP Visualization, Feature Importance

Background: a built-in analytics-themed image is generated in code. To use your
own photo instead, drop a file named background.jpg / .jpeg / .png / .webp into
dashboard/assets/ and it will be picked up automatically.
"""

import base64
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
PROC_DIR = APP_DIR / "data" / "processed"
MODELS_DIR = APP_DIR / "models"
ASSETS_DIR = APP_DIR / "assets"

FEATURE_COLS = ["recency_days", "frequency", "monetary_value", "avg_order_value",
                "avg_basket_size", "customer_tenure_days", "avg_review_rating",
                "preferred_payment_method", "customer_state_grouped"]

PALETTE = ["#4F46E5", "#14B8A6", "#F59E0B", "#EC4899", "#3B82F6", "#8B5CF6", "#10B981", "#EF4444"]

PERSONAS = {
    "Premium Customers": ("💎", "#4F46E5", "High spenders who buy often. Protect and reward them."),
    "Regular Customers": ("🛒", "#14B8A6", "Steady, average buyers. Nudge them toward more frequent orders."),
    "Bargain Hunters": ("🏷️", "#F59E0B", "Frequent but low-value orders. Respond well to discounts and bundles."),
    "New Customers": ("🌱", "#3B82F6", "Recently joined, few orders so far. Focus on a great second purchase."),
    "Dormant Customers": ("😴", "#EF4444", "Haven't bought in a long time. Candidates for win-back campaigns."),
}


# ============================================================
# Background image + global styling
# ============================================================
def build_background_svg() -> str:
    """Abstract 'analytics' backdrop: gradient, soft glow blobs, grid, chart line and bars."""
    bars = "".join(
        f"<rect x='{1180 + i * 46}' y='{760 - h}' width='26' height='{h}' rx='6' fill='white' fill-opacity='0.07'/>"
        for i, h in enumerate([90, 140, 110, 190, 160, 240, 210, 300, 270, 360])
    )
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='1920' height='1080' viewBox='0 0 1920 1080'>
<defs>
<linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
<stop offset='0' stop-color='#0B1F3A'/><stop offset='0.55' stop-color='#1E1B4B'/><stop offset='1' stop-color='#0F766E'/>
</linearGradient>
<filter id='b' x='-50%' y='-50%' width='200%' height='200%'><feGaussianBlur stdDeviation='90'/></filter>
<pattern id='grid' width='64' height='64' patternUnits='userSpaceOnUse'>
<path d='M64 0H0V64' fill='none' stroke='white' stroke-opacity='0.05'/>
</pattern>
<linearGradient id='line' x1='0' y1='0' x2='1' y2='0'>
<stop offset='0' stop-color='#14B8A6' stop-opacity='0'/><stop offset='0.5' stop-color='#14B8A6' stop-opacity='0.55'/><stop offset='1' stop-color='#F59E0B' stop-opacity='0.55'/>
</linearGradient>
</defs>
<rect width='1920' height='1080' fill='url(#g)'/>
<rect width='1920' height='1080' fill='url(#grid)'/>
<g filter='url(#b)'>
<circle cx='260' cy='200' r='260' fill='#4F46E5' fill-opacity='0.45'/>
<circle cx='1650' cy='180' r='230' fill='#14B8A6' fill-opacity='0.35'/>
<circle cx='1450' cy='900' r='300' fill='#EC4899' fill-opacity='0.22'/>
<circle cx='300' cy='930' r='260' fill='#F59E0B' fill-opacity='0.18'/>
</g>
{bars}
<polyline points='0,820 160,780 320,800 480,700 640,730 800,610 960,640 1120,520 1280,540 1440,420 1600,440 1760,330 1920,300'
 fill='none' stroke='url(#line)' stroke-width='5' stroke-linecap='round' stroke-linejoin='round'/>
<g fill='white' fill-opacity='0.25'>
<circle cx='480' cy='700' r='7'/><circle cx='800' cy='610' r='7'/><circle cx='1120' cy='520' r='7'/><circle cx='1440' cy='420' r='7'/><circle cx='1760' cy='330' r='7'/>
</g>
</svg>"""


@st.cache_resource
def background_data_uri() -> str:
    for ext, mime in [("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("png", "image/png"), ("webp", "image/webp")]:
        p = ASSETS_DIR / f"background.{ext}"
        if p.exists():
            return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    return "data:image/svg+xml;base64," + base64.b64encode(build_background_svg().encode()).decode()


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

/* ---------- Background ---------- */
.stApp {
    background-image: linear-gradient(rgba(8, 18, 38, 0.30), rgba(8, 18, 38, 0.30)), url("__BG__");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Main content panel ---------- */
.block-container, [data-testid="stMainBlockContainer"] {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 26px;
    padding: 2rem 2.6rem 2.4rem 2.6rem !important;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    max-width: 1400px;
    box-shadow: 0 25px 60px rgba(2, 6, 23, 0.45);
    color: #0F172A;
}
.block-container h1, .block-container h2, .block-container h3, .block-container h4,
.block-container p, .block-container label, .block-container li, .block-container span {
    color: #0F172A;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1F3A 0%, #1E1B4B 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.06);
    padding: 0.65rem 0.9rem;
    border-radius: 12px;
    margin-bottom: 0.4rem;
    transition: all .15s ease;
    border: 1px solid transparent;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.14);
    border-color: rgba(255,255,255,0.18);
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight: 600; font-size: 0.95rem; }
.brand { display:flex; align-items:center; gap:.7rem; padding:.4rem 0 1.1rem 0; }
.brand-logo { font-size:1.9rem; background: linear-gradient(135deg,#4F46E5,#14B8A6);
    border-radius:14px; padding:.35rem .55rem; box-shadow:0 6px 18px rgba(79,70,229,.45); }
.brand-title { font-weight:800; font-size:1.05rem; line-height:1.15; }
.brand-sub { font-size:.72rem; opacity:.7; letter-spacing:.04em; text-transform:uppercase; }
.side-note { font-size:.78rem; opacity:.75; line-height:1.5; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(120deg, #0B1F3A 0%, #4F46E5 60%, #14B8A6 130%);
    border-radius: 20px; padding: 1.5rem 1.8rem; margin-bottom: 1.4rem;
    display:flex; align-items:center; gap:1.1rem;
    box-shadow: 0 14px 30px rgba(79,70,229,.30);
}
.hero-icon { font-size:2.4rem; background:rgba(255,255,255,.16); border-radius:16px; padding:.5rem .8rem; }
.hero h1 { color:#fff !important; margin:0; font-size:1.7rem; font-weight:800; letter-spacing:-.01em; }
.hero p { color:rgba(255,255,255,.85) !important; margin:.15rem 0 0 0; font-size:.95rem; }

/* ---------- KPI cards ---------- */
.kpi { background:#fff; border-radius:18px; padding:1.1rem 1.2rem; display:flex; align-items:center; gap:1rem;
    border:1px solid #E2E8F0; border-left:6px solid var(--accent);
    box-shadow:0 8px 22px rgba(15,23,42,.07); transition: transform .15s ease, box-shadow .15s ease; height:100%; }
.kpi:hover { transform: translateY(-3px); box-shadow:0 14px 28px rgba(15,23,42,.12); }
.kpi-icon { font-size:1.7rem; width:3rem; height:3rem; display:flex; align-items:center; justify-content:center;
    border-radius:14px; background: color-mix(in srgb, var(--accent) 14%, white); }
.kpi-label { font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:#64748B !important; font-weight:600; }
.kpi-value { font-size:1.55rem; font-weight:800; color:#0F172A !important; line-height:1.2; }
.kpi-sub { font-size:.78rem; color:#64748B !important; }

/* ---------- Generic cards / chips ---------- */
.card { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:16px; padding:1rem 1.2rem; height:100%; }
.chip { display:inline-block; padding:.35rem .9rem; border-radius:999px; font-weight:700; font-size:.9rem; color:#fff !important; }
.persona-box { border-radius:18px; padding:1.2rem 1.4rem; color:#fff; height:100%;
    box-shadow:0 10px 24px rgba(15,23,42,.18); }
.persona-box h3, .persona-box p { color:#fff !important; margin:0; }
.insight { background: linear-gradient(135deg,#EEF2FF,#ECFEFF); border:1px solid #C7D2FE; border-radius:16px;
    padding:.9rem 1.1rem; font-size:.95rem; height:100%; }
.rec-card { display:flex; gap:1rem; align-items:center; background:#fff; border:1px solid #E2E8F0; border-radius:16px;
    padding:.9rem 1.1rem; margin-bottom:.7rem; box-shadow:0 6px 16px rgba(15,23,42,.06); }
.rec-rank { width:2.6rem; height:2.6rem; border-radius:50%; background:linear-gradient(135deg,#4F46E5,#14B8A6);
    color:#fff !important; font-weight:800; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.rec-title { font-weight:700; font-size:.95rem; color:#0F172A !important; word-break:break-all; }
.rec-meta { color:#64748B !important; font-size:.82rem; }
.match-bar { height:8px; border-radius:99px; background:#E2E8F0; overflow:hidden; margin-top:.4rem; }
.match-fill { height:100%; background:linear-gradient(90deg,#4F46E5,#14B8A6); border-radius:99px; }
.footer { text-align:center; color:#94A3B8 !important; font-size:.8rem; margin-top:1.5rem; }

/* ---------- Tabs & inputs ---------- */
.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { background:#F1F5F9; border-radius:12px 12px 0 0; padding:.55rem 1.1rem; font-weight:600; }
.stTabs [aria-selected="true"] { background:#4F46E5 !important; }
.stTabs [aria-selected="true"] p { color:#fff !important; }
.stButton > button { border-radius:12px; font-weight:600; border:1px solid #CBD5E1; }
.stButton > button:hover { border-color:#4F46E5; color:#4F46E5; }
</style>
"""


def inject_css():
    st.markdown(CSS.replace("__BG__", background_data_uri()), unsafe_allow_html=True)


# ============================================================
# UI helpers
# ============================================================
def hero(icon: str, title: str, subtitle: str):
    st.markdown(
        f"<div class='hero'><div class='hero-icon'>{icon}</div>"
        f"<div><h1>{title}</h1><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, icon: str, color: str, sub: str = "") -> str:
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='kpi' style='--accent:{color}'><div class='kpi-icon'>{icon}</div>"
        f"<div><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>{sub_html}</div></div>"
    )


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        colorway=PALETTE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Only style the title when one exists. Setting a title font on a chart with
    # no title text makes Plotly render the literal word "undefined".
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(size=16))
    else:
        fig.update_layout(title_text="", margin=dict(l=10, r=10, t=20, b=10))
    return fig


def pretty_feature(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ").title()


def money(x: float) -> str:
    return f"R$ {x:,.0f}"


BRAZIL_STATES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais", "PA": "Pará",
    "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins",
}


def state_name(code) -> str:
    """'SP' -> 'São Paulo'. Unknown codes (e.g. the grouped 'other' bucket) are handled gracefully."""
    code = str(code).strip().upper()
    if code == "OTHER":
        return "Other states"
    return BRAZIL_STATES.get(code, code)


# ============================================================
# Data / model loaders (cached)
# ============================================================
def _missing_file_error(path: Path, phase_hint: str):
    st.error(
        f"Couldn't find `{path.relative_to(BASE_DIR)}`.\n\n"
        f"Run **{phase_hint}** first. It generates this file."
    )
    st.stop()


@st.cache_data
def load_transactions():
    path = PROC_DIR / "olist_with_time_features.csv"
    if not path.exists():
        _missing_file_error(path, "04_feature_engineering.ipynb")
    df = pd.read_csv(path, parse_dates=["order_purchase_timestamp"])
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def load_csv(name: str, phase_hint: str):
    path = PROC_DIR / name
    if not path.exists():
        _missing_file_error(path, phase_hint)
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    p_path = MODELS_DIR / "purchase_prediction_model.pkl"
    c_path = MODELS_DIR / "clv_prediction_model.pkl"
    if not p_path.exists():
        _missing_file_error(p_path, "06_purchase_prediction.ipynb")
    if not c_path.exists():
        _missing_file_error(c_path, "07_clv_prediction.ipynb")
    return joblib.load(p_path), joblib.load(c_path)


@st.cache_resource
def load_rec_artifacts():
    path = MODELS_DIR / "recommendation_artifacts.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_brazil_geojson():
    try:
        import requests
        url = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-uf.json"
        return requests.get(url, timeout=8).json()
    except Exception:
        return None


def make_predict_fn_transformed(pipeline, mode="proba"):
    """Explain the model in the preprocessed numeric space (same fix as Phase 9)."""
    est_name = "clf" if "clf" in pipeline.named_steps else "reg"
    estimator = pipeline.named_steps[est_name]
    preprocessor = pipeline.named_steps["preprocess"]

    def transform(X_raw_df):
        X_t = preprocessor.transform(X_raw_df)
        return X_t.toarray() if hasattr(X_t, "toarray") else X_t

    def predict_fn(X_transformed):
        if mode == "proba":
            return estimator.predict_proba(X_transformed)[:, 1]
        return estimator.predict(X_transformed)

    return transform, predict_fn


def get_feature_importance(pipeline, feature_names):
    est_name = "clf" if "clf" in pipeline.named_steps else "reg"
    est = pipeline.named_steps[est_name]
    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
    elif hasattr(est, "coef_"):
        importances = np.abs(np.ravel(est.coef_))
    else:
        return None
    return pd.Series(importances, index=[pretty_feature(f) for f in feature_names]).sort_values(ascending=False)


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_pipeline, model_name, mode, background_raw):
    transform, predict_fn = make_predict_fn_transformed(_pipeline, mode=mode)
    background = transform(background_raw)
    feature_names = _pipeline.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
    return explainer, transform, predict_fn


# ============================================================
# Shared customer picker
# ============================================================
if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None


PAGE_SIZE = 1000        # customers per page when browsing without a search
MAX_SEARCH_RESULTS = 2000


def customer_selector(customer_ids, key):
    """Search + random + paged dropdown. Every customer is reachable:
    - Search matches ANY part of the ID (case-insensitive); an exact ID jumps straight to that customer.
    - With no search, the dropdown is paged (PAGE_SIZE per page) so the full list can be browsed.
    The selection persists when you switch pages of the dashboard."""
    customer_ids = list(customer_ids)
    total = len(customer_ids)
    n_pages = max(1, -(-total // PAGE_SIZE))
    page_key = f"{key}_page"
    st.session_state.setdefault(page_key, 1)

    c1, c2 = st.columns([5, 1])
    with c1:
        search = st.text_input(
            "🔍 Find a customer",
            placeholder="Paste or type any part of a customer ID, then press Enter…",
            key=f"{key}_search",
        ).strip().lower()
    with c2:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random", key=f"{key}_random"):
            pick = random.choice(customer_ids)
            st.session_state.selected_customer = pick
            st.session_state[page_key] = customer_ids.index(pick) // PAGE_SIZE + 1  # land on its page
            search = ""  # Random overrides the search box for this run

    selected = st.session_state.selected_customer

    if search:
        matches = [c for c in customer_ids if search in c.lower()]
        if not matches:
            st.warning(f"No customer ID contains “{search}”. Check for typos or try a shorter fragment.")
            return None
        exact = [c for c in matches if c.lower() == search]
        if exact:
            options, idx = exact, 0
        else:
            options = matches[:MAX_SEARCH_RESULTS]
            idx = options.index(selected) if selected in options else 0
        note = f"{len(matches):,} match(es)" + (
            f" (showing the first {MAX_SEARCH_RESULTS:,}; type more characters to narrow down)"
            if len(matches) > MAX_SEARCH_RESULTS else "")
        d1 = st.container()
    else:
        d1, d2 = st.columns([5, 1])
        with d2:
            page = st.number_input("Go to page", min_value=1, max_value=n_pages, step=1,
                                   key=page_key, disabled=(n_pages == 1))
            st.markdown(f"<div style='font-weight:700;color:#4F46E5;margin-top:-.4rem'>"
                        f"Page {page:,} of {n_pages:,}</div>", unsafe_allow_html=True)
        start = (page - 1) * PAGE_SIZE
        options = customer_ids[start:start + PAGE_SIZE]
        idx = options.index(selected) if selected in options else 0
        note = (f"Page {page:,} of {n_pages:,} · "
                f"customers {start + 1:,}–{start + len(options):,} of {total:,}")

    with d1:
        choice = st.selectbox(f"Select a customer · {note}", options, index=idx)
    st.session_state.selected_customer = choice
    return choice


# ============================================================
# Sidebar
# ============================================================
inject_css()

PAGES = {
    "🏠  Executive Dashboard": "exec",
    "👤  Customer Analytics": "customer",
    "🎁  Recommendation Engine": "rec",
    "📈  Business Insights": "insights",
    "🔍  Explainability": "explain",
}

with st.sidebar:
    st.markdown(
        "<div class='brand'><div class='brand-logo'>🛍️</div>"
        "<div><div class='brand-title'>Customer Analytics</div>"
        "<div class='brand-sub'>E-Commerce Platform</div></div></div>",
        unsafe_allow_html=True,
    )
    page_label = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]


# ============================================================
# PAGE: Executive Dashboard
# ============================================================
if page == "exec":
    hero("🏠", "Executive Dashboard", "A quick pulse on revenue, customers and orders.")

    df = load_transactions()
    order_level = df.drop_duplicates("order_id")

    total_revenue = df["price"].sum()
    total_customers = df["customer_unique_id"].nunique()
    total_orders = df["order_id"].nunique()
    avg_order_value = order_level["payment_value"].sum() / total_orders

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card("Total Revenue", money(total_revenue), "💰", "#4F46E5"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Total Customers", f"{total_customers:,}", "👥", "#14B8A6"), unsafe_allow_html=True)
    k3.markdown(kpi_card("Orders", f"{total_orders:,}", "📦", "#F59E0B"), unsafe_allow_html=True)
    k4.markdown(kpi_card("Avg Order Value", f"R$ {avg_order_value:,.2f}", "🧾", "#EC4899"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Auto-generated takeaways
    monthly = df.groupby("order_month")["price"].sum().reset_index(name="revenue").sort_values("order_month")
    peak = monthly.loc[monthly["revenue"].idxmax()]
    orders_per_cust = df.groupby("customer_unique_id")["order_id"].nunique()
    repeat_share = (orders_per_cust > 1).mean() * 100
    top_cat = df.groupby("product_category_name_english")["price"].sum().idxmax().replace("_", " ").title()

    i1, i2, i3 = st.columns(3)
    i1.markdown(f"<div class='insight'>📅 <b>Peak month</b><br>{peak['order_month']} brought in "
                f"<b>{money(peak['revenue'])}</b>.</div>", unsafe_allow_html=True)
    i2.markdown(f"<div class='insight'>🔁 <b>Repeat customers</b><br><b>{repeat_share:.1f}%</b> of customers "
                f"ordered more than once.</div>", unsafe_allow_html=True)
    i3.markdown(f"<div class='insight'>🏆 <b>Top category</b><br><b>{top_cat}</b> leads on revenue.</div>",
                unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        fig = px.area(monthly, x="order_month", y="revenue", title="Monthly revenue",
                      labels={"order_month": "Month", "revenue": "Revenue (R$)"})
        fig.update_traces(line_color="#4F46E5", fillcolor="rgba(79,70,229,0.18)")
        st.plotly_chart(style_fig(fig, 360))
    with right:
        pay = order_level["payment_type"].value_counts().reset_index()
        pay.columns = ["payment_type", "orders"]
        fig = px.pie(pay, names="payment_type", values="orders", hole=0.55, title="Payment methods")
        st.plotly_chart(style_fig(fig, 360))

    left, right = st.columns(2)
    with left:
        reviewed = order_level.dropna(subset=["review_score"])
        rs = reviewed["review_score"].value_counts().sort_index().reset_index()
        rs.columns = ["score", "reviews"]
        fig = px.bar(rs, x="score", y="reviews", title=f"Review scores (avg {reviewed['review_score'].mean():.2f} / 5)",
                     labels={"score": "Stars", "reviews": "Reviews"})
        fig.update_traces(marker_color="#14B8A6")
        st.plotly_chart(style_fig(fig, 330))
    with right:
        st_rev = df.groupby("customer_state")["price"].sum().sort_values(ascending=False).head(8).reset_index()
        st_rev["state"] = st_rev["customer_state"].map(state_name)
        fig = px.bar(st_rev, x="price", y="state", orientation="h", title="Top 8 states by revenue",
                     labels={"price": "Revenue (R$)", "state": ""})
        fig.update_traces(marker_color="#F59E0B")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, 330))


# ============================================================
# PAGE: Customer Analytics
# ============================================================
elif page == "customer":
    hero("👤", "Customer Analytics", "Look up any customer to see their segment, repeat-purchase chance and expected value.")

    purchase_df = load_csv("purchase_prediction_dataset.csv", "06_purchase_prediction.ipynb")
    segments_df = load_csv("customer_segments.csv", "05_segmentation.ipynb")
    purchase_model, clv_model = load_models()

    selected = customer_selector(sorted(purchase_df["customer_unique_id"].unique()), key="cust")

    if selected:
        row = purchase_df[purchase_df["customer_unique_id"] == selected].iloc[0]
        X_row = pd.DataFrame([row[FEATURE_COLS]])

        purchase_proba = float(purchase_model.predict_proba(X_row)[0, 1])
        clv_pred = max(0.0, float(clv_model.predict(X_row)[0]))
        base_rate = float(purchase_df["repeat_purchase"].mean())

        match = segments_df[segments_df["customer_unique_id"] == selected]
        persona = match["persona"].iloc[0] if len(match) else None

        st.markdown(f"#### Customer `{selected}`")
        c1, c2, c3 = st.columns([1.2, 1.3, 1.2])

        with c1:
            if persona in PERSONAS:
                icon, color, desc = PERSONAS[persona]
                st.markdown(
                    f"<div class='persona-box' style='background:linear-gradient(135deg,{color},#0B1F3A)'>"
                    f"<div style='font-size:2rem'>{icon}</div><h3>{persona}</h3><p>{desc}</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='card'><b>Segment</b><br>Not available. This customer isn't in the "
                            "Phase 5 output.</div>", unsafe_allow_html=True)

        with c2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=purchase_proba * 100, number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Chance of buying again (90 days)", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4F46E5"},
                    "steps": [{"range": [0, 100], "color": "#EEF2FF"}],
                    "threshold": {"line": {"color": "#EF4444", "width": 4}, "thickness": 0.8, "value": base_rate * 100},
                },
            ))
            gauge.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter, sans-serif"))
            st.plotly_chart(gauge)
            ratio = purchase_proba / base_rate if base_rate > 0 else np.nan
            st.caption(f"Red marker = average customer ({base_rate:.1%}). "
                       + (f"This customer is **{ratio:.1f}x** the average." if np.isfinite(ratio) else ""))

        with c3:
            st.markdown(kpi_card("Predicted Lifetime Value", f"R$ {clv_pred:,.2f}", "💎", "#14B8A6",
                                 "Expected spend, next 90 days"), unsafe_allow_html=True)
            st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            st.markdown(kpi_card("Spent So Far", f"R$ {row['monetary_value']:,.2f}", "🧾", "#F59E0B",
                                 f"{int(row['frequency'])} order(s) to date"), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Customer profile")
        p1, p2, p3, p4 = st.columns(4)
        p1.markdown(kpi_card("Recency", f"{int(row['recency_days'])} days", "⏱️", "#4F46E5", "Since last order"), unsafe_allow_html=True)
        p2.markdown(kpi_card("Tenure", f"{int(row['customer_tenure_days'])} days", "📆", "#14B8A6", "Since first order"), unsafe_allow_html=True)
        p3.markdown(kpi_card("Avg Order Value", f"R$ {row['avg_order_value']:,.2f}", "💳", "#F59E0B"), unsafe_allow_html=True)
        p4.markdown(kpi_card("Avg Basket Size", f"{row['avg_basket_size']:.2f} items", "🧺", "#EC4899"), unsafe_allow_html=True)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        rating = f"{row['avg_review_rating']:.2f} / 5" if pd.notna(row["avg_review_rating"]) else "N/A"
        q1.markdown(kpi_card("Avg Review Rating", rating, "⭐", "#F59E0B"), unsafe_allow_html=True)
        q2.markdown(kpi_card("Preferred Payment", str(row["preferred_payment_method"]).replace("_", " ").title(), "💰", "#4F46E5"), unsafe_allow_html=True)
        q3.markdown(kpi_card("State", state_name(row["customer_state_grouped"]), "📍", "#14B8A6"), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        with st.expander("📊 How does this customer compare with everyone else?", expanded=True):
            metrics = {"Frequency": "frequency", "Total spend": "monetary_value",
                       "Avg order value": "avg_order_value", "Tenure": "customer_tenure_days"}
            def midrank_pct(series, value, higher_is_better=True):
                """Share of customers this one beats; ties count as half, so a
                value shared by most customers (e.g. 1 order) lands mid-chart
                instead of at 0% or 100%."""
                beaten = (series < value) if higher_is_better else (series > value)
                tied = series == value
                return float((beaten.mean() + 0.5 * tied.mean()) * 100)

            fmt = {"frequency": lambda v: f"{int(v)} order(s)",
                   "monetary_value": lambda v: f"R$ {v:,.0f}",
                   "avg_order_value": lambda v: f"R$ {v:,.0f}",
                   "customer_tenure_days": lambda v: f"{int(v)} days"}
            rows = [{"metric": label, "percentile": midrank_pct(purchase_df[col], row[col]),
                     "value": fmt[col](row[col])} for label, col in metrics.items()]
            # Recency: fewer days since last order is better, so flip it
            rows.append({"metric": "Recency (more recent = higher)",
                         "percentile": midrank_pct(purchase_df["recency_days"], row["recency_days"], False),
                         "value": f"{int(row['recency_days'])} days ago"})
            comp = pd.DataFrame(rows)
            fig = px.bar(comp, x="percentile", y="metric", orientation="h", range_x=[0, 100], text="value",
                         labels={"percentile": "Ahead of this % of customers", "metric": ""})
            fig.update_traces(marker_color="#4F46E5", textposition="inside", insidetextanchor="start",
                              textfont=dict(color="white"))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, 300))
            st.caption("Many customers share the same value (for example, a single order), so those bars sit "
                       "near the middle rather than at the extremes.")


# ============================================================
# PAGE: Recommendation Engine (optional)
# ============================================================
elif page == "rec":
    hero("🎁", "Recommendation Engine", "Top 5 products a customer is most likely to want next.")

    rec_artifacts = load_rec_artifacts()
    if rec_artifacts is None:
        st.warning("No recommendation artifacts found. Run **08_recommendation.ipynb** first. "
                   "This module is optional, so everything else still works without it.")
        st.stop()

    products = rec_artifacts["products"]
    prod_idx = rec_artifacts["prod_idx"]
    customers_rec = rec_artifacts["customers"]
    content_matrix = rec_artifacts["content_matrix"]
    item_similarity = rec_artifacts["item_similarity"]
    product_meta = rec_artifacts["product_meta"]
    train_purchased = rec_artifacts["train_purchased"]

    selected = customer_selector(list(customers_rec), key="rec")

    if selected:
        purchased = train_purchased.get(selected, set())
        if not purchased:
            st.info("This customer has no history in the popular-product catalog, so there isn't enough "
                    "signal to recommend from (see the sparsity note in Phase 8).")
        else:
            def minmax(arr):
                lo, hi = arr.min(), arr.max()
                return np.zeros_like(arr) if hi - lo < 1e-9 else (arr - lo) / (hi - lo)

            purchased_idxs = [prod_idx[p] for p in purchased if p in prod_idx]
            profile = content_matrix[purchased_idxs].mean(axis=0, keepdims=True)
            content_scores = cosine_similarity(profile, content_matrix)[0]

            user_vector = np.zeros(len(products))
            for p in purchased:
                if p in prod_idx:
                    user_vector[prod_idx[p]] = 1
            cf_scores = user_vector @ item_similarity

            combined = 0.5 * minmax(content_scores) + 0.5 * minmax(cf_scores)
            for p in purchased:
                if p in prod_idx:
                    combined[prod_idx[p]] = -np.inf

            top5_idx = np.argsort(-combined)[:5]
            left, right = st.columns([3, 2])

            with left:
                st.markdown(f"#### Recommended for `{selected}`")
                for rank, idx in enumerate(top5_idx, start=1):
                    pid = products[idx]
                    category = product_meta.loc[pid, "category"] if pid in product_meta.index else "unknown"
                    price = product_meta.loc[pid, "avg_price"] if pid in product_meta.index else np.nan
                    match_pct = max(0.0, min(1.0, float(combined[idx]))) * 100
                    price_txt = f"R$ {price:,.2f}" if pd.notna(price) else "n/a"
                    st.markdown(
                        f"<div class='rec-card'><div class='rec-rank'>{rank}</div>"
                        f"<div style='flex:1'><div class='rec-title'>{pid}</div>"
                        f"<div class='rec-meta'>{str(category).replace('_', ' ').title()} · avg price {price_txt}</div>"
                        f"<div class='match-bar'><div class='match-fill' style='width:{match_pct:.0f}%'></div></div>"
                        f"<div class='rec-meta'>Match score {match_pct:.0f}%</div></div></div>",
                        unsafe_allow_html=True,
                    )

            with right:
                st.markdown("#### Already bought")
                rows = []
                for p in list(purchased)[:8]:
                    cat = product_meta.loc[p, "category"] if p in product_meta.index else "unknown"
                    rows.append({"Product": f"{p[:10]}…", "Category": str(cat).replace("_", " ").title()})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.caption("Recommendations blend category/price similarity with what similar customers bought.")


# ============================================================
# PAGE: Business Insights
# ============================================================
elif page == "insights":
    hero("📈", "Business Insights", "Sales trends, categories, sellers and geography.")

    df = load_transactions()
    seller_features = load_csv("seller_features.csv", "04_feature_engineering.ipynb")

    dmin, dmax = df["order_purchase_timestamp"].min().date(), df["order_purchase_timestamp"].max().date()
    date_range = st.date_input("📅 Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
        mask = (df["order_purchase_timestamp"].dt.date >= start) & (df["order_purchase_timestamp"].dt.date <= end)
        df = df[mask]
    if df.empty:
        st.warning("No orders in that date range.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Trends", "🗂️ Category Performance",
                                      "🏪 Seller Performance", "🗺️ Geographic Sales"])

    with tab1:
        grain = st.radio("View by", ["Daily", "Weekly", "Monthly"], horizontal=True, index=1)
        freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[grain]
        ts = (df.set_index("order_purchase_timestamp")
                .resample(freq).agg(orders=("order_id", "nunique"), revenue=("price", "sum")).reset_index())
        metric = st.radio("Metric", ["Orders", "Revenue"], horizontal=True)
        col = metric.lower()
        fig = px.line(ts, x="order_purchase_timestamp", y=col, title=f"{grain} {metric.lower()}",
                      labels={"order_purchase_timestamp": "Date", col: metric})
        fig.update_traces(line_color="#4F46E5", line_width=3)
        if grain == "Daily":
            ts["rolling_7d"] = ts[col].rolling(7).mean()
            fig.add_scatter(x=ts["order_purchase_timestamp"], y=ts["rolling_7d"], name="7-day average",
                            line=dict(color="#F59E0B", width=3))
        st.plotly_chart(style_fig(fig, 420))

    with tab2:
        c1, c2 = st.columns([1, 1])
        top_n = c1.slider("How many categories?", 5, 30, 12)
        by = c2.radio("Rank by", ["Revenue", "Orders"], horizontal=True)
        cat = (df.groupby("product_category_name_english")
                 .agg(revenue=("price", "sum"), orders=("order_id", "nunique")).reset_index())
        cat["category"] = cat["product_category_name_english"].str.replace("_", " ").str.title()
        cat = cat.sort_values(by.lower(), ascending=False).head(top_n)
        fig = px.bar(cat, x=by.lower(), y="category", orientation="h", color=by.lower(),
                     color_continuous_scale="Viridis", title=f"Top {top_n} categories by {by.lower()}",
                     labels={by.lower(): by, "category": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, max(380, top_n * 30)))

    with tab3:
        c1, c2 = st.columns(2)
        min_orders = c1.slider("Minimum orders per seller", 1, 100, 10)
        top_n_s = c2.slider("Sellers to show", 5, 30, 15)
        sf = seller_features[seller_features["n_orders"] >= min_orders]
        if sf.empty:
            st.info("No sellers meet that minimum. Try lowering it.")
        else:
            top_sellers = sf.sort_values("seller_performance_score", ascending=False).head(top_n_s).copy()
            top_sellers["seller"] = top_sellers["seller_id"].str[:8] + "…"
            fig = px.bar(top_sellers, x="seller_performance_score", y="seller", orientation="h",
                         color="avg_review_score", color_continuous_scale="RdYlGn",
                         title="Top sellers by performance score",
                         labels={"seller_performance_score": "Performance score (0-100)", "seller": "",
                                 "avg_review_score": "Avg review"})
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, max(380, top_n_s * 28)))

            fig2 = px.scatter(sf, x="avg_delivery_time_days", y="avg_review_score", size="n_orders",
                              color="seller_performance_score", color_continuous_scale="Viridis",
                              title="Delivery speed vs. review score (bubble = order volume)",
                              labels={"avg_delivery_time_days": "Avg delivery time (days)",
                                      "avg_review_score": "Avg review score"})
            st.plotly_chart(style_fig(fig2, 400))

    with tab4:
        state_rev = (df.groupby("customer_state")
                       .agg(revenue=("price", "sum"), orders=("order_id", "nunique")).reset_index()
                       .sort_values("revenue", ascending=False))
        state_rev["state"] = state_rev["customer_state"].map(state_name)
        geo = load_brazil_geojson()
        c1, c2 = st.columns([3, 2])
        with c1:
            if geo is not None:
                # The map still locates states by their official code; hovering shows the full name.
                fig = px.choropleth(state_rev, geojson=geo, locations="customer_state", featureidkey="properties.id",
                                    color="revenue", color_continuous_scale="YlOrRd", title="Revenue by state",
                                    hover_name="state", hover_data={"customer_state": False, "orders": ":,"},
                                    labels={"revenue": "Revenue (R$)", "orders": "Orders"})
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(style_fig(fig, 460))
            else:
                st.info("The map needs internet access to load Brazil's state outlines. Showing bars instead.")
                fig = px.bar(state_rev, x="state", y="revenue", title="Revenue by state",
                             labels={"state": "", "revenue": "Revenue (R$)"})
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(style_fig(fig, 460))
        with c2:
            fig = px.bar(state_rev.head(10), x="revenue", y="state", orientation="h",
                         title="Top 10 states", labels={"revenue": "Revenue (R$)", "state": ""})
            fig.update_traces(marker_color="#F59E0B")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, 460))


# ============================================================
# PAGE: Explainability
# ============================================================
elif page == "explain":
    hero("🔍", "Explainability", "See what drives the models, and why a specific customer got their score.")

    purchase_df = load_csv("purchase_prediction_dataset.csv", "06_purchase_prediction.ipynb")
    clv_df = load_csv("clv_prediction_dataset.csv", "07_clv_prediction.ipynb")
    purchase_model, clv_model = load_models()

    model_choice = st.radio("Which model do you want to explain?",
                            ["Purchase Prediction", "CLV Prediction"], horizontal=True)
    if model_choice == "Purchase Prediction":
        pipeline, mode, source_df = purchase_model, "proba", purchase_df
        st.caption("Predicts the chance a customer buys again within 90 days.")
    else:
        pipeline, mode, source_df = clv_model, "regression", clv_df
        st.caption("Predicts how much a customer will spend over the next 90 days.")

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    st.markdown("### Which features matter most overall?")
    importance = get_feature_importance(pipeline, feature_names)
    if importance is not None:
        top = importance.head(12).reset_index()
        top.columns = ["feature", "importance"]
        fig = px.bar(top, x="importance", y="feature", orientation="h", title="Feature importance (top 12)",
                     labels={"importance": "Importance", "feature": ""})
        fig.update_traces(marker_color="#4F46E5")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, 420))
    else:
        st.info("The best model for this target doesn't expose built-in feature importances. "
                "The SHAP view below works for any model type.")

    st.markdown("### Why did *this* customer get this score?")
    selected = customer_selector(sorted(source_df["customer_unique_id"].unique()), key=f"shap_{mode}")

    if selected:
        with st.spinner("Calculating explanation… the first one can take a few seconds."):
            background_raw = source_df[FEATURE_COLS].sample(min(100, len(source_df)), random_state=42)
            explainer, transform, predict_fn = get_shap_explainer(pipeline, model_choice, mode, background_raw)

            row = source_df[source_df["customer_unique_id"] == selected].iloc[0]
            X_row_transformed = transform(pd.DataFrame([row[FEATURE_COLS]]))
            shap_values = explainer(X_row_transformed)
            pred = float(predict_fn(X_row_transformed)[0])

        pred_label = f"{pred:.1%} chance of repeat purchase" if mode == "proba" \
            else f"R$ {max(0, pred):,.2f} expected spend (90 days)"
        st.markdown(kpi_card("Model prediction", pred_label, "🎯", "#4F46E5"), unsafe_allow_html=True)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        # Plain-language top reasons
        vals = np.array(shap_values.values[0]).ravel()
        order = np.argsort(-np.abs(vals))[:3]
        reasons = []
        for i in order:
            direction = "raises" if vals[i] > 0 else "lowers"
            reasons.append(f"<li><b>{pretty_feature(str(feature_names[i]))}</b> {direction} the prediction</li>")
        st.markdown(f"<div class='insight'><b>Top 3 reasons for this score</b><ul style='margin:.4rem 0 0 0'>"
                    f"{''.join(reasons)}</ul></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        tab_w, tab_f = st.tabs(["🌊 Waterfall plot", "⚡ Force plot"])
        with tab_w:
            shap.plots.waterfall(shap_values[0], show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Starts at the average prediction. Red bars push the score up, blue bars push it down.")
        with tab_f:
            shap.plots.force(shap_values[0], matplotlib=True, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Same idea as the waterfall, shown as a single tug-of-war bar.")

        with st.expander("ℹ️ How to read these charts"):
            st.markdown(
                "- **Feature importance** shows which inputs the model relies on most across *all* customers.\n"
                "- **SHAP plots** explain *one* customer: how much each input moved their score away from the average.\n"
                "- Feature names starting with a category (like payment method or state) are one-hot encoded "
                "versions of those fields."
            )

st.markdown("<div class='footer'>Intelligent E-Commerce Customer Analytics Platform</div>", unsafe_allow_html=True)
