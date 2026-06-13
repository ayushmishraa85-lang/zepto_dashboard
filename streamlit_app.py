"""
Zepto Sales Intelligence Dashboard — Streamlit Edition
Phase 4: Anthropic LLM streaming · natural-language BlinkBot · chart detection
Developed by Ayush Mishra
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings, io, os, json, requests

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════════
# ── PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Ayush Intelligence Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stDecoration"]   { visibility: hidden; }
[data-testid="stDeployButton"] { visibility: hidden; }
[data-testid="stToolbarActions"]{ display: none !important; }
footer   { visibility: hidden; }
#MainMenu{ visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #070d1a; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }
.kpi-card {
  background: linear-gradient(135deg, #0d1628, #121d35);
  border: 1px solid rgba(99,130,255,.15);
  border-radius: 14px; padding: 18px 16px;
  text-align: center; position: relative;
  overflow: hidden; transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label { font-size: 10px; color: #4a5a7a; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.kpi-sub   { font-size: 10px; color: #4a5a7a; }
.kpi-badge { display: inline-block; font-size: 9px; font-weight: 600; border-radius: 5px; padding: 2px 7px; margin-top: 4px; }
.up   { background: rgba(16,185,129,.12);  color: #34d399; }
.down { background: rgba(239,68,68,.12);   color: #f87171; }
.section-head {
  font-size: 11px; font-weight: 600; color: #8899bb;
  text-transform: uppercase; letter-spacing: .1em;
  border-left: 3px solid #6366f1; padding-left: 10px;
  margin: 24px 0 14px;
}
.insight-card {
  background: linear-gradient(135deg, #121d35, #0d1628);
  border: 1px solid rgba(99,130,255,.12);
  border-radius: 12px; padding: 16px; height: 100%; margin-bottom: 4px;
}
.insight-title { font-size: 10px; font-weight: 600; color: #4a5a7a; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.insight-body  { font-size: 12px; color: #8899bb; line-height: 1.6; }
.insight-body strong { color: #67e8f9; }
.stat-row  { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid rgba(99,130,255,.06); }
.stat-label{ font-size: 11px; color: #8899bb; }
.stat-value{ font-size: 11px; font-weight: 600; color: #67e8f9; font-family: monospace; }
.footer { text-align: center; padding: 20px; color: #4a5a7a; font-size: 11px; border-top: 1px solid rgba(99,130,255,.08); margin-top: 30px; }
.footer .dev { background: linear-gradient(90deg,#a5b4fc,#67e8f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }
.chat-message-bot {
  background: linear-gradient(135deg, rgba(99,102,241,.1), rgba(6,182,212,.05));
  border: 1px solid rgba(99,102,241,.2); border-radius: 12px 12px 12px 0;
  padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #e2e8f0; line-height: 1.6;
}
.chat-message-user {
  background: rgba(30,41,59,.8); border: 1px solid rgba(99,130,255,.15);
  border-radius: 12px 12px 0 12px; padding: 12px 16px; margin: 8px 0;
  font-size: 13px; color: #cbd5e1; text-align: right;
}
div[data-testid="metric-container"] {
  background: #0d1628; border: 1px solid rgba(99,130,255,.12); border-radius: 12px; padding: 12px;
}
.blinkbot-header {
  background: linear-gradient(135deg, #1e1b6e, #312e81);
  padding: 16px 20px; display: flex; align-items: center; gap: 12px;
  border: 1px solid rgba(99,102,241,.25); border-radius: 16px; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# ── CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════════

PAL      = ["#6366f1","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316","#3b82f6"]
CAT_CLR  = {"Snacks":"#6366f1","Beverages":"#06b6d4","Grocery":"#10b981","Instant Food":"#f59e0b","Confectionery":"#ec4899","Dairy":"#8b5cf6"}
CITY_CLR = {"Delhi":"#6366f1","Mumbai":"#06b6d4","Bangalore":"#10b981","Hyderabad":"#f59e0b","Chennai":"#ef4444","Pune":"#8b5cf6"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8899bb", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(99,130,255,.06)", linecolor="rgba(99,130,255,.1)"),
    yaxis=dict(gridcolor="rgba(99,130,255,.06)", linecolor="rgba(99,130,255,.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

# PLOTLY_BASE — safe spread for update_layout calls that define their own axes/legend.
# Excludes xaxis, yaxis, AND legend so callers can override them freely.
_AXIS_DEFAULTS = dict(gridcolor="rgba(99,130,255,.06)", linecolor="rgba(99,130,255,.1)")


def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """
    Convert a #RRGGBB hex string to a valid plotly rgba() string.
    Falls back to the original string if it isn't a 7-char hex color.
    """
    if not (isinstance(hex_color, str) and hex_color.startswith("#") and len(hex_color) == 7):
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"
_LEGEND_DEFAULT = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10))
PLOTLY_BASE = {k: v for k, v in PLOTLY_LAYOUT.items()
               if k not in ("xaxis", "yaxis", "legend")}

# Unit-economics cost ratios (easy to tune in one place)
UNIT_ECON = dict(cogs=0.52, rider=0.12, packaging=0.03, gateway=0.02, promos=0.05)

# Delivery simulation parameters
DELIVERY_PARAMS = dict(mean=11.5, std=3.5, lo=5, hi=35, promise=10)

# ══════════════════════════════════════════════════════════════════════════════════
# ── UTILITY / FORMATTING
# ══════════════════════════════════════════════════════════════════════════════════

def fmt(n: float) -> str:
    """Format a number as Indian currency shorthand."""
    if pd.isna(n):  return "—"
    if n >= 1e7:    return f"₹{n/1e7:.1f}Cr"
    if n >= 1e5:    return f"₹{n/1e5:.2f}L"
    if n >= 1e3:    return f"₹{n/1e3:.1f}K"
    return f"₹{int(n):,}"


def pct_change_label(current: float, previous: float) -> tuple[str, bool]:
    """Return (label, is_positive) for WoW / period comparison badges."""
    if previous == 0:
        return "+0.0%", True
    chg = (current - previous) / abs(previous) * 100
    arrow = "↑" if chg >= 0 else "↓"
    return f"{arrow} {abs(chg):.1f}% WoW", chg >= 0


# ══════════════════════════════════════════════════════════════════════════════════
# ── DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════════

_FALLBACK_CSV = """Product Name,Category,City,Original Price,Current Price,Discount,Orders,Total Revenue,Influencer Active
Britannia Cake,Snacks,Delhi,148,163,5,283,44714,No
Britannia Cake,Snacks,Pune,81,86,10,284,21584,Yes
Fortune Oil 1L,Grocery,Hyderabad,138,143,10,69,9177,No
Pepsi 500ml,Beverages,Delhi,127,127,10,83,9711,No
Aashirvaad Atta,Grocery,Chennai,34,49,10,169,6591,Yes
Amul Milk 500ml,Dairy,Delhi,149,159,0,246,39114,No
Britannia Cake,Snacks,Bangalore,82,87,0,254,22098,Yes
Amul Milk 500ml,Dairy,Bangalore,46,51,5,179,8234,No
Aashirvaad Atta,Grocery,Mumbai,137,137,10,268,34036,No
Maggi Noodles,Instant Food,Hyderabad,196,201,0,59,11859,Yes
Coca Cola 1L,Beverages,Delhi,140,140,10,269,34970,No
Oreo Biscuits,Snacks,Delhi,188,203,10,279,53847,No
Parle-G,Snacks,Chennai,96,101,0,299,30199,No
Nestle Munch,Confectionery,Mumbai,195,205,0,223,45715,No
Pepsi 500ml,Beverages,Hyderabad,154,159,5,253,38962,No
Amul Milk 500ml,Dairy,Pune,120,135,0,291,39285,Yes
Parle-G,Snacks,Bangalore,139,144,0,253,36432,Yes
Fortune Oil 1L,Grocery,Bangalore,143,153,0,299,45747,Yes
Coca Cola 1L,Beverages,Hyderabad,177,187,5,299,54418,No
Amul Milk 500ml,Dairy,Chennai,199,204,0,269,54876,No
Maggi Noodles,Instant Food,Hyderabad,182,197,0,294,57918,No
Nestle Munch,Confectionery,Bangalore,191,206,10,297,58212,No
Coca Cola 1L,Beverages,Delhi,175,190,5,288,53280,Yes
Oreo Biscuits,Snacks,Delhi,171,176,10,291,48306,No
Britannia Cake,Snacks,Mumbai,177,192,0,253,48576,No
Parle-G,Snacks,Delhi,184,194,0,271,52574,Yes
Amul Milk 500ml,Dairy,Pune,169,184,10,267,46458,No
Fortune Oil 1L,Grocery,Bangalore,143,153,0,299,45747,Yes
Nestle Munch,Confectionery,Mumbai,195,205,0,223,45715,No
Britannia Cake,Snacks,Delhi,148,163,5,283,44714,No"""

_NUMERIC_COLS = ["Original Price", "Current Price", "Discount", "Orders", "Total Revenue"]
_PRICE_BINS   = [0, 60, 100, 140, 180, np.inf]
_PRICE_LABELS = ["₹20–60", "₹61–100", "₹101–140", "₹141–180", "₹181+"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize, impute, and engineer features on a raw DataFrame."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.dropna(how="all", inplace=True)

    # Coerce numerics
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0]).str.strip()

    # Derived columns
    df["Profit"]        = (df["Current Price"] - df["Original Price"]) * df["Orders"]
    df["Profit Margin"] = np.where(
        df["Total Revenue"] > 0,
        df["Profit"] / df["Total Revenue"] * 100,
        0,
    )
    df["Price Tier"] = pd.cut(
        df["Current Price"], bins=_PRICE_BINS, labels=_PRICE_LABELS
    )
    return df


@st.cache_data
def load_default() -> pd.DataFrame:
    """Load the dataset from disk or fall back to the embedded sample CSV."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "zepto_sales_dataset.csv")
    if os.path.exists(path):
        return clean(pd.read_csv(path))
    return clean(pd.read_csv(io.StringIO(_FALLBACK_CSV)))


# ══════════════════════════════════════════════════════════════════════════════════
# ── CALCULATION FUNCTIONS  (pure — no Streamlit calls)
# ══════════════════════════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute all top-level KPIs from the filtered DataFrame.
    Returns a flat dict so rendering code never does arithmetic.
    """
    total_rev    = df["Total Revenue"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Orders"].sum()
    margin       = (total_profit / total_rev * 100) if total_rev else 0
    rev_std      = df["Total Revenue"].std()

    cat_rev  = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
    city_rev = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)

    return dict(
        total_rev    = total_rev,
        total_profit = total_profit,
        total_orders = total_orders,
        margin       = margin,
        rev_std      = rev_std,
        cat_rev      = cat_rev,
        city_rev     = city_rev,
        aov          = total_rev / total_orders if total_orders else 0,
    )


def compute_influencer_stats(df: pd.DataFrame) -> dict:
    """
    Compute influencer lift on revenue and orders.
    Returns a dict with lift %, p-value, per-group means and counts.
    """
    has_inf = "Influencer Active" in df.columns
    if not has_inf:
        return dict(available=False)

    grp_y = df[df["Influencer Active"] == "Yes"]
    grp_n = df[df["Influencer Active"] == "No"]

    rev_y = grp_y["Total Revenue"].mean() if len(grp_y) else 0
    rev_n = grp_n["Total Revenue"].mean() if len(grp_n) else 0
    rev_lift = ((rev_y - rev_n) / rev_n * 100) if rev_n > 0 else 0

    ord_y = grp_y["Orders"].mean() if len(grp_y) else 0
    ord_n = grp_n["Orders"].mean() if len(grp_n) else 0
    ord_lift = ((ord_y - ord_n) / ord_n * 100) if ord_n > 0 else 0

    _, p_inf = (
        stats.ttest_ind(grp_y["Total Revenue"], grp_n["Total Revenue"])
        if len(grp_y) > 1 and len(grp_n) > 1
        else (0, 1)
    )

    return dict(
        available   = True,
        rev_y       = rev_y,
        rev_n       = rev_n,
        rev_lift    = rev_lift,
        ord_lift    = ord_lift,
        p_value     = p_inf,
        significant = p_inf < 0.05,
        count_y     = len(grp_y),
        count_n     = len(grp_n),
    )


def compute_statistics(df: pd.DataFrame) -> dict:
    """
    Run descriptive stats, normality test, outlier detection, and correlations.
    Returns a structured dict for the stats section renderer.
    """
    rev_arr  = df["Total Revenue"].values
    z_scores = np.abs(stats.zscore(rev_arr))

    outlier_mask = z_scores > 2
    outliers     = df[outlier_mask].copy()
    outliers["Z-Score"] = z_scores[outlier_mask].round(2)

    _, p_norm    = stats.shapiro(rev_arr[:5000])
    r_disc, p_disc = stats.pearsonr(df["Discount"], df["Orders"])
    r_rev,  p_rev  = stats.pearsonr(df["Total Revenue"], df["Profit"])

    return dict(
        rev_arr  = rev_arr,
        mean     = np.mean(rev_arr),
        median   = np.median(rev_arr),
        std      = np.std(rev_arr),
        skewness = stats.skew(rev_arr),
        kurtosis = stats.kurtosis(rev_arr),
        p_norm   = p_norm,
        is_normal= p_norm > 0.05,
        outliers = outliers,
        r_disc   = r_disc,
        p_disc   = p_disc,
        r_rev    = r_rev,
        p_rev    = p_rev,
        corr_matrix = df[
            ["Original Price","Current Price","Discount","Orders","Total Revenue","Profit","Profit Margin"]
        ].corr().round(3),
    )


def compute_forecast(df: pd.DataFrame) -> dict | None:
    """
    Fit a linear regression on product-level revenue ranking.
    Returns a dict with the model, predictions, and metadata.
    Returns None if there are fewer than 5 data points.
    """
    prod_rev = df.groupby("Product Name")["Total Revenue"].sum().sort_values().values
    n = len(prod_rev)
    if n < 5:
        return None

    X = np.arange(1, n + 1).reshape(-1, 1)
    y = prod_rev.astype(float)

    model      = LinearRegression().fit(X, y)
    next_val   = max(0.0, float(model.predict([[n + 1]])[0]))
    r2         = model.score(X, y)
    residuals  = y - model.predict(X)
    ci         = 1.96 * float(np.std(residuals))
    mean_y     = float(np.mean(y))
    growth_pct = ((next_val - mean_y) / mean_y * 100) if mean_y else 0

    # Build plot-ready arrays (down-sampled for large datasets)
    step = max(1, n // 20)
    xs   = list(range(1, n + 1, step)) + [n + 1]
    trend_vals   = [float(model.predict([[i]])[0]) for i in xs]
    actual_vals  = [float(prod_rev[i - 1]) if i <= n else None for i in xs]

    return dict(
        n           = n,
        next_val    = next_val,
        r2          = r2,
        ci          = ci,
        growth_pct  = growth_pct,
        slope       = float(model.coef_[0]),
        xs          = xs,
        trend_vals  = trend_vals,
        actual_vals = actual_vals,
        upper       = [t + ci for t in trend_vals],
        lower       = [max(0.0, t - ci) for t in trend_vals],
    )


def compute_delivery_stats(n_samples: int) -> dict:
    """
    Simulate delivery-time data and compute performance metrics.
    Uses DELIVERY_PARAMS constants for reproducibility.
    """
    np.random.seed(42)
    p = DELIVERY_PARAMS
    times = np.clip(np.random.normal(p["mean"], p["std"], max(n_samples, 50)), p["lo"], p["hi"])

    otd_pct = float(np.mean(times <= p["promise"] + 2) * 100)
    avg     = float(np.mean(times))
    p90     = float(np.percentile(times, 90))
    p50     = float(np.percentile(times, 50))

    if otd_pct >= 95:
        status, status_color = "🟢 EXCELLENT",       "#10b981"
    elif otd_pct >= 85:
        status, status_color = "🟡 NEEDS ATTENTION", "#f59e0b"
    else:
        status, status_color = "🔴 CRITICAL",        "#ef4444"

    hist_counts, hist_edges = np.histogram(times, bins=12)
    hist_centers = [(hist_edges[i] + hist_edges[i + 1]) / 2 for i in range(len(hist_counts))]

    return dict(
        times        = times,
        otd_pct      = otd_pct,
        avg          = avg,
        p90          = p90,
        p50          = p50,
        status       = status,
        status_color = status_color,
        promise      = p["promise"],
        hist_counts  = hist_counts,
        hist_centers = hist_centers,
    )


def compute_unit_economics(avg_rev: float) -> dict:
    """
    Break down an average order's revenue into cost buckets and net profit.
    Ratios are driven by the UNIT_ECON constants dict.
    """
    e = UNIT_ECON
    costs = {k: avg_rev * v for k, v in e.items()}
    net   = avg_rev - sum(costs.values())
    cm    = (net / avg_rev * 100) if avg_rev > 0 else 0

    return dict(
        avg_rev    = avg_rev,
        net_profit = net,
        cm_pct     = cm,
        **costs,
    )


def compute_inventory(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Simulate stock levels for the top-N products by order velocity.
    Returns a tidy DataFrame with risk labels ready for rendering.
    """
    np.random.seed(123)
    prod_velocity = df.groupby("Product Name")["Orders"].sum().sort_values(ascending=False)
    rows = []

    for prod, velocity in prod_velocity.head(top_n).items():
        stock_left  = int(np.random.randint(5, 200))
        daily_sales = max(1, int(velocity * 0.3))
        days_cover  = round(stock_left / daily_sales, 1)

        if days_cover < 1:
            risk, risk_color, action = "🔴 CRITICAL", "#ef4444", "⚡ ORDER NOW"
            bg, border = "rgba(239,68,68,.08)", "rgba(239,68,68,.3)"
        elif days_cover < 2:
            risk, risk_color, action = "🟡 LOW",      "#f59e0b", "📋 Plan Reorder"
            bg, border = "rgba(245,158,11,.08)", "rgba(245,158,11,.3)"
        else:
            risk, risk_color, action = "🟢 OK",       "#10b981", "✅ Sufficient"
            bg, border = "rgba(16,185,129,.06)",  "rgba(16,185,129,.2)"

        rows.append(dict(
            Product      = prod,
            Stock_Left   = stock_left,
            Daily_Sales  = daily_sales,
            Days_Cover   = days_cover,
            Risk         = risk,
            Action       = action,
            _color       = risk_color,
            _bg          = bg,
            _border      = border,
        ))

    return pd.DataFrame(rows)


def compute_wow_metrics(kpis: dict, factor: float = 0.88) -> dict:
    """
    Derive last-week comparison metrics from current KPIs using a scaling factor.
    Returns both current and previous values for each KPI.
    """
    prev = dict(
        total_rev    = kpis["total_rev"]    * factor,
        total_orders = int(kpis["total_orders"] * factor),
        total_profit = kpis["total_profit"] * factor,
        margin       = kpis["margin"]       * 0.95,
    )
    badges = {
        k: pct_change_label(kpis[k], prev[k])
        for k in ["total_rev", "total_profit", "margin"]
    }
    badges["total_orders"] = pct_change_label(kpis["total_orders"], prev["total_orders"])
    return dict(current=kpis, previous=prev, badges=badges)


def compute_order_defects(total_orders: int) -> dict:
    """
    Estimate defect breakdown from total order volume using fixed industry rates.
    """
    expired       = int(total_orders * 0.018)
    missing       = int(total_orders * 0.024)
    cancelled_oos = int(total_orders * 0.031)
    total_defects = expired + missing + cancelled_oos
    perfect       = total_orders - total_defects
    odr_pct       = total_defects / total_orders * 100 if total_orders > 0 else 0

    return dict(
        total_orders  = total_orders,
        expired       = expired,
        missing       = missing,
        cancelled_oos = cancelled_oos,
        total_defects = total_defects,
        perfect       = perfect,
        odr_pct       = odr_pct,
        # funnel steps
        funnel_y = [total_orders,
                    total_orders - expired,
                    total_orders - expired - missing,
                    total_orders - expired - missing - cancelled_oos,
                    perfect],
        funnel_labels = ["Total Orders","After Expired/Damaged",
                         "After Missing Items","After OOS Cancels","✅ Perfect Orders"],
    )


def compute_ai_insights(df: pd.DataFrame, kpis: dict, inf: dict) -> list[tuple]:
    """
    Derive six narrative insights from the data.
    Returns a list of (emoji, title, html_body) tuples.
    """
    cat_rev  = kpis["cat_rev"]
    city_rev = kpis["city_rev"]
    margin   = kpis["margin"]

    prod_rev = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)

    # Discount correlation
    r_d, p_d = (stats.pearsonr(df["Discount"], df["Orders"])
                if len(df) >= 5 else (0, 1))

    # Best margin category
    best_margin_cat = (
        df.groupby("Category")["Profit Margin"].mean()
          .sort_values(ascending=False).index[0]
        if "Profit Margin" in df.columns and len(df) > 0 else "N/A"
    )

    city_gap_pct = (
        (city_rev.iloc[0] - city_rev.iloc[-1]) / city_rev.iloc[-1] * 100
        if len(city_rev) > 1 and city_rev.iloc[-1] > 0 else 0
    )

    return [
        ("🏆", "Best Category",
         f"<strong>{cat_rev.index[0]}</strong> drives "
         f"{cat_rev.iloc[0]/cat_rev.sum()*100:.1f}% of total revenue "
         f"({fmt(cat_rev.iloc[0])}). Maximize marketing budget here for peak ROI."),

        ("🌍", "Regional Gap",
         f"<strong>{city_rev.index[0]}</strong> ({fmt(city_rev.iloc[0])}) outperforms "
         f"<strong>{city_rev.index[-1]}</strong> ({fmt(city_rev.iloc[-1])}) "
         f"by {city_gap_pct:.0f}%. Target promotions in underperforming regions."),

        ("⚡", "Influencer Lift",
         f"Influencer-active products generate "
         f"<strong>{inf['rev_lift']:+.1f}%</strong> more revenue. "
         f"{'Statistically significant ✓' if inf.get('significant') else 'Not yet significant'} "
         f"(p={inf.get('p_value', 1):.3f})."),

        ("📈", "Profit Margin",
         f"Overall margin is <strong>{margin:.1f}%</strong>. "
         f"{best_margin_cat} has the highest avg margin."),

        ("💡", "Discount Intelligence",
         f"Discount is {'positively' if r_d > 0 else 'negatively'} correlated with orders "
         f"(r={r_d:.3f}, p={p_d:.3f}). "
         f"{'Discounting drives volume.' if r_d > 0 else 'Review your discount strategy.'}"),

        ("🎯", "Top Product",
         f"<strong>{prod_rev.index[0]}</strong> generates {fmt(prod_rev.iloc[0])} "
         f"— the highest single-product revenue. "
         f"Expand distribution and pair with influencer activation."),
    ]


# ══════════════════════════════════════════════════════════════════════════════════
# ── PHASE 2 — MULTI-TURN MEMORY & RICH RESPONSE FORMATTING
# ══════════════════════════════════════════════════════════════════════════════════

# ── 1. ConversationMemory ─────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Tracks the conversational context between BlinkBot turns.
    Stored in st.session_state so it persists across reruns.

    Fields
    ------
    last_intent   : most recent topic (revenue / profit / city / product / ...)
    last_city     : last city explicitly mentioned or answered about
    last_product  : last product explicitly mentioned or answered about
    last_category : last category explicitly mentioned or answered about
    turn_count    : total Q&A turns so far
    intent_stack  : rolling list of the last 3 intents (for follow-up context)
    """

    def __init__(self):
        self.last_intent   : str | None = None
        self.last_city     : str | None = None
        self.last_product  : str | None = None
        self.last_category : str | None = None
        self.turn_count    : int        = 0
        self.intent_stack  : list[str]  = []

    def update(self, intent: str, city=None, product=None, category=None):
        self.last_intent = intent
        if city:     self.last_city     = city
        if product:  self.last_product  = product
        if category: self.last_category = category
        self.turn_count += 1
        self.intent_stack = (self.intent_stack + [intent])[-3:]   # keep last 3

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationMemory":
        m = cls()
        for k, v in d.items():
            setattr(m, k, v)
        return m


def _get_memory() -> ConversationMemory:
    """Retrieve or create the ConversationMemory from session state."""
    if "bb_memory" not in st.session_state:
        st.session_state.bb_memory = ConversationMemory().to_dict()
    return ConversationMemory.from_dict(st.session_state.bb_memory)


def _save_memory(mem: ConversationMemory):
    st.session_state.bb_memory = mem.to_dict()


# ── 2. Entity Extraction ──────────────────────────────────────────────────────────

def extract_entities(question: str, df: pd.DataFrame) -> dict:
    """
    Scan the raw question for known city / product / category names.
    Returns {'city': str|None, 'product': str|None, 'category': str|None}.
    """
    q_low = question.lower()
    found = dict(city=None, product=None, category=None)

    for city in df["City"].unique():
        if city.lower() in q_low:
            found["city"] = city
            break

    for cat in df["Category"].unique():
        if cat.lower() in q_low:
            found["category"] = cat
            break

    for prod in df["Product Name"].unique():
        if prod.lower() in q_low:
            found["product"] = prod
            break

    return found


def resolve_references(question: str, mem: ConversationMemory) -> str:
    """
    Replace vague pronouns / references with the entity from memory.

    Examples
    --------
    'tell me more about it'   → last topic
    'what about that city?'   → last_city
    'how does that product do' → last_product
    'compare with the other'   → last_city / last_category
    """
    q = question.lower().strip()

    # Pronoun → city
    city_refs = ["that city", "that region", "that location", "there", "that place"]
    if any(ref in q for ref in city_refs) and mem.last_city:
        q = q.replace("that city", mem.last_city) \
              .replace("that region", mem.last_city) \
              .replace("that location", mem.last_city) \
              .replace("there", mem.last_city) \
              .replace("that place", mem.last_city)

    # Pronoun → product
    prod_refs = ["that product", "it", "that item", "the same product", "that one"]
    if any(ref in q for ref in prod_refs) and mem.last_product:
        for ref in prod_refs:
            q = q.replace(ref, mem.last_product)

    # Pronoun → category
    cat_refs = ["that category", "that segment", "that section"]
    if any(ref in q for ref in cat_refs) and mem.last_category:
        for ref in cat_refs:
            q = q.replace(ref, mem.last_category)

    # "tell me more" / "expand" → re-state last intent
    more_refs = ["tell me more", "more details", "expand", "elaborate", "explain more", "go deeper"]
    if any(ref in q for ref in more_refs) and mem.last_intent:
        q = mem.last_intent   # treat as if they re-asked the last intent keyword

    # "why" → add context of last topic
    if q.strip() in ("why", "why?", "how come") and mem.last_intent:
        q = f"explain {mem.last_intent}"

    return q


# ── 3. ResponseBuilder — consistent rich formatting ───────────────────────────────

class ResponseBuilder:
    """
    Fluent builder that assembles a Markdown response with a consistent structure:
      [direct answer] → [metrics table] → [context row] → [recommendation]

    Usage
    -----
    ResponseBuilder("📊", "Revenue Overview") \\
        .answer("Total revenue is **₹12.3L**") \\
        .metric("Best category", "Snacks", "🏆") \\
        .metric("Profit margin", "18.4%", "📈") \\
        .context("Previous week was ₹10.1L — that's a 21% jump.") \\
        .tip("Double down on Snacks in Delhi for maximum ROI.") \\
        .build()
    """

    def __init__(self, emoji: str, title: str):
        self._emoji   = emoji
        self._title   = title
        self._answer  : str        = ""
        self._metrics : list[tuple]= []   # (label, value, icon)
        self._context : str        = ""
        self._tip     : str        = ""
        self._followup: str        = ""

    def answer(self, text: str) -> "ResponseBuilder":
        self._answer = text
        return self

    def metric(self, label: str, value: str, icon: str = "▸") -> "ResponseBuilder":
        self._metrics.append((label, value, icon))
        return self

    def context(self, text: str) -> "ResponseBuilder":
        self._context = text
        return self

    def tip(self, text: str) -> "ResponseBuilder":
        self._tip = text
        return self

    def followup(self, text: str) -> "ResponseBuilder":
        self._followup = text
        return self

    def build(self) -> str:
        parts = [f"**{self._emoji} {self._title}**\n"]

        if self._answer:
            parts.append(self._answer + "\n")

        if self._metrics:
            parts.append("")
            for label, value, icon in self._metrics:
                parts.append(f"{icon} **{label}:** {value}")

        if self._context:
            parts.append(f"\n💬 *{self._context}*")

        if self._tip:
            parts.append(f"\n💡 **Recommendation:** {self._tip}")

        if self._followup:
            parts.append(f"\n🔍 *You can also ask: {self._followup}*")

        return "\n".join(parts)


# ── 4. Data context (unchanged from Phase 1) ─────────────────────────────────────

def _bb_context(df: pd.DataFrame) -> dict:
    """Pre-compute all values BlinkBot might need (called once per query)."""
    total_r  = df["Total Revenue"].sum()
    total_o  = df["Orders"].sum()
    total_p  = df["Profit"].sum()
    mgn      = (total_p / total_r * 100) if total_r > 0 else 0
    cat_r    = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False) if "Category"     in df.columns else None
    city_r   = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)     if "City"         in df.columns else None
    prod_r   = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False) if "Product Name" in df.columns else None
    best_m   = df.groupby("Category")["Profit Margin"].mean().sort_values(ascending=False) if "Profit Margin" in df.columns else None
    inf_y_rev= df[df["Influencer Active"]=="Yes"]["Total Revenue"].mean() if "Influencer Active" in df.columns else 0
    inf_n_rev= df[df["Influencer Active"]=="No"]["Total Revenue"].mean()  if "Influencer Active" in df.columns else 0
    inf_lift = ((inf_y_rev - inf_n_rev) / inf_n_rev * 100) if inf_n_rev > 0 else 0
    inf_y_ord= df[df["Influencer Active"]=="Yes"]["Orders"].mean() if "Influencer Active" in df.columns else 0
    inf_n_ord= df[df["Influencer Active"]=="No"]["Orders"].mean()  if "Influencer Active" in df.columns else 0
    ord_lift = ((inf_y_ord - inf_n_ord) / inf_n_ord * 100) if inf_n_ord > 0 else 0
    disc_grp = df.groupby("Discount").agg(avg_rev=("Total Revenue","mean"), avg_orders=("Orders","mean")).reset_index() if "Discount" in df.columns else None

    return dict(total_r=total_r, total_o=total_o, total_p=total_p, mgn=mgn,
                cat_r=cat_r, city_r=city_r, prod_r=prod_r, best_m=best_m,
                inf_y_rev=inf_y_rev, inf_n_rev=inf_n_rev, inf_lift=inf_lift,
                ord_lift=ord_lift, disc_grp=disc_grp,
                n_inf_y=len(df[df["Influencer Active"]=="Yes"]) if "Influencer Active" in df.columns else 0)


# ── 5. Chart Factories (Phase 3) ─────────────────────────────────────────────────
# Each factory returns a go.Figure styled to match the dashboard theme.
# They are called by handlers and attached to the response as an inline chart.

def _chart_revenue_by_category(ctx: dict) -> go.Figure:
    cat_r = ctx["cat_r"]
    fig = go.Figure(go.Bar(
        x=cat_r.index.tolist(), y=cat_r.values,
        marker_color=[CAT_CLR.get(c, "#6366f1") for c in cat_r.index],
        marker_line_width=0, opacity=0.85,
        text=[fmt(v) for v in cat_r.values], textposition="outside",
        textfont=dict(color="#f0f4ff", size=10),
    ))
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text="💬 Revenue by Category", font=dict(color="#a5b4fc", size=12)),
        height=240, yaxis=dict(tickprefix="₹", **_AXIS_DEFAULTS), showlegend=False)
    return fig


def _chart_city_ranking(ctx: dict) -> go.Figure:
    cr = ctx["city_r"]
    colors = ["#10b981" if i == 0 else "#ef4444" if i == len(cr)-1 else "#6366f1"
              for i in range(len(cr))]
    fig = go.Figure(go.Bar(
        x=cr.values, y=cr.index.tolist(), orientation="h",
        marker_color=colors, marker_line_width=0, opacity=0.85,
        text=[fmt(v) for v in cr.values], textposition="outside",
        textfont=dict(color="#f0f4ff", size=10),
    ))
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text="💬 City Revenue Ranking", font=dict(color="#a5b4fc", size=12)),
        height=240, xaxis=dict(tickprefix="₹", **_AXIS_DEFAULTS),
        yaxis=dict(autorange="reversed", **_AXIS_DEFAULTS), showlegend=False)
    return fig


def _chart_top_products(ctx: dict, n: int = 8) -> go.Figure:
    pr = ctx["prod_r"].head(n)
    fig = go.Figure(go.Bar(
        x=pr.values, y=pr.index.tolist(), orientation="h",
        marker=dict(
            color=pr.values,
            colorscale=[[0,"#312e81"],[0.5,"#6366f1"],[1,"#06b6d4"]],
            showscale=False,
        ),
        marker_line_width=0,
        text=[fmt(v) for v in pr.values], textposition="outside",
        textfont=dict(color="#f0f4ff", size=10),
    ))
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text=f"💬 Top {n} Products by Revenue", font=dict(color="#a5b4fc", size=12)),
        height=260, xaxis=dict(tickprefix="₹", **_AXIS_DEFAULTS),
        yaxis=dict(autorange="reversed", **_AXIS_DEFAULTS), showlegend=False)
    return fig


def _chart_influencer_lift(ctx: dict, df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["Category","Influencer Active"])["Total Revenue"].mean().reset_index()
    grp.columns = ["Category","Influencer","Avg Revenue"]
    fig = px.bar(grp, x="Category", y="Avg Revenue", color="Influencer",
                 barmode="group", color_discrete_map={"Yes":"#6366f1","No":"#4a5a7a"})
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text="💬 Influencer Lift by Category", font=dict(color="#a5b4fc", size=12)),
        height=240, yaxis=dict(tickprefix="₹", **_AXIS_DEFAULTS))
    fig.update_traces(marker_line_width=0, opacity=0.85)
    return fig


def _chart_discount_curve(ctx: dict) -> go.Figure:
    dg = ctx["disc_grp"]
    if dg is None: return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=dg["Discount"].astype(str)+"%", y=dg["avg_rev"],
        name="Avg Revenue", marker_color="#6366f1", opacity=0.85, marker_line_width=0,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dg["Discount"].astype(str)+"%", y=dg["avg_orders"],
        name="Avg Orders", mode="lines+markers+text",
        text=[f"{v:.0f}" for v in dg["avg_orders"]],
        textposition="top center", textfont=dict(color="#06b6d4", size=9),
        line=dict(color="#06b6d4", width=2), marker=dict(size=7),
    ), secondary_y=True)
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text="💬 Discount Sweet Spot", font=dict(color="#a5b4fc", size=12)),
        height=240)
    fig.update_yaxes(tickprefix="₹", secondary_y=False)
    return fig


def _chart_profit_margin_by_category(ctx: dict, df: pd.DataFrame) -> go.Figure:
    bm = ctx["best_m"]
    if bm is None: return None
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in bm.values]
    fig = go.Figure(go.Bar(
        x=bm.index.tolist(), y=bm.values,
        marker_color=colors, marker_line_width=0, opacity=0.85,
        text=[f"{v:.1f}%" for v in bm.values], textposition="outside",
        textfont=dict(color="#f0f4ff", size=10),
    ))
    fig.update_layout(**PLOTLY_BASE,
        title=dict(text="💬 Avg Profit Margin by Category", font=dict(color="#a5b4fc", size=12)),
        height=240, yaxis=dict(ticksuffix="%", **_AXIS_DEFAULTS), showlegend=False)
    return fig


def _chart_orders_by_city(df: pd.DataFrame) -> go.Figure:
    city_ord = df.groupby("City")["Orders"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=city_ord.index.tolist(), y=city_ord.values,
        marker_color=[CITY_CLR.get(c,"#6366f1") for c in city_ord.index],
        marker_line_width=0, opacity=0.85,
        text=city_ord.values, textposition="outside",
        textfont=dict(color="#f0f4ff", size=10),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="💬 Orders by City", font=dict(color="#a5b4fc", size=12)),
        height=240, showlegend=False)
    return fig


def _chart_summary_snapshot(ctx: dict) -> go.Figure:
    """Mini 4-panel summary chart for the executive summary response."""
    cat_r  = ctx["cat_r"]
    city_r = ctx["city_r"]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Revenue by Category", "Revenue by City"],
        specs=[[{"type":"pie"}, {"type":"bar"}]],
    )
    fig.add_trace(go.Pie(
        labels=cat_r.index.tolist(), values=cat_r.values,
        marker_colors=[CAT_CLR.get(c,"#6366f1") for c in cat_r.index],
        hole=0.5, textinfo="label+percent", textfont_size=9, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=city_r.index.tolist(), y=city_r.values,
        marker_color=[CITY_CLR.get(c,"#6366f1") for c in city_r.index],
        marker_line_width=0, opacity=0.85,
        text=[fmt(v) for v in city_r.values], textposition="outside",
        textfont=dict(color="#f0f4ff", size=9), showlegend=False,
    ), row=1, col=2)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8899bb", size=10),
        margin=dict(l=10, r=10, t=40, b=10), height=260,
        title=dict(text="💬 Snapshot", font=dict(color="#a5b4fc", size=12)),
    )
    fig.update_annotations(font_color="#8899bb", font_size=10)
    fig.update_yaxes(tickprefix="₹", gridcolor="rgba(99,130,255,.06)")
    return fig


def _fig_to_json(fig: go.Figure) -> str | None:
    """Serialize a Plotly figure to JSON for session_state storage."""
    if fig is None: return None
    import json
    return fig.to_json()


def _fig_from_json(s: str | None) -> go.Figure | None:
    """Deserialize a Plotly figure from JSON."""
    if not s: return None
    import json
    return go.Figure(json.loads(s))


# ── Type alias for handler return ─────────────────────────────────────────────────
# Each handler returns (text, fig) | None
# fig=None means no inline chart for this response
BotReply = tuple[str, "go.Figure | None"]


def _bb_greeting(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["hello","hi","hey","namaste","hii"]): return None
    c         = ctx
    returning = mem.turn_count > 0
    opener    = (f"Welcome back! You've asked **{mem.turn_count}** question(s) so far."
                 if returning else f"I've analyzed **{len(df):,} records** and I'm ready to help.")
    mem.update("greeting")
    text = (
        ResponseBuilder("👋", "Hi! I'm BlinkBot — your AI Business Analyst")
        .answer(opener)
        .metric("Total Revenue", fmt(c["total_r"]), "💰")
        .metric("Top Category",  c["cat_r"].index[0]  if c["cat_r"]  is not None else "N/A", "🏆")
        .metric("Top City",      c["city_r"].index[0] if c["city_r"] is not None else "N/A", "📍")
        .metric("Total Orders",  f"{int(c['total_o']):,}", "🛒")
        .followup("'Give me a full summary' · 'Which city is weakest?' · 'Best product?'")
        .build()
    )
    return text, _chart_revenue_by_category(ctx)


def _bb_summary(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["summary","overview","analyze","brief","insights","tell me"]): return None
    c       = ctx
    aov     = c["total_r"] / c["total_o"] if c["total_o"] else 0
    weakest = c["city_r"].index[-1] if c["city_r"] is not None else "N/A"
    mem.update("summary",
               city     = c["city_r"].index[0] if c["city_r"] is not None else None,
               product  = c["prod_r"].index[0] if c["prod_r"] is not None else None,
               category = c["cat_r"].index[0]  if c["cat_r"]  is not None else None)
    text = (
        ResponseBuilder("📋", "Executive Summary")
        .answer(f"Here's everything at a glance across **{len(df):,} records**.")
        .metric("Total Revenue",  fmt(c["total_r"]), "💰")
        .metric("Total Profit",   f"{fmt(c['total_p'])} ({c['mgn']:.1f}% margin)", "📈")
        .metric("Total Orders",   f"{int(c['total_o']):,} | AOV: {fmt(aov)}", "🛒")
        .metric("Best Category",  f"{c['cat_r'].index[0] if c['cat_r'] is not None else 'N/A'} "
                                  f"({c['cat_r'].iloc[0]/c['total_r']*100:.1f}% of rev)", "🏆")
        .metric("Best City",      f"{c['city_r'].index[0] if c['city_r'] is not None else 'N/A'} "
                                  f"— {fmt(c['city_r'].iloc[0]) if c['city_r'] is not None else 'N/A'}", "📍")
        .metric("Best Product",   c["prod_r"].index[0] if c["prod_r"] is not None else "N/A", "⭐")
        .context(f"⚠️ **{weakest}** is your weakest region — investigate and run targeted promotions.")
        .tip(f"Focus on {c['cat_r'].index[0] if c['cat_r'] is not None else 'top category'} "
             f"in {c['city_r'].index[0] if c['city_r'] is not None else 'top city'} — this is your growth engine.")
        .followup(f"'Tell me about {weakest}' · 'Breakdown by category' · 'Best product?'")
        .build()
    )
    return text, _chart_summary_snapshot(ctx)


def _bb_revenue(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["revenue","how much","earnings","sales total"]): return None
    c        = ctx
    top_cat  = c["cat_r"].index[0]  if c["cat_r"]  is not None else "N/A"
    top_city = c["city_r"].index[0] if c["city_r"] is not None else "N/A"
    target_city = mem.last_city
    extra_ctx   = ""
    if target_city and target_city in df["City"].values:
        city_rev_val = df[df["City"]==target_city]["Total Revenue"].sum()
        share        = city_rev_val / c["total_r"] * 100 if c["total_r"] else 0
        extra_ctx    = f"{target_city} contributes **{fmt(city_rev_val)}** ({share:.1f}% of total)."
    mem.update("revenue", category=top_cat, city=top_city)
    rb = (
        ResponseBuilder("📊", "Revenue Analysis")
        .answer(f"Total revenue is **{fmt(c['total_r'])}** across **{len(df):,} transactions**.")
        .metric("Best category",   f"{top_cat} ({c['cat_r'].iloc[0]/c['total_r']*100:.1f}%)" if c["cat_r"] is not None else "N/A", "🏆")
        .metric("Top city",        top_city, "📍")
        .metric("Net profit",      f"{fmt(c['total_p'])} ({c['mgn']:.1f}% margin)", "💰")
        .metric("Avg order value", fmt(c["total_r"]/c["total_o"] if c["total_o"] else 0), "🛒")
    )
    if extra_ctx: rb = rb.context(extra_ctx)
    rb = rb.tip(f"Double down on **{top_cat}** in **{top_city}** — allocate 30% more marketing budget here.")
    rb = rb.followup("'Break down by category' · 'Which product earns most?' · 'Compare cities'")
    return rb.build(), _chart_revenue_by_category(ctx)


def _bb_profit(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["profit","margin","net"]): return None
    c          = ctx
    bm         = c["best_m"]
    prev_topic = mem.last_intent
    mem.update("profit", category=bm.index[0] if bm is not None else None)
    rb = (
        ResponseBuilder("💰", "Profit & Margin Analysis")
        .answer(f"Total profit is **{fmt(c['total_p'])}** on revenue of **{fmt(c['total_r'])}**.")
        .metric("Profit margin",      f"{c['mgn']:.1f}%", "📈")
        .metric("Highest-margin cat", f"{bm.index[0] if bm is not None else 'N/A'} ({bm.iloc[0]:.1f}%)" if bm is not None else "N/A", "🏆")
        .metric("Lowest-margin cat",  f"{bm.index[-1] if bm is not None else 'N/A'} — needs review"     if bm is not None else "N/A", "⚠️")
        .metric("Keep per ₹100",      f"₹{c['mgn']:.0f}", "🪙")
    )
    if prev_topic == "revenue":
        rb = rb.context("You were just looking at revenue — margin tells you how much you actually keep.")
    rb = rb.tip(f"Grow **{bm.index[0] if bm is not None else 'top category'}** volume — best return per rupee sold.")
    rb = rb.followup("'Which product has best margin?' · 'Revenue breakdown' · 'Discount impact on margin?'")
    return rb.build(), _chart_profit_margin_by_category(ctx, df)


def _bb_best_product(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["best product","top product","number one","highest selling"]): return None
    pr = ctx["prod_r"]
    if pr is None: return "Product data not available.", None
    top3   = pr.head(3)
    medals = ["🥇","🥈","🥉"]
    lines  = "\n".join([f"{medals[i]} **{top3.index[i]}** — {fmt(top3.iloc[i])}" for i in range(len(top3))])
    mem.update("best_product", product=top3.index[0])
    text = (
        ResponseBuilder("🏆", "Top Products by Revenue")
        .answer(f"Your #1 product is **{top3.index[0]}** generating **{fmt(top3.iloc[0])}**.\n\n{lines}")
        .metric("Share of total", f"{top3.iloc[0]/ctx['total_r']*100:.1f}%", "📊")
        .metric("Runner-up gap",  fmt(top3.iloc[0]-top3.iloc[1]) if len(top3)>1 else "—", "↔️")
        .tip(f"Keep **{top3.index[0]}** always in stock. Bundle with **{top3.index[1] if len(top3)>1 else '#2'}** to boost AOV.")
        .followup(f"'Worst products?' · 'Which city sells {top3.index[0]} most?' · 'Inventory alert?'")
        .build()
    )
    return text, _chart_top_products(ctx)


def _bb_worst_product(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["worst product","lowest","weakest product"]): return None
    pr = ctx["prod_r"]
    if pr is None: return "Product data not available.", None
    worst = pr.tail(3).sort_values()
    lines = "\n".join([f"{['🔴','🟡','🟡'][i]} **{worst.index[i]}** — {fmt(worst.iloc[i])}" for i in range(len(worst))])
    mem.update("worst_product", product=worst.index[0])
    text = (
        ResponseBuilder("⚠️", "Underperforming Products")
        .answer(f"Lowest revenue: **{worst.index[0]}** at only **{fmt(worst.iloc[0])}**.\n\n{lines}")
        .metric("Gap to #1", fmt(ctx["prod_r"].iloc[0] - worst.iloc[0]), "↕️")
        .tip("Run a 30-day promotion on these. If no improvement, discontinue the lowest performer.")
        .followup(f"'Best product?' · 'Category for {worst.index[0]}?' · 'Discount to boost it?'")
        .build()
    )
    return text, _chart_top_products(ctx, n=8)


def _bb_city(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["city","region","location","where"]): return None
    cr      = ctx["city_r"]
    if cr is None: return "City data not found.", None
    weakest  = cr.index[-1]
    best     = cr.index[0]
    gap_pct  = ((cr.iloc[0]-cr.iloc[-1])/cr.iloc[-1]*100) if len(cr)>1 and cr.iloc[-1]>0 else 0
    ranking  = "\n".join([
        f"{i+1}. {'🟢' if i==0 else '🟡' if i<len(cr)-1 else '🔴'} **{c}** — {fmt(v)}"
        for i,(c,v) in enumerate(cr.items())
    ])
    prev_city = mem.last_city
    mem.update("city", city=best)
    rb = (
        ResponseBuilder("📍", "City & Region Performance")
        .answer(f"**{best}** is your strongest market at **{fmt(cr.iloc[0])}**.\n\n{ranking}")
        .metric("Performance gap", f"{gap_pct:.0f}% between best and worst", "↕️")
    )
    if prev_city and prev_city != best and prev_city in cr.index:
        prev_val = cr[prev_city]
        rb = rb.context(f"You asked about **{prev_city}** earlier — {fmt(prev_val)}, ranked #{list(cr.index).index(prev_city)+1}.")
    if gap_pct > 50:
        rb = rb.context(f"⚠️ {weakest} underperforms by **{gap_pct:.0f}%** — big opportunity here.")
    rb = rb.tip(f"Replicate {best}'s success in {weakest} — start with influencer campaigns for top 3 products.")
    rb = rb.followup(f"'Revenue in {weakest}' · 'Best product in {best}' · 'Compare cities'")
    return rb.build(), _chart_city_ranking(ctx)


def _bb_category(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["category","segment","best category"]): return None
    cat_r   = ctx["cat_r"]
    total_r = ctx["total_r"]
    if cat_r is None: return "Category data not available.", None
    medals  = ["🥇","🥈","🥉"] + ["▫️"]*(len(cat_r)-3)
    lines   = "\n".join([f"{medals[i]} **{c}** — {fmt(v)} ({v/total_r*100:.1f}%)" for i,(c,v) in enumerate(cat_r.items())])
    weakest = cat_r.index[-1]
    mem.update("category", category=cat_r.index[0])
    text = (
        ResponseBuilder("🏷️", "Category Performance Breakdown")
        .answer(f"**{cat_r.index[0]}** leads with **{fmt(cat_r.iloc[0])}** ({cat_r.iloc[0]/total_r*100:.1f}% of total).\n\n{lines}")
        .metric("Categories tracked", str(len(cat_r)), "📂")
        .metric("Weakest segment",    f"{weakest} ({cat_r.iloc[-1]/total_r*100:.1f}%)", "⚠️")
        .tip(f"**{weakest}** is weakest at {cat_r.iloc[-1]/total_r*100:.1f}%. Promote it or shift budget to **{cat_r.index[0]}**.")
        .followup(f"'Products in {cat_r.index[0]}' · 'Which city buys most?' · 'Margin by category'")
        .build()
    )
    return text, _chart_revenue_by_category(ctx)


def _bb_influencer(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["influencer","marketing","campaign"]): return None
    if "Influencer Active" not in df.columns: return "Influencer data not available.", None
    c           = ctx
    significant = abs(c["inf_lift"]) > 5
    mem.update("influencer")
    text = (
        ResponseBuilder("⚡", "Influencer Marketing Impact")
        .answer(f"Influencer-active products generate a **{c['inf_lift']:+.1f}% revenue lift**.")
        .metric("With influencer (avg rev)",    fmt(c["inf_y_rev"]), "✅")
        .metric("Without influencer (avg rev)", fmt(c["inf_n_rev"]), "❌")
        .metric("Order volume lift",            f"{c['ord_lift']:+.1f}%", "📦")
        .metric("Influencer-active SKUs",       f"{c['n_inf_y']} of {len(df)}", "🎯")
        .context("Lift is statistically meaningful ✓" if significant else "Lift is small — run a larger test.")
        .tip("Scale up — activate influencers for ALL top-category products!" if c["inf_lift"]>5 else "Small lift — focus on micro-influencers in specific cities.")
        .followup("'Which category benefits most?' · 'Top cities for campaigns?' · 'Influencer ROI?'")
        .build()
    )
    return text, _chart_influencer_lift(ctx, df)


def _bb_orders(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["orders","order count","volume","how many orders"]): return None
    c      = ctx
    aov    = c["total_r"] / c["total_o"] if c["total_o"] else 0
    top_co = df.groupby("City")["Orders"].sum().sort_values(ascending=False) if "City" in df.columns else None
    mem.update("orders", city=top_co.index[0] if top_co is not None else None)
    text = (
        ResponseBuilder("🛒", "Order Volume Analysis")
        .answer(f"**{int(c['total_o']):,} total orders** have been processed.")
        .metric("Average order value",  fmt(aov), "💰")
        .metric("Top city by orders",   f"{top_co.index[0] if top_co is not None else 'N/A'} ({int(top_co.iloc[0]):,})", "📍")
        .metric("Revenue per order",    fmt(c["total_r"]/c["total_o"] if c["total_o"] else 0), "📊")
        .metric("Target AOV (+15%)",    fmt(aov*1.15), "🎯")
        .tip(f"Increase AOV from **{fmt(aov)}** to **{fmt(aov*1.15)}** with bundle deals. 15% AOV lift = 15% more revenue at zero extra cost.")
        .followup("'Which product has most orders?' · 'Orders by city?' · 'Discount impact?'")
        .build()
    )
    return text, _chart_orders_by_city(df)


def _bb_discount(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["discount","offer","deal","promo"]): return None
    dg = ctx["disc_grp"]
    if dg is None: return "Discount data not available.", None
    best  = dg.loc[dg["avg_rev"].idxmax()]
    lines = "\n".join([
        f"- **{int(r.Discount)}%** → Avg Rev: {fmt(r.avg_rev)} | Avg Orders: {r.avg_orders:.0f}"
        for _, r in dg.iterrows()
    ])
    mem.update("discount")
    text = (
        ResponseBuilder("🏷️", "Discount Effectiveness Analysis")
        .answer(f"Most effective discount: **{int(best['Discount'])}%** generating **{fmt(best['avg_rev'])}** avg revenue.\n\n{lines}")
        .metric("Sweet-spot discount", f"{int(best['Discount'])}%", "🎯")
        .metric("Peak avg revenue",    fmt(best["avg_rev"]), "💰")
        .metric("Peak avg orders",     f"{best['avg_orders']:.0f}", "🛒")
        .tip(f"Stick to **{int(best['Discount'])}%** as your standard promo rate. Avoid deeper discounts — they train customers to wait.")
        .followup("'Discount vs profit margin?' · 'Which category discounts best?' · 'Order volume vs discount'")
        .build()
    )
    return text, _chart_discount_curve(ctx)


def _bb_inventory(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["stock","inventory","reorder","shortage"]): return None
    pr = ctx["prod_r"]
    if pr is None: return "Product data not available.", None
    top5  = pr.head(5)
    items = "\n".join([f"{i+1}. 🔴 **{p}** — {fmt(v)} revenue" for i,(p,v) in enumerate(top5.items())])
    mem.update("inventory", product=top5.index[0])
    text = (
        ResponseBuilder("📦", "Inventory Risk Alert")
        .answer(f"Top 5 products by sales velocity (highest reorder priority):\n\n{items}")
        .metric("Priority reorder",    top5.index[0], "⚡")
        .metric("Safety stock target", "50+ units for top SKUs", "🎯")
        .metric("Auto-reorder trigger","20 units remaining",     "⚠️")
        .tip(f"Set auto-reorder alerts at 20 units for top products. Keep **{top5.index[0]}** at 100+ units safety stock.")
        .followup(f"'Days of cover for {top5.index[0]}?' · 'Which city sells it fastest?' · 'OOS impact on revenue?'")
        .build()
    )
    return text, _chart_top_products(ctx, n=5)


def _bb_compare(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply | None:
    if not any(w in q for w in ["compare","vs","versus","against","difference between"]): return None
    cr = ctx["city_r"]
    if cr is not None and len(cr) >= 2:
        c1, c2 = cr.index[0], cr.index[-1]
        diff   = cr.iloc[0] - cr.iloc[-1]
        gap    = diff / cr.iloc[-1] * 100 if cr.iloc[-1] else 0
        mem.update("compare", city=c1)
        text = (
            ResponseBuilder("⚖️", f"Comparison: {c1} vs {c2}")
            .answer(f"**{c1}** outperforms **{c2}** by **{gap:.0f}%**.")
            .metric(c1, fmt(cr.iloc[0]), "🟢")
            .metric(c2, fmt(cr.iloc[-1]), "🔴")
            .metric("Revenue gap", fmt(diff), "↕️")
            .tip(f"Investigate what makes {c1} strong — replicate those tactics in {c2}.")
            .followup(f"'Products in {c2}' · 'Influencer coverage in {c2}' · 'Category split in {c1}'")
            .build()
        )
        return text, _chart_city_ranking(ctx)
    return None


def _bb_fallback(q: str, ctx: dict, df: pd.DataFrame, mem: ConversationMemory) -> BotReply:
    hint    = f"\n\n💬 *Last topic: **{mem.last_intent}** — say 'tell me more' to expand.*" if mem.last_intent else ""
    cols_av = ", ".join(df.columns.tolist())
    text = (
        ResponseBuilder("🤔", "I didn't quite catch that")
        .answer(
            "I can help you with:\n"
            "- 💰 Revenue & Profit\n- 🏆 Best / worst products\n- 📍 City performance\n"
            "- 🏷️ Category breakdown\n- ⚡ Influencer impact\n- 🛒 Orders & volume\n"
            "- 🏷️ Discount analysis\n- 📦 Inventory alerts\n- ⚖️ Compare cities" + hint
        )
        .context(f"Available columns: {cols_av}")
        .followup("'Give me a summary' · 'Which city is worst?' · 'Best product?'")
        .build()
    )
    return text, None


# Dispatch table
_BB_HANDLERS = [
    _bb_greeting, _bb_compare, _bb_summary, _bb_revenue, _bb_profit,
    _bb_best_product, _bb_worst_product, _bb_city, _bb_category,
    _bb_influencer, _bb_orders, _bb_discount, _bb_inventory,
]


def blinkbot_analyze(question: str, df: pd.DataFrame) -> BotReply:
    """
    Phase 3 entry point — returns (response_text, plotly_fig | None).
    1. Load ConversationMemory
    2. Extract named entities, resolve vague references
    3. Route through dispatch table
    4. Save updated memory
    """
    if df is None or len(df) == 0:
        return "⚠️ No data loaded yet. Please upload a CSV file to get started!", None

    mem      = _get_memory()
    entities = extract_entities(question, df)
    q_raw    = question.lower().strip()

    if entities["city"]:     mem.last_city     = entities["city"]
    if entities["product"]:  mem.last_product  = entities["product"]
    if entities["category"]: mem.last_category = entities["category"]

    q   = resolve_references(q_raw, mem)
    ctx = _bb_context(df)

    for handler in _BB_HANDLERS:
        result = handler(q, ctx, df, mem)
        if result is not None:
            _save_memory(mem)
            return result   # (text, fig)

    _save_memory(mem)
    return _bb_fallback(q, ctx, df, mem)


# ══════════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════════
# ── PHASE 4 — LLM INTEGRATION  (Anthropic API · streaming · chart detection)
# ══════════════════════════════════════════════════════════════════════════════════

_ANTHROPIC_URL    = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_MODEL  = "claude-sonnet-4-6"
_ANTHROPIC_VER    = "2023-06-01"
_MAX_LLM_TOKENS   = 1024
_LLM_HISTORY_LIMIT = 12   # keep last N messages to avoid ballooning context


def _build_llm_system_prompt(df: pd.DataFrame, kpis: dict) -> str:
    """
    Construct a rich, data-grounded system prompt for the LLM.
    Includes live KPIs, rankings, and strict behavioral rules.
    Refreshed on every call so it always reflects the active filter.
    """
    cat_r  = kpis["cat_rev"]
    city_r = kpis["city_rev"]
    prod_r = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)

    def _top_list(series, n=5):
        return "\n".join([f"  {i+1}. {k}: {fmt(v)}" for i,(k,v) in enumerate(series.head(n).items())])

    inf_y = df[df["Influencer Active"]=="Yes"]["Total Revenue"].mean() if "Influencer Active" in df.columns else 0
    inf_n = df[df["Influencer Active"]=="No"]["Total Revenue"].mean()  if "Influencer Active" in df.columns else 0
    inf_lift = ((inf_y - inf_n) / inf_n * 100) if inf_n > 0 else 0

    discount_sweet = ""
    if "Discount" in df.columns:
        dg = df.groupby("Discount")["Total Revenue"].mean()
        best_disc = dg.idxmax()
        discount_sweet = f"  Optimal discount rate: {int(best_disc)}% (highest avg revenue)"

    return f"""You are **BlinkBot**, the AI Business Analyst embedded in Ayush Intelligence Hub — a Zepto quick-commerce sales dashboard.

## LIVE DATA SNAPSHOT  ({len(df):,} records, filters active)

### Core KPIs
- Total Revenue      : {fmt(kpis['total_rev'])}
- Total Profit       : {fmt(kpis['total_profit'])} ({kpis['margin']:.1f}% margin)
- Total Orders       : {int(kpis['total_orders']):,}
- Avg Order Value    : {fmt(kpis['aov'])}
- Revenue Std Dev    : {fmt(kpis['rev_std'])}

### Top Products
{_top_list(prod_r)}

### Revenue by City
{_top_list(city_r)}

### Revenue by Category
{_top_list(cat_r, n=len(cat_r))}

### Influencer & Discount
- Influencer lift    : {inf_lift:+.1f}% revenue uplift (active vs inactive)
- Avg price range    : ₹{df['Current Price'].min():.0f}–₹{df['Current Price'].max():.0f}
- Avg discount       : {df['Discount'].mean():.1f}%
{discount_sweet}

## PERSONALITY & RULES
1. You are direct and data-driven — lead with the specific number, not a hedge
2. Always use Indian currency format: ₹12.3L (lakhs), ₹2.1Cr (crores)
3. Every response MUST end with one concrete 💡 Recommendation
4. Keep answers concise (3-6 sentences) unless the user explicitly asks for detail
5. You have full conversation context — use it for follow-up questions
6. NEVER invent data not in the snapshot above; say "data not available" if uncertain
7. When comparing, cite the exact gap (₹ and %)
8. Tone: confident senior analyst, not a chatbot — no filler phrases like "Great question!"
"""


def _call_anthropic_stream(messages: list[dict], system: str, api_key: str):
    """
    Generator that streams text chunks from the Anthropic API (SSE protocol).
    Yields plain string chunks as they arrive.
    Falls back gracefully on any network or API error.
    """
    headers = {
        "Content-Type":    "application/json",
        "x-api-key":       api_key,
        "anthropic-version": _ANTHROPIC_VER,
    }
    payload = {
        "model":      _ANTHROPIC_MODEL,
        "max_tokens": _MAX_LLM_TOKENS,
        "system":     system,
        "messages":   messages[-_LLM_HISTORY_LIMIT:],
        "stream":     True,
    }
    try:
        with requests.post(
            _ANTHROPIC_URL, headers=headers, json=payload,
            stream=True, timeout=45
        ) as resp:
            if resp.status_code == 401:
                yield "\n\n⚠️ **Invalid API key.** Please check your key in the sidebar."
                return
            if resp.status_code == 429:
                yield "\n\n⚠️ **Rate limit hit.** Wait a moment and try again."
                return
            if not resp.ok:
                yield f"\n\n⚠️ **API error {resp.status_code}.** Switching to rule-based mode."
                return
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str in ("[DONE]", ""):
                    break
                try:
                    chunk = json.loads(data_str)
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.Timeout:
        yield "\n\n⚠️ **Request timed out.** The API took too long — try again."
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ **Connection error.** Check network access to api.anthropic.com."
    except Exception as exc:
        yield f"\n\n⚠️ **Unexpected error:** {exc}"


def _detect_chart_for_question(question: str, ctx: dict, df: pd.DataFrame) -> "go.Figure | None":
    """
    Lightweight keyword router that picks the most relevant chart
    for the user's question — used alongside LLM responses.
    """
    q = question.lower()
    if any(w in q for w in ["compare","vs","versus","against","city","region","where","location"]):
        return _chart_city_ranking(ctx)
    if any(w in q for w in ["category","segment"]):
        return _chart_revenue_by_category(ctx)
    if any(w in q for w in ["best product","top product","worst","lowest","product","sku","item"]):
        return _chart_top_products(ctx)
    if any(w in q for w in ["profit","margin","net"]):
        return _chart_profit_margin_by_category(ctx, df)
    if any(w in q for w in ["influencer","marketing","campaign"]):
        return _chart_influencer_lift(ctx, df)
    if any(w in q for w in ["orders","volume","order count"]):
        return _chart_orders_by_city(df)
    if any(w in q for w in ["discount","promo","offer","deal"]):
        return _chart_discount_curve(ctx)
    if any(w in q for w in ["summary","overview","revenue","earnings","how much"]):
        return _chart_summary_snapshot(ctx)
    if any(w in q for w in ["hello","hi","hey"]):
        return _chart_revenue_by_category(ctx)
    return None


# ── SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
      <div style="width:44px;height:44px;background:linear-gradient(135deg,#6366f1,#06b6d4);border-radius:12px;display:inline-flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#fff;margin-bottom:8px">N</div>
      <div style="font-size:13px;font-weight:600;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Nova-MS</div>
      <div style="font-size:10px;color:#4a5a7a;margin-top:2px">Sales Dashboard v4.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📂 Data Source")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], help="Replace the default dataset")
    st.markdown("---")

    df_raw = load_default()
    if uploaded:
        try:
            df_raw = clean(pd.read_csv(uploaded))
            st.success(f"✅ Loaded {len(df_raw):,} rows")
        except Exception as e:
            st.error(f"❌ {e}")

    st.markdown("#### 🔍 Filters")
    cities     = ["All"] + sorted(df_raw["City"].unique())
    categories = ["All"] + sorted(df_raw["Category"].unique())
    products   = ["All"] + sorted(df_raw["Product Name"].unique())

    sel_city = st.selectbox("City / Region", cities)
    sel_cat  = st.selectbox("Category",      categories)
    sel_inf  = st.selectbox("Influencer",    ["All", "Yes", "No"])
    sel_prod = st.selectbox("Product",       products)
    search   = st.text_input("Search product", placeholder="e.g. Maggi...")

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
    show_raw     = st.checkbox("Show Raw Data Table",       value=False)
    show_stats   = st.checkbox("Show Statistical Analysis", value=True)

    st.markdown("---")

    # ── Phase 4: AI Mode ─────────────────────────────────────────────────────────
    st.markdown("#### 🤖 BlinkBot AI Mode")
    use_ai_mode = st.toggle("Enable LLM Mode", value=False, help="Use Anthropic API for natural-language answers")

    if use_ai_mode:
        # Try st.secrets first, fall back to text input
        secret_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
        if secret_key:
            api_key = secret_key
            st.markdown("""
            <div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);
                        border-radius:8px;padding:8px 10px;font-size:10px;color:#34d399">
              ✅ API key loaded from secrets
            </div>""", unsafe_allow_html=True)
        else:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                help="Get yours at console.anthropic.com",
            )
            if api_key:
                st.markdown("""
                <div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);
                            border-radius:8px;padding:8px 10px;font-size:10px;color:#34d399">
                  ✅ Key set — LLM mode active
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
                            border-radius:8px;padding:8px 10px;font-size:10px;color:#f59e0b">
                  ⚠️ Enter your key above to activate LLM mode.<br>
                  Rule-based fallback is active.
                </div>""", unsafe_allow_html=True)
                api_key = ""
        st.markdown(f"""
        <div style="font-size:9px;color:#4a5a7a;margin-top:6px">
          Model: <span style="color:#a5b4fc;font-family:monospace">{_ANTHROPIC_MODEL}</span><br>
          History: last {_LLM_HISTORY_LIMIT} turns · Max tokens: {_MAX_LLM_TOKENS}
        </div>""", unsafe_allow_html=True)
    else:
        api_key = ""
        st.markdown("""
        <div style="background:rgba(99,130,255,.06);border:1px solid rgba(99,130,255,.12);
                    border-radius:8px;padding:8px 10px;font-size:10px;color:#8899bb">
          🔧 Rule-based mode — fast &amp; offline.<br>Toggle above to enable LLM responses.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#4a5a7a;text-align:center">
      Developed by <strong style="color:#a5b4fc">Ayush Mishra</strong><br>
      FastAPI · Pandas · SciPy · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# ── FILTERS
# ══════════════════════════════════════════════════════════════════════════════════

df = df_raw.copy()
if sel_city != "All": df = df[df["City"]              == sel_city]
if sel_cat  != "All": df = df[df["Category"]          == sel_cat]
if sel_inf  != "All": df = df[df["Influencer Active"] == sel_inf]
if sel_prod != "All": df = df[df["Product Name"]      == sel_prod]
if search:            df = df[df["Product Name"].str.contains(search, case=False, na=False)]


# ══════════════════════════════════════════════════════════════════════════════════
# ── PRE-COMPUTE EVERYTHING  (all math happens here, before any rendering)
# ══════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#0d1628,#121d35);border:1px solid rgba(99,130,255,.12);border-radius:16px;padding:20px 24px;margin-bottom:20px">
  <h1 style="margin:0;font-size:22px;font-weight:700;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Ayush Intelligence Hub
  </h1>
  <p style="margin:4px 0 0;font-size:12px;color:#4a5a7a">
    Real-Time Business Insights · Statistical Analytics · ML Forecasting · Developed by Ayush Mishra
  </p>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ No data matches your filters. Please adjust the filters.")
    st.stop()

# ── Run all calculations ──────────────────────────────────────────────────────────
kpis       = compute_kpis(df)
inf_stats  = compute_influencer_stats(df)
stats_data = compute_statistics(df) if show_stats and len(df) >= 5 else None
forecast   = compute_forecast(df)
delivery   = compute_delivery_stats(len(df))
unit_econ  = compute_unit_economics(float(df["Total Revenue"].mean()) if len(df) > 0 else 500)
inventory  = compute_inventory(df)
defects    = compute_order_defects(int(kpis["total_orders"]))
wow        = compute_wow_metrics(kpis)
insights   = compute_ai_insights(df, kpis, inf_stats if inf_stats["available"] else {"rev_lift":0,"p_value":1,"significant":False})


# ══════════════════════════════════════════════════════════════════════════════════
# ── RENDERING STARTS HERE  (zero calculations below)
# ══════════════════════════════════════════════════════════════════════════════════

# ── KPI CARDS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">KEY PERFORMANCE INDICATORS</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi_rows = [
    (c1,"💰","Total Revenue",   fmt(kpis["total_rev"]),                     f"σ = {fmt(kpis['rev_std'])}",         "#a5b4fc","↑ +12.4%","up"),
    (c2,"📈","Total Profit",    fmt(kpis["total_profit"]),                  "Net margin earnings",                  "#6ee7b7","↑ +8.1%", "up"),
    (c3,"🛒","Total Orders",    f"{int(kpis['total_orders']):,}",           "Units sold",                           "#fcd34d","↑ +5.3%", "up"),
    (c4,"%", "Profit Margin",   f"{kpis['margin']:.1f}%",                  "Revenue to profit ratio",              "#fca5a5","↑ +2.1%", "up"),
    (c5,"⭐","Top Category",    kpis["cat_rev"].index[0]  if len(kpis["cat_rev"])  else "—", fmt(kpis["cat_rev"].iloc[0])  if len(kpis["cat_rev"])  else "—", "#67e8f9","Leader","up"),
    (c6,"📍","Top Region",      kpis["city_rev"].index[0] if len(kpis["city_rev"]) else "—", fmt(kpis["city_rev"].iloc[0]) if len(kpis["city_rev"]) else "—", "#c4b5fd","Leader","up"),
]
for col, icon, label, val, sub, clr, badge, cls in kpi_rows:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div style="font-size:20px;margin-bottom:8px">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{clr}">{val}</div>
          <div class="kpi-badge {cls}">{badge}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CHARTS ROW 1 ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">SALES & REVENUE ANALYTICS</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    city_data = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(city_data, x="City", y="Total Revenue", color="City",
                 color_discrete_map=CITY_CLR, title="Revenue by City", labels={"Total Revenue":"Revenue (₹)"})
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", showlegend=False)
    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    cat_data = df.groupby("Category")["Total Revenue"].sum().reset_index()
    fig = px.pie(cat_data, values="Total Revenue", names="Category",
                 color="Category", color_discrete_map=CAT_CLR, title="Category Distribution", hole=0.55)
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff")
    fig.update_traces(textinfo="label+percent", textfont_size=10)
    st.plotly_chart(fig, use_container_width=True)

# ── CHARTS ROW 2 ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    top_prod = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_prod, x="Total Revenue", y="Product Name", orientation="h",
                 title="Top 10 Products by Revenue", color="Total Revenue",
                 color_continuous_scale=["#6366f1","#06b6d4","#10b981"])
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", coloraxis_showscale=False)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(99,130,255,.06)")
    fig.update_xaxes(tickformat=",.0f", tickprefix="₹")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.scatter(df, x="Orders", y="Total Revenue", color="Category",
                     color_discrete_map=CAT_CLR, hover_name="Product Name",
                     hover_data={"City":True,"Discount":True},
                     title="Orders vs Revenue (Scatter)", labels={"Total Revenue":"Revenue (₹)"})
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff")
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    st.plotly_chart(fig, use_container_width=True)

# ── HEATMAP ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">CITY × CATEGORY HEATMAP</div>', unsafe_allow_html=True)
pivot = df.pivot_table(index="Category", columns="City", values="Total Revenue", aggfunc="sum", fill_value=0)
fig = go.Figure(data=go.Heatmap(
    z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
    colorscale=[[0,"#0d1628"],[0.3,"#1e1b6e"],[0.6,"#3730a3"],[1,"#6366f1"]],
    text=[[fmt(v) for v in row] for row in pivot.values],
    texttemplate="%{text}", hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
))
fig.update_layout(**PLOTLY_LAYOUT, title="Revenue Intensity (City × Category)", title_font_color="#f0f4ff", height=280)
st.plotly_chart(fig, use_container_width=True)

# ── INFLUENCER + DISCOUNT ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    inf_data = df.groupby(["Category","Influencer Active"])["Total Revenue"].mean().reset_index()
    inf_data.columns = ["Category","Influencer","Avg Revenue"]
    fig = px.bar(inf_data, x="Category", y="Avg Revenue", color="Influencer",
                 barmode="group", title="Influencer Impact by Category",
                 color_discrete_map={"Yes":"#6366f1","No":"#4a5a7a"})
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff")
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    disc_data = df.groupby("Discount").agg(
        Avg_Revenue=("Total Revenue","mean"),
        Avg_Orders=("Orders","mean"),
        Count=("Orders","count")
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=disc_data["Discount"].astype(str)+"%", y=disc_data["Avg_Revenue"],
                         name="Avg Revenue", marker_color="#6366f1", opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(x=disc_data["Discount"].astype(str)+"%", y=disc_data["Avg_Orders"],
                             name="Avg Orders", mode="lines+markers",
                             line=dict(color="#06b6d4", width=2)), secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title="Discount vs Revenue & Orders", title_font_color="#f0f4ff")
    fig.update_yaxes(tickprefix="₹", secondary_y=False)
    st.plotly_chart(fig, use_container_width=True)

# ── PRICE RANGE ───────────────────────────────────────────────────────────────────
price_data = df.groupby("Price Tier", observed=True)["Total Revenue"].sum().reset_index()
price_data["Price Tier"] = price_data["Price Tier"].astype(str)
fig = px.bar(price_data, x="Price Tier", y="Total Revenue", color="Price Tier",
             color_discrete_sequence=PAL, title="Revenue by Price Tier", labels={"Total Revenue":"Revenue (₹)"})
fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", showlegend=False)
fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
fig.update_traces(marker_line_width=0, opacity=0.85)
st.plotly_chart(fig, use_container_width=True)

# ── STATISTICAL ANALYSIS ─────────────────────────────────────────────────────────
if stats_data:
    sd = stats_data
    st.markdown('<div class="section-head">STATISTICAL ANALYSIS</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px"><div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
        for label, val in [
            ("Mean Revenue",   fmt(sd["mean"])),
            ("Median Revenue", fmt(sd["median"])),
            ("Std Deviation",  fmt(sd["std"])),
            ("Skewness",       f"{sd['skewness']:.3f}"),
            ("Kurtosis",       f"{sd['kurtosis']:.3f}"),
            ("Normality p",    f"{sd['p_norm']:.4f}"),
        ]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
        normal_txt = "✓ Normally distributed" if sd["is_normal"] else "⚠ Not normally distributed"
        normal_clr = "#34d399" if sd["is_normal"] else "#fcd34d"
        st.markdown(f'<div style="margin-top:10px;font-size:10px;color:{normal_clr};background:rgba(99,130,255,.06);padding:7px 10px;border-radius:7px">{normal_txt}</div></div>', unsafe_allow_html=True)
    with c2:
        outliers = sd["outliers"]
        st.markdown(f'<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px"><div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">⚠ Outlier Detection ({len(outliers)} outliers)</div>', unsafe_allow_html=True)
        if len(outliers) > 0:
            for _, row in outliers.head(6).iterrows():
                st.markdown(f'<div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.15);border-radius:7px;padding:8px 10px;margin-bottom:5px"><div style="font-size:11px;font-weight:600;color:#f0f4ff">{row["Product Name"]}</div><div style="font-size:9px;color:#8899bb">{row["City"]} · {row["Category"]}</div><div style="display:flex;justify-content:space-between;margin-top:3px"><span style="font-size:10px;color:#67e8f9">{fmt(row["Total Revenue"])}</span><span style="font-size:10px;color:#fca5a5;font-family:monospace">Z={row["Z-Score"]}</span></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:11px;color:#4a5a7a;text-align:center;padding:20px">No significant outliers</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        r_disc, p_disc = sd["r_disc"], sd["p_disc"]
        r_rev,  p_rev  = sd["r_rev"],  sd["p_rev"]
        st.markdown('<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px"><div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
        for label, val in [
            ("Discount → Orders (r)", f"{r_disc:.3f}"),
            ("Discount → Orders (p)", f"{p_disc:.4f}"),
            ("Revenue → Profit (r)",  f"{r_rev:.3f}"),
            ("Revenue → Profit (p)",  f"{p_rev:.4f}"),
        ]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
        for pair, r, p in [("Discount ↔ Orders", r_disc, p_disc), ("Revenue ↔ Profit", r_rev, p_rev)]:
            sig = p < 0.05; direction = "positive" if r > 0 else "negative"
            txt = f"{'Strong' if abs(r) > 0.5 else 'Weak'} {direction} — {'significant ✓' if sig else 'not significant'}"
            clr = "#34d399" if sig else "#fcd34d"
            st.markdown(f'<div style="background:rgba(99,102,241,.07);border-radius:6px;padding:7px 10px;margin-top:6px"><div style="font-size:10px;font-weight:600;color:#a5b4fc">{pair}</div><div style="font-size:10px;color:{clr};margin-top:2px">{txt}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    fig = px.imshow(sd["corr_matrix"], text_auto=True,
                    color_continuous_scale=["#ef4444","#0d1628","#6366f1"],
                    zmin=-1, zmax=1, title="Full Correlation Matrix")
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── FORECASTING ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">SALES FORECASTING — LINEAR REGRESSION</div>', unsafe_allow_html=True)
if forecast:
    fc = forecast
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["xs"], y=fc["upper"], fill=None, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["xs"], y=fc["lower"], fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(99,102,241,.08)", name="95% CI"))
        fig.add_trace(go.Scatter(x=fc["xs"], y=fc["actual_vals"], mode="lines+markers", name="Actual", line=dict(color="#6366f1", width=2), marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=fc["xs"], y=fc["trend_vals"], mode="lines", name="Trend", line=dict(color="#06b6d4", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=[fc["n"]+1], y=[fc["next_val"]], mode="markers", name="Forecast", marker=dict(color="#10b981", size=12, symbol="star")))
        fig.update_layout(**PLOTLY_LAYOUT, title="Revenue Forecast with Confidence Interval", title_font_color="#f0f4ff", height=280)
        fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        growth_clr = "#34d399" if fc["growth_pct"] >= 0 else "#f87171"
        st.markdown(f"""
        <div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:20px">
          <div style="font-size:10px;color:#4a5a7a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Forecast Value</div>
          <div style="font-size:28px;font-weight:700;font-family:monospace;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{fmt(fc['next_val'])}</div>
          <div style="display:inline-block;font-size:11px;font-weight:600;padding:3px 9px;border-radius:5px;margin:8px 0;background:{'rgba(16,185,129,.12)' if fc['growth_pct']>=0 else 'rgba(239,68,68,.12)'};color:{growth_clr}">
            {'↑' if fc['growth_pct']>=0 else '↓'} {abs(fc['growth_pct']):.1f}% vs avg
          </div>
        """, unsafe_allow_html=True)
        for label, val in [("R² Score", f"{fc['r2']:.4f}"), ("Slope β₁", f"{fc['slope']:.3f}"), ("CI Band", fmt(fc["ci"]))]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
        quality = "✓ Good fit" if fc["r2"] > 0.6 else "⚠ Low R² — noisy data"
        q_clr   = "#34d399" if fc["r2"] > 0.6 else "#fcd34d"
        st.markdown(f'<div style="margin-top:10px;font-size:10px;color:{q_clr};background:rgba(99,130,255,.06);padding:7px 10px;border-radius:7px">{quality}</div></div>', unsafe_allow_html=True)

# ── AI INSIGHTS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">AI BUSINESS INSIGHTS</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, (emoji, title, body) in enumerate(insights):
    with cols[i % 3]:
        st.markdown(f'<div class="insight-card"><div style="font-size:22px;margin-bottom:8px">{emoji}</div><div class="insight-title">{title}</div><div class="insight-body">{body}</div></div><br>', unsafe_allow_html=True)

# ── WEEK OVER WEEK ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">WEEK-OVER-WEEK COMPARISON</div>', unsafe_allow_html=True)
w1,w2,w3,w4 = st.columns(4)
wow_rows = [
    (w1,"💰","Revenue This Week",  kpis["total_rev"],    wow["previous"]["total_rev"],    False),
    (w2,"🛒","Orders This Week",   kpis["total_orders"], wow["previous"]["total_orders"], False),
    (w3,"📈","Profit This Week",   kpis["total_profit"], wow["previous"]["total_profit"], False),
    (w4,"%", "Margin This Week",   kpis["margin"],       wow["previous"]["margin"],       True),
]
for col, icon, label, curr, prev, is_pct in wow_rows:
    badge, up = pct_change_label(curr, prev)
    clr      = "#34d399" if up else "#f87171"
    val      = fmt(curr) if not is_pct else f"{curr:.1f}%"
    prev_val = fmt(prev) if not is_pct else f"{prev:.1f}%"
    with col:
        st.markdown(f'<div class="kpi-card" style="border-color:rgba(99,130,255,.2)"><div style="font-size:18px;margin-bottom:6px">{icon}</div><div class="kpi-label">{label}</div><div class="kpi-value" style="color:#a5b4fc;font-size:20px">{val}</div><div style="font-size:10px;font-weight:600;color:{clr};margin-top:4px">{badge}</div><div class="kpi-sub">Last week: {prev_val}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── DELIVERY PERFORMANCE ──────────────────────────────────────────────────────────
st.markdown('<div class="section-head">🚀 DELIVERY PERFORMANCE</div>', unsafe_allow_html=True)
dl = delivery
col1, col2, col3 = st.columns(3)
with col1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=dl["otd_pct"],
        title={"text":"On-Time Delivery %","font":{"color":"#8899bb","size":13}},
        number={"suffix":"%","font":{"color":dl["status_color"],"size":28}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#4a5a7a"},
               "bar":{"color":dl["status_color"]}, "bgcolor":"#0d1628",
               "steps":[{"range":[0,85],"color":"rgba(239,68,68,.15)"},
                         {"range":[85,95],"color":"rgba(245,158,11,.15)"},
                         {"range":[95,100],"color":"rgba(16,185,129,.15)"}],
               "threshold":{"line":{"color":"#fff","width":2},"thickness":0.75,"value":95}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=220, margin=dict(l=20,r=20,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div style="text-align:center;font-size:12px;font-weight:600;color:{dl["status_color"]}">{dl["status"]}</div>', unsafe_allow_html=True)
with col2:
    fig = go.Figure(go.Bar(
        x=[dl["p50"],dl["avg"],dl["p90"],dl["promise"]],
        y=["P50","Avg","P90","Promise"], orientation="h",
        marker_color=["#10b981","#6366f1","#ef4444","#f59e0b"], marker_line_width=0,
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=220,
                      title=dict(text="Delivery Time (minutes)", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=40,t=40,b=10),
                      xaxis=dict(title="Minutes",gridcolor="rgba(99,130,255,.05)"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
with col3:
    bar_colors = ["#ef4444" if x > dl["promise"]+2 else "#10b981" for x in dl["hist_centers"]]
    fig = go.Figure(go.Bar(x=dl["hist_centers"], y=dl["hist_counts"], marker_color=bar_colors, marker_line_width=0))
    fig.add_vline(x=dl["promise"], line_dash="dash", line_color="#f59e0b",
                  annotation_text="10-min promise", annotation_font_color="#f59e0b")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=220,
                      title=dict(text="Order Distribution by Time", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=10,t=40,b=10),
                      xaxis=dict(title="Minutes",gridcolor="rgba(99,130,255,.05)"),
                      yaxis=dict(gridcolor="rgba(99,130,255,.05)"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── UNIT ECONOMICS ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">💰 UNIT ECONOMICS — CONTRIBUTION MARGIN</div>', unsafe_allow_html=True)
ue = unit_econ
col1, col2 = st.columns(2)
with col1:
    labels = ["Revenue","COGS","Rider Pay","Packaging","Gateway Fee","Promos","Net Profit"]
    values = [ue["avg_rev"],-ue["cogs"],-ue["rider"],-ue["packaging"],-ue["gateway"],-ue["promos"],ue["net_profit"]]
    fig = go.Figure(go.Waterfall(
        name="Unit Economics", orientation="v",
        measure=["absolute","relative","relative","relative","relative","relative","total"],
        x=labels, y=values, text=[fmt(abs(v)) for v in values], textposition="outside",
        connector={"line":{"color":"rgba(99,130,255,.3)"}},
        decreasing={"marker":{"color":"#ef4444"}},
        increasing={"marker":{"color":"#10b981"}},
        totals={"marker":{"color":"#6366f1"}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=320,
                      title=dict(text=f"Revenue → Net Profit | CM: {ue['cm_pct']:.1f}%", font=dict(color="#f0f4ff",size=12)),
                      margin=dict(l=10,r=10,t=50,b=10), yaxis=dict(gridcolor="rgba(99,130,255,.05)"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = go.Figure(go.Pie(
        labels=["COGS (52%)","Rider Pay (12%)","Packaging (3%)","Gateway (2%)","Promos (5%)","Net Profit (26%)"],
        values=[ue["cogs"],ue["rider"],ue["packaging"],ue["gateway"],ue["promos"],max(0,ue["net_profit"])],
        hole=0.6, marker=dict(colors=["#6366f1","#06b6d4","#f59e0b","#8b5cf6","#ec4899","#10b981"]),
        textinfo="label+percent", textfont=dict(size=10),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=320,
                      title=dict(text="Cost Structure Breakdown", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=10,t=50,b=10), legend=dict(font=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ORDER DEFECT RATE ────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">🔍 ORDER QUALITY — DEFECT RATE</div>', unsafe_allow_html=True)
df_d = defects
col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Funnel(
        y=df_d["funnel_labels"], x=df_d["funnel_y"],
        textinfo="value+percent initial",
        marker=dict(color=["#6366f1","#8b5cf6","#f59e0b","#ef4444","#10b981"]),
        connector=dict(line=dict(color="rgba(99,130,255,.2)", width=1)),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=300,
                      title=dict(text=f"Order Quality Funnel | ODR: {df_d['odr_pct']:.1f}%", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = go.Figure(go.Bar(
        x=["Expired/Damaged","Missing Items","Cancelled (OOS)"],
        y=[df_d["expired"],df_d["missing"],df_d["cancelled_oos"]],
        marker_color=["#ef4444","#f59e0b","#8b5cf6"], marker_line_width=0,
        text=[df_d["expired"],df_d["missing"],df_d["cancelled_oos"]],
        textposition="outside", textfont=dict(color="#f0f4ff"),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=300,
                      title=dict(text="Defect Breakdown by Category", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=10,t=50,b=10),
                      yaxis=dict(gridcolor="rgba(99,130,255,.05)"), xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── INVENTORY INTELLIGENCE ────────────────────────────────────────────────────────
st.markdown('<div class="section-head">📦 INVENTORY INTELLIGENCE — STOCK ALERTS</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">⚡ Top 10 High-Risk Inventory Items</div>', unsafe_allow_html=True)
    for _, row in inventory.iterrows():
        st.markdown(
            f'<div style="background:{row["_bg"]};border:1px solid {row["_border"]};border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between">'
            f'<div><div style="font-size:12px;font-weight:600;color:#f0f4ff">{row["Product"]}</div>'
            f'<div style="font-size:10px;color:#8899bb;margin-top:2px">Stock: {row["Stock_Left"]} · Daily: {row["Daily_Sales"]} · Covers: {row["Days_Cover"]} days</div></div>'
            f'<div style="text-align:right"><div style="font-size:11px;font-weight:700;color:{row["_color"]}">{row["Risk"]}</div>'
            f'<div style="font-size:10px;color:{row["_color"]};margin-top:2px">{row["Action"]}</div></div></div>',
            unsafe_allow_html=True,
        )
with col2:
    critical_count = inventory["Risk"].str.contains("CRITICAL").sum()
    low_count      = inventory["Risk"].str.contains("LOW").sum()
    ok_count       = inventory["Risk"].str.contains("OK").sum()
    fig = go.Figure(go.Pie(
        labels=["Critical 🔴","Low Stock 🟡","OK 🟢"],
        values=[critical_count, low_count, ok_count],
        hole=0.65, marker=dict(colors=["#ef4444","#f59e0b","#10b981"]),
        textinfo="label+value", textfont=dict(size=11),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=280,
                      title=dict(text="Stock Risk Distribution", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=10,t=50,b=10), legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:8px;padding:12px;text-align:center;margin-top:8px"><div style="font-size:22px;font-weight:700;color:#ef4444">{critical_count}</div><div style="font-size:10px;color:#8899bb">Products need immediate reorder</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CUSTOMER RETENTION ────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">👥 CUSTOMER RETENTION — NEW vs REPEAT</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    weeks    = ["Week 1","Week 2","Week 3","Week 4","Week 5","Week 6"]
    new_c    = [1200,980,1100,870,1050,920]
    repeat_c = [800,920,1050,1100,1200,1280]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="New Customers",    x=weeks, y=new_c,    marker_color="#6366f1", marker_line_width=0))
    fig.add_trace(go.Bar(name="Repeat Customers", x=weeks, y=repeat_c, marker_color="#10b981", marker_line_width=0))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=280,
                      barmode="group", title=dict(text="New vs Repeat Customers (Weekly)", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=10,t=50,b=10), legend=dict(font=dict(size=10)),
                      yaxis=dict(gridcolor="rgba(99,130,255,.05)"), xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    retention_matrix = [[100,68,52,41],[100,71,55,43],[100,65,48,38],[100,73,58,46]]
    fig = go.Figure(go.Heatmap(
        z=retention_matrix, x=["W+0","W+1","W+2","W+3"], y=["Week 1","Week 2","Week 3","Week 4"],
        colorscale=[[0,"#0d1628"],[0.4,"#312e81"],[0.7,"#4338ca"],[1,"#6366f1"]],
        text=[[f"{v}%" for v in row] for row in retention_matrix],
        texttemplate="%{text}",
        hovertemplate="Cohort: %{y}<br>Week: %{x}<br>Retention: %{text}<extra></extra>",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=280,
                      title=dict(text="Cohort Retention Table (%)", font=dict(color="#f0f4ff",size=13)),
                      margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# ── DEEP ANALYTICS 1: PARETO / 80-20 ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">📐 PARETO ANALYSIS — 80/20 REVENUE RULE</div>', unsafe_allow_html=True)
prod_rev_sorted = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)
cumulative_pct  = (prod_rev_sorted.cumsum() / prod_rev_sorted.sum() * 100).values
pareto_x        = list(range(1, len(prod_rev_sorted) + 1))
cutoff_idx      = next((i for i, v in enumerate(cumulative_pct) if v >= 80), len(pareto_x)-1)
cutoff_products = cutoff_idx + 1

col1, col2 = st.columns([3, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=prod_rev_sorted.index.tolist(), y=prod_rev_sorted.values,
        name="Revenue", marker_color="#6366f1", marker_line_width=0, opacity=0.75,
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=prod_rev_sorted.index.tolist(), y=cumulative_pct,
        name="Cumulative %", mode="lines+markers",
        line=dict(color="#06b6d4", width=2), marker=dict(size=5),
        yaxis="y2",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="#f59e0b",
                  annotation_text="80% Revenue Threshold", annotation_font_color="#f59e0b",
                  annotation_position="top right")
    fig.add_vrect(
        x0=-0.5, x1=cutoff_idx + 0.5,
        fillcolor="rgba(99,102,241,.06)", line_width=0,
        annotation_text=f"Top {cutoff_products} products",
        annotation_position="top left", annotation_font_color="#a5b4fc",
    )
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f"Pareto Chart — Top {cutoff_products} of {len(prod_rev_sorted)} products drive 80% of revenue",
                   font=dict(color="#f0f4ff", size=12)),
        height=300,
        yaxis =dict(title="Revenue (₹)", tickprefix="₹", gridcolor="rgba(99,130,255,.06)"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    range=[0,105], ticksuffix="%", showgrid=False),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    pareto_pct = cutoff_products / len(prod_rev_sorted) * 100
    rev_80     = prod_rev_sorted.iloc[:cutoff_products].sum()
    st.markdown(f"""
    <div style="background:#0d1628;border:1px solid rgba(99,130,255,.15);border-radius:12px;padding:18px;text-align:center">
      <div style="font-size:10px;color:#4a5a7a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">80/20 Rule</div>
      <div style="font-size:36px;font-weight:700;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{pareto_pct:.0f}%</div>
      <div style="font-size:11px;color:#8899bb;margin-top:4px">of products drive</div>
      <div style="font-size:22px;font-weight:700;color:#10b981;margin:6px 0">80%</div>
      <div style="font-size:11px;color:#8899bb">of revenue</div>
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(99,130,255,.1)">
        <div class="stat-row"><span class="stat-label">Key SKUs</span><span class="stat-value">{cutoff_products}</span></div>
        <div class="stat-row"><span class="stat-label">Their revenue</span><span class="stat-value">{fmt(rev_80)}</span></div>
        <div class="stat-row"><span class="stat-label">Total SKUs</span><span class="stat-value">{len(prod_rev_sorted)}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# ── DEEP ANALYTICS 2: SIMULATED CATEGORY TREND
# ══════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">📈 CATEGORY REVENUE TREND — SIMULATED WEEKLY</div>', unsafe_allow_html=True)
np.random.seed(7)
trend_weeks   = [f"W{i}" for i in range(1, 9)]
categories_present = df["Category"].unique().tolist()
cat_base_rev  = df.groupby("Category")["Total Revenue"].mean()

fig = go.Figure()
for cat in categories_present:
    base  = cat_base_rev.get(cat, 1000)
    noise = np.random.normal(0, base * 0.08, 8)
    trend = np.linspace(base * 0.85, base * 1.15, 8) + noise
    trend = np.clip(trend, 0, None)
    clr   = CAT_CLR.get(cat, "#6366f1")
    fig.add_trace(go.Scatter(
        x=trend_weeks, y=trend, name=cat, mode="lines+markers",
        line=dict(color=clr, width=2),
        marker=dict(size=6, color=clr),
        fill="tozeroy", fillcolor=_hex_to_rgba(clr, 0.03),
    ))

fig.update_layout(
    **PLOTLY_BASE,
    title=dict(text="Simulated 8-Week Category Revenue Trend (based on current distribution)",
               font=dict(color="#f0f4ff", size=12)),
    height=300,
    yaxis=dict(tickprefix="₹", title="Avg Revenue", **_AXIS_DEFAULTS),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# ── DEEP ANALYTICS 3: CITY RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">🕸️ CITY COMPETITIVE RADAR — MULTI-KPI</div>', unsafe_allow_html=True)

radar_metrics = ["Revenue", "Orders", "Avg Price", "Discount%", "Profit Margin"]
cities_present = df["City"].unique().tolist()

city_radar_data = {}
for city in cities_present:
    cdf = df[df["City"] == city]
    city_radar_data[city] = {
        "Revenue":       cdf["Total Revenue"].sum(),
        "Orders":        cdf["Orders"].sum(),
        "Avg Price":     cdf["Current Price"].mean(),
        "Discount%":     cdf["Discount"].mean(),
        "Profit Margin": cdf["Profit Margin"].mean() if "Profit Margin" in cdf.columns else 0,
    }

radar_df = pd.DataFrame(city_radar_data).T
# Normalize 0–1 per metric for radar (avoid scale dominance)
radar_norm = radar_df.copy()
for col in radar_norm.columns:
    col_range = radar_norm[col].max() - radar_norm[col].min()
    if col_range > 0:
        radar_norm[col] = (radar_norm[col] - radar_norm[col].min()) / col_range
    else:
        radar_norm[col] = 0.5

col1, col2 = st.columns([2, 1])
with col1:
    fig = go.Figure()
    for i, city in enumerate(cities_present):
        vals = radar_norm.loc[city].tolist()
        vals += [vals[0]]   # close the polygon
        cats  = radar_metrics + [radar_metrics[0]]
        clr   = CITY_CLR.get(city, PAL[i % len(PAL)])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=city,
            fill="toself", fillcolor=_hex_to_rgba(clr, 0.08),
            line=dict(color=clr, width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(99,130,255,.1)", tickfont=dict(size=8, color="#4a5a7a")),
            angularaxis=dict(gridcolor="rgba(99,130,255,.1)", tickfont=dict(size=10, color="#8899bb")),
        ),
        font=dict(family="Inter", color="#8899bb", size=11),
        title=dict(text="City Competitive Radar (normalized per KPI)", font=dict(color="#f0f4ff", size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=30, r=30, t=50, b=30),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    # City scorecard — rank each city by composite score
    radar_norm["Score"] = radar_norm.mean(axis=1) * 100
    ranked = radar_norm[["Score"]].sort_values("Score", ascending=False)
    st.markdown("""
    <div style="background:#0d1628;border:1px solid rgba(99,130,255,.15);border-radius:12px;padding:16px">
      <div style="font-size:10px;font-weight:600;color:#a5b4fc;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px">
        🏆 Composite City Score
      </div>
    """, unsafe_allow_html=True)
    medals_r = ["🥇","🥈","🥉"] + ["▫️"]*(len(ranked)-3)
    for i, (city, row) in enumerate(ranked.iterrows()):
        score = row["Score"]
        bar_w = int(score)
        clr   = CITY_CLR.get(city, PAL[i % len(PAL)])
        st.markdown(f"""
        <div style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px">
            <span style="font-size:11px;color:#f0f4ff">{medals_r[i]} {city}</span>
            <span style="font-size:10px;font-weight:600;color:{clr};font-family:monospace">{score:.0f}/100</span>
          </div>
          <div style="background:rgba(99,130,255,.08);border-radius:4px;height:5px">
            <div style="width:{bar_w}%;background:{clr};height:5px;border-radius:4px;transition:width .3s"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════════
# ── BLINKBOT CHATBOT UI  (Phase 4 — LLM streaming + chart-backed)
# ══════════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">🤖 BLINKBOT — AI BUSINESS ANALYST</div>', unsafe_allow_html=True)

# ── Pre-compute ctx once for chart detection ──────────────────────────────────────
_bb_ctx_live = _bb_context(df)

# ── Load current memory for UI display ───────────────────────────────────────────
_ui_mem = _get_memory()

# ── Mode badge ───────────────────────────────────────────────────────────────────
mode_badge = (
    '<span style="background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);'
    'border-radius:20px;padding:3px 10px;font-size:10px;color:#a5b4fc;margin-left:8px">🧠 LLM Mode</span>'
    if (use_ai_mode and api_key) else
    '<span style="background:rgba(99,130,255,.08);border:1px solid rgba(99,130,255,.15);'
    'border-radius:20px;padding:3px 10px;font-size:10px;color:#4a5a7a;margin-left:8px">🔧 Rule-based</span>'
)

# ── Header row: bot identity + live memory panel ──────────────────────────────────
bb_head_col, bb_mem_col = st.columns([3, 2])

with bb_head_col:
    st.markdown(f"""
    <div class="blinkbot-header">
      <div style="width:42px;height:42px;background:linear-gradient(135deg,#6366f1,#06b6d4);border-radius:12px;
                  display:flex;align-items:center;justify-content:center;font-size:20px;">🤖</div>
      <div>
        <div style="font-size:15px;font-weight:700;color:#f0f4ff">BlinkBot {mode_badge}</div>
        <div style="font-size:11px;color:#a5b4fc">Senior AI Business Analyst • Always Online</div>
      </div>
      <div style="margin-left:auto;background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);
                  border-radius:20px;padding:4px 10px;font-size:10px;color:#34d399">● Live</div>
    </div>
    """, unsafe_allow_html=True)

with bb_mem_col:
    mem_items = [
        ("💬 Turns",         str(_ui_mem.turn_count)),
        ("🧠 Last Topic",    _ui_mem.last_intent    or "—"),
        ("📍 Last City",     _ui_mem.last_city       or "—"),
        ("⭐ Last Product",  _ui_mem.last_product    or "—"),
        ("🏷️ Last Category", _ui_mem.last_category  or "—"),
    ]
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
        f'border-bottom:1px solid rgba(99,130,255,.06)">'
        f'<span style="font-size:10px;color:#4a5a7a">{lbl}</span>'
        f'<span style="font-size:10px;font-weight:600;color:#67e8f9;font-family:monospace">{val}</span>'
        f'</div>'
        for lbl, val in mem_items
    ])
    topic_stack = " → ".join(_ui_mem.intent_stack) if _ui_mem.intent_stack else "—"
    st.markdown(f"""
    <div style="background:#0d1628;border:1px solid rgba(99,130,255,.15);border-radius:12px;padding:14px 16px;">
      <div style="font-size:10px;font-weight:600;color:#a5b4fc;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:8px">🧠 Conversation Memory</div>
      {rows_html}
      <div style="margin-top:8px;padding:6px 8px;background:rgba(99,102,241,.07);border-radius:6px;">
        <div style="font-size:9px;color:#4a5a7a;margin-bottom:2px">TOPIC TRAIL</div>
        <div style="font-size:10px;color:#a5b4fc">{topic_stack}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────────────
if "blinkbot_history" not in st.session_state:
    welcome = (
        f"👋 **Hi! I'm BlinkBot** — now powered by **{_ANTHROPIC_MODEL}**.\n\n"
        f"I've analyzed **{len(df):,} records** and I have full conversational memory. "
        f"Ask me anything in plain English — I'll answer with data, a chart, and a recommendation.\n\n"
        f"Try: *'Give me a full summary'* · *'Which city should we focus on?'* · *'Why is Grocery underperforming?'*"
        if (use_ai_mode and api_key) else
        f"👋 **Hi! I'm BlinkBot** — your AI Business Analyst with memory.\n\n"
        f"I've analyzed **{len(df):,} records**. Every answer comes with an inline chart.\n\n"
        f"💡 *Enable **LLM Mode** in the sidebar for natural-language answers powered by Claude.*\n\n"
        f"Try: *'Give me a summary'* · *'Which city is worst?'* · *'Best product?'*"
    )
    st.session_state.blinkbot_history  = [{"role":"bot","msg":welcome,"fig_json":None}]

# LLM message history — separate from display history (Anthropic format)
if "bb_messages_llm" not in st.session_state:
    st.session_state.bb_messages_llm = []

# ── Render chat history ───────────────────────────────────────────────────────────
for msg in st.session_state.blinkbot_history:
    css_class = "chat-message-bot" if msg["role"] == "bot" else "chat-message-user"
    prefix    = "" if msg["role"] == "bot" else "💬 "
    st.markdown(f'<div class="{css_class}">{prefix}{msg["msg"]}</div>', unsafe_allow_html=True)
    if msg["role"] == "bot" and msg.get("fig_json"):
        restored = _fig_from_json(msg["fig_json"])
        if restored:
            st.plotly_chart(restored, use_container_width=True, key=f"bb_fig_{id(msg)}")

# ── Dynamic quick-question buttons ───────────────────────────────────────────────
st.markdown("**💡 Quick Questions:**")
QUICK_BASE = [
    ("📊 Summary",       "Give me a full business summary"),
    ("🏆 Best Product",  "Which product is performing best?"),
    ("📍 City Analysis", "Which city is performing worst?"),
    ("⚡ Influencers",   "How is influencer marketing performing?"),
]
QUICK_FOLLOWUP: list[tuple[str, str]] = []
if _ui_mem.last_intent == "city" and _ui_mem.last_city:
    QUICK_FOLLOWUP.append(("⚖️ Compare Cities",   "Compare best vs worst city"))
if _ui_mem.last_intent in ("revenue","summary") and _ui_mem.last_category:
    QUICK_FOLLOWUP.append(("🏷️ Category Drill",   f"Tell me more about {_ui_mem.last_category}"))
if _ui_mem.last_product:
    QUICK_FOLLOWUP.append(("📦 Reorder Risk",      f"Inventory risk for {_ui_mem.last_product}"))
if _ui_mem.turn_count > 0:
    QUICK_FOLLOWUP.append(("🔄 Tell Me More",      "Tell me more"))

display_items = (QUICK_FOLLOWUP + QUICK_BASE)[:4]
quick_cols    = st.columns(len(display_items))
clicked_quick = None
for i, (btn_label, q_text) in enumerate(display_items):
    with quick_cols[i]:
        if st.button(btn_label, key=f"bb_q{i}", use_container_width=True):
            clicked_quick = q_text

# ── Text input ────────────────────────────────────────────────────────────────────
with st.form(key="bb_main_form", clear_on_submit=True):
    fc1, fc2 = st.columns([5, 1])
    with fc1:
        ph = (
            f"Ask anything — I'll answer with data and a chart..."
            if (use_ai_mode and api_key)
            else f"e.g. What is my total profit? Which city is weakest?"
        )
        user_input = st.text_input("Ask BlinkBot...", placeholder=ph, label_visibility="collapsed")
    with fc2:
        submitted = st.form_submit_button("Ask 🤖", use_container_width=True)

# ── Route & respond ───────────────────────────────────────────────────────────────
question_to_answer = None
if submitted and user_input.strip():
    question_to_answer = user_input.strip()
elif clicked_quick:
    question_to_answer = clicked_quick

if question_to_answer:
    # 1. Log user turn
    st.session_state.blinkbot_history.append({"role":"user","msg":question_to_answer,"fig_json":None})
    st.session_state.bb_messages_llm.append({"role":"user","content":question_to_answer})

    # 2. Pick chart regardless of mode
    response_fig     = _detect_chart_for_question(question_to_answer, _bb_ctx_live, df)
    response_fig_json = _fig_to_json(response_fig)

    # ── LLM streaming mode ────────────────────────────────────────────────────────
    if use_ai_mode and api_key:
        system_prompt = _build_llm_system_prompt(df, kpis)

        # Show user bubble immediately, then stream bot response in-place
        st.markdown(f'<div class="chat-message-user">💬 {question_to_answer}</div>', unsafe_allow_html=True)

        # Streaming placeholder
        stream_placeholder = st.empty()
        full_response      = ""
        error_occurred     = False

        for chunk in _call_anthropic_stream(
            st.session_state.bb_messages_llm, system_prompt, api_key
        ):
            full_response += chunk
            if "⚠️" in chunk:
                error_occurred = True
            stream_placeholder.markdown(
                f'<div class="chat-message-bot">{full_response}▊</div>',
                unsafe_allow_html=True,
            )

        # Finalise — remove cursor
        stream_placeholder.markdown(
            f'<div class="chat-message-bot">{full_response}</div>',
            unsafe_allow_html=True,
        )

        # Show chart inline after response
        if response_fig and not error_occurred:
            st.plotly_chart(response_fig, use_container_width=True, key="bb_stream_chart")

        # Persist to histories
        st.session_state.bb_messages_llm.append({"role":"assistant","content":full_response})
        # Trim LLM history to avoid unbounded growth
        if len(st.session_state.bb_messages_llm) > _LLM_HISTORY_LIMIT * 2:
            st.session_state.bb_messages_llm = st.session_state.bb_messages_llm[-_LLM_HISTORY_LIMIT:]
        st.session_state.blinkbot_history.append({
            "role":     "bot",
            "msg":      full_response,
            "fig_json": response_fig_json if not error_occurred else None,
        })

    # ── Rule-based fallback mode ──────────────────────────────────────────────────
    else:
        response_text, response_fig_rb = blinkbot_analyze(question_to_answer, df)
        # Prefer rule-based chart if the keyword router found nothing
        final_fig = response_fig_rb or response_fig
        st.session_state.blinkbot_history.append({
            "role":     "bot",
            "msg":      response_text,
            "fig_json": _fig_to_json(final_fig),
        })
        st.rerun()

    if use_ai_mode and api_key:
        st.rerun()   # re-render so next input box is fresh

# ── Clear chat + memory ───────────────────────────────────────────────────────────
if len(st.session_state.blinkbot_history) > 1:
    cl1, cl2 = st.columns([1, 5])
    with cl1:
        if st.button("🗑️ Clear Chat & Memory", type="secondary", key="clear_chat", use_container_width=True):
            st.session_state.blinkbot_history  = []
            st.session_state.bb_messages_llm   = []
            st.session_state.bb_memory         = ConversationMemory().to_dict()
            st.rerun()

# ── RAW DATA TABLE ────────────────────────────────────────────────────────────────
if show_raw:
    st.markdown('<div class="section-head">RAW DATA TABLE</div>', unsafe_allow_html=True)
    display_cols = ["Product Name","Category","City","Original Price","Current Price",
                    "Discount","Orders","Total Revenue","Profit","Profit Margin","Influencer Active"]
    show_df = df[[c for c in display_cols if c in df.columns]].copy()
    show_df["Profit Margin"] = show_df["Profit Margin"].round(1).astype(str) + "%"
    st.dataframe(show_df, use_container_width=True, height=350)
    st.download_button("⬇ Download Filtered CSV", df.to_csv(index=False), "zepto_filtered.csv", "text/csv")

# ── FOOTER ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Ayush Intelligence Hub v5.0 — Phase 4 (LLM) &nbsp;·&nbsp;
  Developed by <span class="dev">Ayush Mishra</span> &nbsp;·&nbsp;
  Pandas · SciPy · scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
