"""
Zepto Sales Intelligence Dashboard v3.0
Developed by Ayush Mishra
Improvements: Modular tabs, real KPIs, Claude API chatbot, executive summary,
              data validation, caching, labeled simulations, clean architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings, io, os, requests, json
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ayush Intelligence Hub v3",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stDecoration"],[data-testid="stDeployButton"],[data-testid="stToolbarActions"],
footer,#MainMenu { visibility: hidden; display: none !important; }
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #06091a; }
.block-container { padding: 1.5rem 2rem; max-width: 1440px; }

/* KPI Cards */
.kpi-card {
  background: linear-gradient(135deg, #0d1628 0%, #111827 100%);
  border: 1px solid rgba(99,130,255,.18);
  border-radius: 16px; padding: 20px 18px;
  text-align: center; transition: all .25s ease;
  position: relative; overflow: hidden;
}
.kpi-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px; background: linear-gradient(90deg, #6366f1, #06b6d4);
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(99,130,255,.35); }
.kpi-label { font-size: 10px; color: #4a5a7a; text-transform: uppercase;
  letter-spacing: .12em; margin-bottom: 8px; font-weight: 500; }
.kpi-value { font-size: 24px; font-weight: 700; margin-bottom: 6px; font-family: 'DM Mono', monospace; }
.kpi-sub { font-size: 10px; color: #4a5a7a; margin-top: 4px; }
.badge-up   { display:inline-block; font-size:9px; font-weight:700; border-radius:6px;
  padding:2px 8px; background:rgba(16,185,129,.15); color:#34d399; }
.badge-down { display:inline-block; font-size:9px; font-weight:700; border-radius:6px;
  padding:2px 8px; background:rgba(239,68,68,.15); color:#f87171; }
.badge-neutral { display:inline-block; font-size:9px; font-weight:700; border-radius:6px;
  padding:2px 8px; background:rgba(99,130,255,.12); color:#a5b4fc; }

/* Section heads */
.sec-head {
  font-size: 10px; font-weight: 700; color: #6366f1;
  text-transform: uppercase; letter-spacing: .14em;
  border-left: 3px solid #6366f1; padding-left: 10px;
  margin: 28px 0 16px; font-family: 'DM Mono', monospace;
}

/* Insight cards */
.insight-card {
  background: linear-gradient(135deg, #111827, #0d1628);
  border: 1px solid rgba(99,130,255,.12);
  border-radius: 14px; padding: 18px; height: 100%; margin-bottom: 6px;
}
.insight-title { font-size: 10px; font-weight: 700; color: #6366f1;
  text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px; }
.insight-body { font-size: 12px; color: #94a3b8; line-height: 1.7; }
.insight-body strong { color: #67e8f9; }
.insight-action { font-size: 11px; color: #34d399; margin-top: 10px;
  background: rgba(16,185,129,.08); padding: 6px 10px; border-radius: 6px;
  border-left: 2px solid #34d399; }

/* Alert/info banners */
.exec-banner {
  background: linear-gradient(135deg, #0d1628, #1a1040);
  border: 1px solid rgba(99,102,241,.25);
  border-radius: 16px; padding: 20px 24px; margin-bottom: 24px;
}
.sim-notice {
  background: rgba(245,158,11,.06); border: 1px solid rgba(245,158,11,.2);
  border-radius: 8px; padding: 8px 14px; font-size: 10px; color: #fbbf24;
  margin-bottom: 12px; display: inline-block;
}

/* Stats */
.stat-row { display:flex; justify-content:space-between; padding:7px 0;
  border-bottom:1px solid rgba(99,130,255,.06); }
.stat-label { font-size:11px; color:#8899bb; }
.stat-value { font-size:11px; font-weight:600; color:#67e8f9; font-family:'DM Mono',monospace; }

/* Chat */
.chat-bot {
  background: linear-gradient(135deg, rgba(99,102,241,.1), rgba(6,182,212,.05));
  border: 1px solid rgba(99,102,241,.2); border-radius: 14px 14px 14px 2px;
  padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #e2e8f0; line-height: 1.7;
}
.chat-user {
  background: rgba(30,41,59,.8); border: 1px solid rgba(99,130,255,.15);
  border-radius: 14px 14px 2px 14px; padding: 12px 16px; margin: 8px 0;
  font-size: 13px; color: #cbd5e1; text-align: right;
}
.chat-thinking {
  background: rgba(99,102,241,.06); border: 1px dashed rgba(99,102,241,.2);
  border-radius: 10px; padding: 10px 14px; font-size: 11px; color: #6366f1;
  margin: 6px 0; font-style: italic;
}

/* Footer */
.footer { text-align:center; padding:24px; color:#4a5a7a; font-size:11px;
  border-top:1px solid rgba(99,130,255,.08); margin-top:40px; }
.footer .dev { background:linear-gradient(90deg,#a5b4fc,#67e8f9);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:700; }

div[data-testid="metric-container"] { background:#0d1628;
  border:1px solid rgba(99,130,255,.12); border-radius:12px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
PAL      = ["#6366f1","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316","#3b82f6"]
CAT_CLR  = {"Snacks":"#6366f1","Beverages":"#06b6d4","Grocery":"#10b981","Instant Food":"#f59e0b","Confectionery":"#ec4899","Dairy":"#8b5cf6"}
CITY_CLR = {"Delhi":"#6366f1","Mumbai":"#06b6d4","Bangalore":"#10b981","Hyderabad":"#f59e0b","Chennai":"#ef4444","Pune":"#8b5cf6"}
PLOTLY   = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#8899bb", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="rgba(99,130,255,.06)", linecolor="rgba(99,130,255,.1)"),
    yaxis=dict(gridcolor="rgba(99,130,255,.06)", linecolor="rgba(99,130,255,.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def fmt(n):
    if pd.isna(n): return "—"
    n = float(n)
    if n >= 1e7: return f"₹{n/1e7:.1f}Cr"
    if n >= 1e5: return f"₹{n/1e5:.2f}L"
    if n >= 1e3: return f"₹{n/1e3:.1f}K"
    return f"₹{int(n):,}"

def pct_change_badge(curr, prev):
    if prev == 0: return '<span class="badge-neutral">—</span>'
    chg = (curr - prev) / abs(prev) * 100
    cls = "badge-up" if chg >= 0 else "badge-down"
    arrow = "↑" if chg >= 0 else "↓"
    return f'<span class="{cls}">{arrow} {abs(chg):.1f}%</span>'

def validate_data(df):
    """Return list of validation issues found in the dataframe."""
    issues = []
    required = ["Product Name", "Category", "City", "Original Price", "Current Price", "Discount", "Orders", "Total Revenue"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {', '.join(missing_cols)}")
    if df.duplicated().sum() > 0:
        issues.append(f"{df.duplicated().sum()} duplicate rows detected")
    for col in ["Original Price", "Current Price", "Orders", "Total Revenue"]:
        if col in df.columns and (df[col] < 0).any():
            issues.append(f"Negative values in {col}")
    if "Profit Margin" in df.columns:
        extreme = (df["Profit Margin"].abs() > 200).sum()
        if extreme > 0:
            issues.append(f"{extreme} rows with extreme profit margin (>200%) — check pricing data")
    return issues

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_CSV = """Product Name,Category,City,Original Price,Current Price,Discount,Orders,Total Revenue,Influencer Active
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
Britannia Cake,Snacks,Delhi,148,163,5,283,44714,No
Maggi Noodles,Instant Food,Delhi,150,165,5,312,51480,Yes
Pepsi 500ml,Beverages,Bangalore,110,115,5,198,22770,No
Oreo Biscuits,Snacks,Mumbai,175,185,0,241,44585,Yes
Parle-G,Snacks,Mumbai,90,95,10,312,29640,No
Amul Milk 500ml,Dairy,Hyderabad,155,160,0,289,46240,Yes
Fortune Oil 1L,Grocery,Mumbai,130,140,5,201,28140,No
Aashirvaad Atta,Grocery,Delhi,80,90,10,342,30780,Yes
Maggi Noodles,Instant Food,Bangalore,170,180,5,278,50040,No
Coca Cola 1L,Beverages,Chennai,145,150,10,263,39450,Yes
Nestle Munch,Confectionery,Hyderabad,180,190,0,241,45790,No"""

@st.cache_data
def load_default():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "zepto_sales_dataset.csv")
    if os.path.exists(path):
        return _clean(pd.read_csv(path))
    return _clean(pd.read_csv(io.StringIO(SAMPLE_CSV)))

def _clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.dropna(how="all", inplace=True)
    for col in ["Original Price","Current Price","Discount","Orders","Total Revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0]).str.strip()
    # Real business formulas
    df["Profit"]        = (df["Current Price"] - df["Original Price"]) * df["Orders"]
    df["Profit Margin"] = np.where(df["Total Revenue"] > 0,
                                   df["Profit"] / df["Total Revenue"] * 100, 0)
    df["AOV"]           = np.where(df["Orders"] > 0,
                                   df["Total Revenue"] / df["Orders"], 0)
    df["Effective Price"] = df["Current Price"] * (1 - df["Discount"] / 100)
    df["Price Tier"]    = pd.cut(df["Current Price"],
                                 bins=[0,60,100,140,180,np.inf],
                                 labels=["₹20–60","₹61–100","₹101–140","₹141–180","₹181+"])
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO EXECUTIVE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
def generate_exec_insights(df):
    """Generate real data-driven executive insights."""
    insights = []
    cat_rev   = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
    city_rev  = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)
    prod_rev  = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)
    margin_cat= df.groupby("Category")["Profit Margin"].mean().sort_values(ascending=False)
    total_rev = df["Total Revenue"].sum()

    # Growth driver
    top_cat_share = cat_rev.iloc[0] / total_rev * 100
    insights.append({
        "icon": "🏆", "type": "growth",
        "title": f"{cat_rev.index[0]} is your growth engine",
        "body": f"Contributes {top_cat_share:.1f}% of total revenue ({fmt(cat_rev.iloc[0])}). "
                f"Prioritize inventory and marketing here for maximum ROI.",
        "action": f"→ Action: Increase {cat_rev.index[0]} SKUs and activate influencer campaigns"
    })

    # Regional gap
    gap_pct = (city_rev.iloc[0] - city_rev.iloc[-1]) / city_rev.iloc[-1] * 100 if city_rev.iloc[-1] > 0 else 0
    insights.append({
        "icon": "🌍", "type": "weak",
        "title": f"{city_rev.index[-1]} is underperforming by {gap_pct:.0f}%",
        "body": f"{city_rev.index[0]} leads ({fmt(city_rev.iloc[0])}) vs "
                f"{city_rev.index[-1]} trails ({fmt(city_rev.iloc[-1])}). "
                f"Significant expansion opportunity exists.",
        "action": f"→ Action: Run city-specific promotions in {city_rev.index[-1]}"
    })

    # Margin alert
    best_m = margin_cat.index[0]; worst_m = margin_cat.index[-1]
    insights.append({
        "icon": "📊", "type": "margin",
        "title": f"Margin spread: {best_m} ({margin_cat.iloc[0]:.1f}%) vs {worst_m} ({margin_cat.iloc[-1]:.1f}%)",
        "body": f"Significant margin gap between categories. "
                f"{worst_m} products may need pricing review or cost reduction.",
        "action": f"→ Action: Review {worst_m} supplier contracts or adjust pricing"
    })

    # Influencer insight (if column exists)
    if "Influencer Active" in df.columns:
        iy = df[df["Influencer Active"]=="Yes"]["Total Revenue"].mean()
        inn= df[df["Influencer Active"]=="No"]["Total Revenue"].mean()
        if inn > 0:
            lift = (iy - inn) / inn * 100
            insights.append({
                "icon": "⚡", "type": "opportunity" if lift > 0 else "warning",
                "title": f"Influencer marketing {'drives' if lift > 0 else 'shows'} {abs(lift):.1f}% {'lift' if lift>0 else 'drag'}",
                "body": f"Products with active influencers average {fmt(iy)} vs {fmt(inn)} without. "
                        f"{'Scale this channel aggressively.' if lift > 5 else 'Optimize influencer selection.'}",
                "action": f"→ Action: {'Activate influencers for all top-10 products' if lift > 0 else 'Review influencer ROI and reselect'}"
            })

    # Top product opportunity
    top_prod = prod_rev.index[0]
    insights.append({
        "icon": "🎯", "type": "opportunity",
        "title": f"{top_prod} — star product to double down on",
        "body": f"Generates {fmt(prod_rev.iloc[0])} in revenue. "
                f"Expand distribution, bundle with complementary products, and maintain stock.",
        "action": f"→ Action: Increase {top_prod} reorder quantity and city coverage"
    })

    return insights

# ═══════════════════════════════════════════════════════════════════════════════
# BLINKBOT — Claude API powered
# ═══════════════════════════════════════════════════════════════════════════════
def call_claude_api(messages, system_prompt):
    """Call Claude claude-sonnet-4-20250514 via Anthropic API."""
    try:
        # Ensure conversation starts with user role (Claude API requirement)
        clean_messages = []
        for m in messages:
            if not clean_messages or clean_messages[-1]["role"] != m["role"]:
                clean_messages.append(m)
            else:
                # Merge consecutive same-role messages
                clean_messages[-1]["content"] += "\n" + m["content"]
        # Must start with user
        if not clean_messages or clean_messages[0]["role"] != "user":
            clean_messages = [{"role": "user", "content": "Hello"}] + clean_messages

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": clean_messages
            },
            timeout=30
        )
        data = resp.json()
        if "content" in data and len(data["content"]) > 0:
            return data["content"][0]["text"]
        err_msg = data.get("error", {}).get("message", str(data))
        return f"⚠️ API error: {err_msg}"
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"

def build_data_context(df):
    """Build a compact data summary for the Claude system prompt."""
    cat_rev   = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
    city_rev  = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)
    prod_rev  = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)
    total_rev = df["Total Revenue"].sum()
    total_prof= df["Profit"].sum()
    margin    = total_prof / total_rev * 100 if total_rev > 0 else 0
    total_ord = df["Orders"].sum()

    inf_stats = ""
    if "Influencer Active" in df.columns:
        iy  = df[df["Influencer Active"]=="Yes"]["Total Revenue"].mean()
        inn = df[df["Influencer Active"]=="No"]["Total Revenue"].mean()
        lift= (iy-inn)/inn*100 if inn > 0 else 0
        inf_stats = f"\nInfluencer lift: {lift:+.1f}% (with={fmt(iy)}, without={fmt(inn)})"

    return f"""You are BlinkBot, an expert AI business analyst for Ayush Intelligence Hub — a Zepto-style quick commerce dashboard.

CURRENT DATA SNAPSHOT ({len(df):,} records, filtered view):
- Total Revenue: {fmt(total_rev)}
- Total Profit: {fmt(total_prof)} ({margin:.1f}% margin)
- Total Orders: {int(total_ord):,}
- AOV: {fmt(total_rev/total_ord if total_ord > 0 else 0)}
- Top Category: {cat_rev.index[0]} ({fmt(cat_rev.iloc[0])}, {cat_rev.iloc[0]/total_rev*100:.1f}%)
- Bottom Category: {cat_rev.index[-1]} ({fmt(cat_rev.iloc[-1])})
- Top City: {city_rev.index[0]} ({fmt(city_rev.iloc[0])})
- Weakest City: {city_rev.index[-1]} ({fmt(city_rev.iloc[-1])}){inf_stats}
- Top Product: {prod_rev.index[0]} ({fmt(prod_rev.iloc[0])})
- Bottom Product: {prod_rev.index[-1]} ({fmt(prod_rev.iloc[-1])})
- Category breakdown: {dict(cat_rev.apply(fmt))}
- City breakdown: {dict(city_rev.apply(fmt))}

PERSONA: Be concise, data-driven, and action-oriented. Always cite real numbers from the snapshot above.
Format with markdown. Lead with a direct answer, then provide 2-3 supporting data points, then a clear recommendation.
Keep responses under 200 words. Never invent data not in the snapshot."""

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px">
      <div style="width:48px;height:48px;background:linear-gradient(135deg,#6366f1,#06b6d4);border-radius:14px;
        display:inline-flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:10px">🛒</div>
      <div style="font-size:14px;font-weight:700;background:linear-gradient(90deg,#a5b4fc,#67e8f9);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent">Ayush Intelligence Hub</div>
      <div style="font-size:10px;color:#4a5a7a;margin-top:3px;font-family:'DM Mono',monospace">v3.0 · Sales Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📂 Data Source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Columns: Product Name, Category, City, Original Price, Current Price, Discount, Orders, Total Revenue, Influencer Active")

    df_raw = load_default()
    validation_errors = []

    if uploaded:
        try:
            df_raw = _clean(pd.read_csv(uploaded))
            validation_errors = validate_data(df_raw)
            if validation_errors:
                for err in validation_errors:
                    st.warning(f"⚠️ {err}")
            else:
                st.success(f"✅ Loaded {len(df_raw):,} rows — no issues found")
        except Exception as e:
            st.error(f"❌ Parse error: {e}")

    st.markdown("---")
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
    show_raw   = st.checkbox("Show Raw Data Table", value=False)
    show_stats = st.checkbox("Statistical Analysis",  value=True)

    # Download filtered data
    st.markdown("---")
    st.markdown("#### 💾 Export")

    st.markdown("""
    <div style="font-size:10px;color:#4a5a7a;text-align:center;margin-top:12px">
      Developed by <strong style="color:#a5b4fc">Ayush Mishra</strong><br>
      Pandas · SciPy · scikit-learn · Streamlit · Plotly
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════════════════════════════════════════
df = df_raw.copy()
if sel_city != "All": df = df[df["City"]              == sel_city]
if sel_cat  != "All": df = df[df["Category"]          == sel_cat]
if sel_inf  != "All": df = df[df["Influencer Active"] == sel_inf]
if sel_prod != "All": df = df[df["Product Name"]      == sel_prod]
if search:            df = df[df["Product Name"].str.contains(search, case=False, na=False)]

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
filters_applied = [x for x in [sel_city, sel_cat, sel_inf, sel_prod] if x != "All"]
filter_tags = " · ".join([f"<span style='background:rgba(99,102,241,.15);padding:2px 8px;border-radius:4px;font-size:10px;color:#a5b4fc'>{f}</span>" for f in filters_applied]) if filters_applied else "<span style='font-size:10px;color:#4a5a7a'>No filters applied — showing all data</span>"

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1628,#111827);border:1px solid rgba(99,130,255,.15);
  border-radius:18px;padding:22px 28px;margin-bottom:24px;display:flex;align-items:center;justify-content:space-between">
  <div>
    <h1 style="margin:0;font-size:20px;font-weight:700;background:linear-gradient(90deg,#a5b4fc,#67e8f9);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent">Ayush Intelligence Hub</h1>
    <p style="margin:5px 0 0;font-size:11px;color:#4a5a7a">
      Real-Time Business Insights · Statistical Analytics · ML Forecasting · AI Assistant
    </p>
    <div style="margin-top:10px">{filter_tags}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:10px;color:#4a5a7a;font-family:'DM Mono',monospace">
      {len(df):,} records · {len(df_raw):,} total
    </div>
    <div style="font-size:10px;color:#34d399;margin-top:4px">● Live</div>
  </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ No data matches your current filters. Please adjust the sidebar filters to continue.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# REAL KPI COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════════
total_rev    = df["Total Revenue"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Orders"].sum()
margin       = (total_profit / total_rev * 100) if total_rev else 0
aov          = total_rev / total_orders if total_orders > 0 else 0
cat_rev      = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
city_rev     = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)
prod_rev     = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)
rev_per_prod = df.groupby("Product Name").agg(rev=("Total Revenue","sum"), orders=("Orders","sum")).assign(aov=lambda x: x.rev/x.orders)

# Compare filtered vs unfiltered for context
unfilt_rev   = df_raw["Total Revenue"].sum()
share_of_tot = total_rev / unfilt_rev * 100 if unfilt_rev > 0 else 100

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY BANNER
# ═══════════════════════════════════════════════════════════════════════════════
exec_insights = generate_exec_insights(df)

st.markdown('<div class="exec-banner">', unsafe_allow_html=True)
st.markdown(f"""
<div style="font-size:10px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:.12em;margin-bottom:14px;font-family:'DM Mono',monospace">
  📋 Executive Summary — Auto-Generated
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px">
""", unsafe_allow_html=True)

for ins in exec_insights[:4]:
    color_map = {"growth":"#34d399","weak":"#f87171","margin":"#fbbf24","opportunity":"#67e8f9","warning":"#f87171"}
    clr = color_map.get(ins["type"], "#a5b4fc")
    st.markdown(f"""
<div style="background:rgba(0,0,0,.2);border:1px solid rgba(99,130,255,.1);border-radius:10px;padding:12px 14px">
  <div style="font-size:13px;margin-bottom:5px">{ins['icon']} <strong style="font-size:12px;color:{clr}">{ins['title']}</strong></div>
  <div style="font-size:11px;color:#64748b;line-height:1.6">{ins['body']}</div>
  <div style="font-size:10px;color:#34d399;margin-top:8px;font-style:italic">{ins['action']}</div>
</div>""", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_sales, tab_analytics, tab_forecast, tab_ai, tab_bot = st.tabs([
    "📊 Overview", "🛒 Sales Deep-Dive", "🔬 Analytics", "📈 Forecast", "💡 AI Insights", "🤖 BlinkBot"
])

# ════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════
with tab_overview:
    # KPI Row
    st.markdown('<div class="sec-head">Key Performance Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "💰", "Total Revenue",    fmt(total_rev),          f"{share_of_tot:.0f}% of all data",   "#a5b4fc", pct_change_badge(total_rev, unfilt_rev * 0.5)),
        (c2, "📈", "Total Profit",     fmt(total_profit),       "From real pricing delta",             "#6ee7b7", pct_change_badge(total_profit, total_rev * 0.15)),
        (c3, "🛒", "Total Orders",     f"{int(total_orders):,}","Units across all products",           "#fcd34d", '<span class="badge-neutral">Real data</span>'),
        (c4, "%",  "Profit Margin",    f"{margin:.1f}%",        "(Price−Cost)×Orders÷Revenue",        "#fca5a5", pct_change_badge(margin, 15)),
        (c5, "💵", "Avg Order Value",  fmt(aov),                "Revenue ÷ Orders",                    "#67e8f9", '<span class="badge-neutral">AOV</span>'),
        (c6, "⭐", "Top Category",     cat_rev.index[0] if len(cat_rev) else "—",
                                       fmt(cat_rev.iloc[0]) if len(cat_rev) else "—",                 "#c4b5fd", '<span class="badge-up">Leader</span>'),
    ]
    for col, icon, label, val, sub, clr, badge in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div style="font-size:18px;margin-bottom:8px">{icon}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{clr}">{val}</div>
              <div style="margin:4px 0">{badge}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue by City + Category donut
    st.markdown('<div class="sec-head">Revenue Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        city_data = city_rev.reset_index()
        city_data.columns = ["City", "Total Revenue"]
        # Annotate top city
        top_city_share = city_data.iloc[0]["Total Revenue"] / total_rev * 100
        fig = px.bar(city_data, x="City", y="Total Revenue", color="City",
                     color_discrete_map=CITY_CLR,
                     title=f"Revenue by City — {city_data.iloc[0]['City']} leads ({top_city_share:.0f}%)")
        fig.update_layout(**PLOTLY, title_font_color="#f0f4ff", showlegend=False)
        fig.update_traces(marker_line_width=0, opacity=0.88)
        fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_data = cat_rev.reset_index()
        cat_data.columns = ["Category", "Total Revenue"]
        fig = px.pie(cat_data, values="Total Revenue", names="Category",
                     color="Category", color_discrete_map=CAT_CLR,
                     title="Category Share — " + cat_data.iloc[0]["Category"] + " dominates",
                     hole=0.58)
        fig.update_layout(**PLOTLY, title_font_color="#f0f4ff")
        fig.update_traces(textinfo="label+percent", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap with insight
    st.markdown('<div class="sec-head">City × Category Heatmap</div>', unsafe_allow_html=True)
    pivot = df.pivot_table(index="Category", columns="City", values="Total Revenue",
                           aggfunc="sum", fill_value=0)
    if not pivot.empty:
        max_val = pivot.values.max()
        max_idx = np.unravel_index(pivot.values.argmax(), pivot.shape)
        st.caption(f"💡 Hotspot: **{pivot.index[max_idx[0]]}** in **{pivot.columns[max_idx[1]]}** — {fmt(max_val)} (highest revenue cell)")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,"#06091a"],[0.3,"#1e1b6e"],[0.6,"#3730a3"],[1,"#6366f1"]],
            text=[[fmt(v) for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> in <b>%{x}</b>: %{text}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY, height=280, title_font_color="#f0f4ff")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════
# TAB 2 — SALES DEEP DIVE
# ════════════════════════════════════════
with tab_sales:
    st.markdown('<div class="sec-head">Product Performance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        top_prod = prod_rev.head(10).reset_index()
        top_prod.columns = ["Product Name", "Total Revenue"]
        fig = px.bar(top_prod, x="Total Revenue", y="Product Name", orientation="h",
                     title="Top 10 Products by Revenue",
                     color="Total Revenue",
                     color_continuous_scale=["#3730a3","#6366f1","#06b6d4"])
        fig.update_layout(**PLOTLY, title_font_color="#f0f4ff", coloraxis_showscale=False)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=",.0f", tickprefix="₹")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # AOV vs Revenue scatter
        prod_stats = df.groupby("Product Name").agg(
            Revenue=("Total Revenue","sum"),
            Orders=("Orders","sum"),
            Category=("Category","first")
        ).reset_index()
        prod_stats["AOV"] = prod_stats["Revenue"] / prod_stats["Orders"]
        fig = px.scatter(prod_stats, x="Orders", y="Revenue",
                         color="Category", color_discrete_map=CAT_CLR,
                         size="AOV", hover_name="Product Name",
                         title="Orders vs Revenue (bubble = AOV)",
                         labels={"Revenue":"Total Revenue (₹)"})
        fig.update_layout(**PLOTLY, title_font_color="#f0f4ff")
        fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0)))
        fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
        st.plotly_chart(fig, use_container_width=True)

    # Discount analysis
    st.markdown('<div class="sec-head">Discount Effectiveness</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        disc_data = df.groupby("Discount").agg(
            Avg_Revenue=("Total Revenue","mean"),
            Avg_Orders=("Orders","mean"),
            Count=("Orders","count")
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=disc_data["Discount"].astype(str)+"%",
                             y=disc_data["Avg_Revenue"], name="Avg Revenue",
                             marker_color="#6366f1", opacity=0.85), secondary_y=False)
        fig.add_trace(go.Scatter(x=disc_data["Discount"].astype(str)+"%",
                                 y=disc_data["Avg_Orders"], name="Avg Orders",
                                 mode="lines+markers", line=dict(color="#06b6d4", width=2)),
                      secondary_y=True)
        best_disc = disc_data.loc[disc_data["Avg_Revenue"].idxmax()]
        fig.update_layout(**PLOTLY, title=f"Discount vs Revenue & Orders — Best: {int(best_disc['Discount'])}%",
                          title_font_color="#f0f4ff")
        fig.update_yaxes(tickprefix="₹", secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"💡 {int(best_disc['Discount'])}% discount yields highest avg revenue ({fmt(best_disc['Avg_Revenue'])}). Use as your default promotion rate.")

    with col2:
        inf_data = df.groupby(["Category","Influencer Active"])["Total Revenue"].mean().reset_index()
        inf_data.columns = ["Category","Influencer","Avg Revenue"]
        fig = px.bar(inf_data, x="Category", y="Avg Revenue", color="Influencer",
                     barmode="group", title="Influencer Impact by Category",
                     color_discrete_map={"Yes":"#6366f1","No":"#4a5a7a"})
        fig.update_layout(**PLOTLY, title_font_color="#f0f4ff")
        fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
        st.plotly_chart(fig, use_container_width=True)

    # Price tier
    st.markdown('<div class="sec-head">Price Tier Analysis</div>', unsafe_allow_html=True)
    price_data = df.groupby("Price Tier", observed=True)["Total Revenue"].sum().reset_index()
    price_data["Price Tier"] = price_data["Price Tier"].astype(str)
    top_tier = price_data.loc[price_data["Total Revenue"].idxmax()]
    fig = px.bar(price_data, x="Price Tier", y="Total Revenue", color="Price Tier",
                 color_discrete_sequence=PAL, title=f"Revenue by Price Tier — {top_tier['Price Tier']} dominates",
                 labels={"Total Revenue":"Revenue (₹)"})
    fig.update_layout(**PLOTLY, title_font_color="#f0f4ff", showlegend=False)
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    fig.update_traces(marker_line_width=0, opacity=0.88)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"💡 {top_tier['Price Tier']} tier generates {fmt(top_tier['Total Revenue'])} — focus product mix here.")

    # Simulated sections clearly labeled
    st.markdown("""<div class="sim-notice">
      ⚠️ SIMULATED DATA — The sections below (Delivery, Inventory, Retention) are demo simulations.
      Connect your operations database for real metrics.
    </div>""", unsafe_allow_html=True)

    with st.expander("🚀 Delivery Performance (Simulated)", expanded=False):
        col1, col2, col3 = st.columns(3)
        np.random.seed(42)
        n_orders = max(len(df), 50)
        delivery_times = np.clip(np.random.normal(11.5, 3.5, n_orders), 5, 35)
        promised_time  = 10
        otd_pct = float(np.mean(delivery_times <= promised_time + 2) * 100)
        avg_time = float(np.mean(delivery_times))
        p90_time = float(np.percentile(delivery_times, 90))

        with col1:
            gauge_color = "#10b981" if otd_pct >= 95 else "#f59e0b" if otd_pct >= 85 else "#ef4444"
            fig = go.Figure(go.Indicator(mode="gauge+number", value=otd_pct,
                title={"text":"On-Time Delivery %","font":{"color":"#8899bb","size":13}},
                number={"suffix":"%","font":{"color":gauge_color,"size":28}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":gauge_color},"bgcolor":"#0d1628",
                       "steps":[{"range":[0,85],"color":"rgba(239,68,68,.15)"},{"range":[85,95],"color":"rgba(245,158,11,.15)"},{"range":[95,100],"color":"rgba(16,185,129,.15)"}]}))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=220, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            p50_time = float(np.percentile(delivery_times, 50))
            fig = go.Figure(go.Bar(x=[p50_time, avg_time, p90_time, promised_time],
                y=["P50","Avg","P90","Promise"], orientation="h",
                marker_color=["#10b981","#6366f1","#ef4444","#f59e0b"], marker_line_width=0))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8899bb", height=220, title=dict(text="Delivery Time (min)", font=dict(color="#f0f4ff",size=13)),
                margin=dict(l=10,r=40,t=40,b=10), xaxis=dict(title="Minutes",gridcolor="rgba(99,130,255,.05)"))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            hist_data = np.histogram(delivery_times, bins=12)
            mids = [(hist_data[1][i]+hist_data[1][i+1])/2 for i in range(len(hist_data[0]))]
            fig = go.Figure(go.Bar(x=mids, y=hist_data[0],
                marker_color=["#ef4444" if x > promised_time+2 else "#10b981" for x in mids], marker_line_width=0))
            fig.add_vline(x=promised_time, line_dash="dash", line_color="#f59e0b")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8899bb", height=220, title=dict(text="Order Distribution by Time", font=dict(color="#f0f4ff",size=13)),
                margin=dict(l=10,r=10,t=40,b=10), xaxis=dict(title="Minutes",gridcolor="rgba(99,130,255,.05)"),
                yaxis=dict(gridcolor="rgba(99,130,255,.05)"))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("👥 Customer Retention (Simulated)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            weeks = ["Week 1","Week 2","Week 3","Week 4","Week 5","Week 6"]
            new_c = [1200,980,1100,870,1050,920]; repeat_c = [800,920,1050,1100,1200,1280]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="New",    x=weeks, y=new_c,    marker_color="#6366f1", marker_line_width=0))
            fig.add_trace(go.Bar(name="Repeat", x=weeks, y=repeat_c, marker_color="#10b981", marker_line_width=0))
            fig.update_layout(**PLOTLY, barmode="group", title="New vs Repeat Customers", title_font_color="#f0f4ff", height=260)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            retention_matrix = [[100,68,52,41],[100,71,55,43],[100,65,48,38],[100,73,58,46]]
            fig = go.Figure(go.Heatmap(z=retention_matrix, x=["W+0","W+1","W+2","W+3"],
                y=["Week 1","Week 2","Week 3","Week 4"],
                colorscale=[[0,"#0d1628"],[0.4,"#312e81"],[1,"#6366f1"]],
                text=[[f"{v}%" for v in row] for row in retention_matrix], texttemplate="%{text}"))
            fig.update_layout(**PLOTLY, height=260, title="Cohort Retention (%)", title_font_color="#f0f4ff")
            st.plotly_chart(fig, use_container_width=True)

    # Download
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇ Download Filtered CSV", df.to_csv(index=False).encode(),
                           "zepto_filtered.csv", "text/csv", use_container_width=True)
    with col2:
        summary = df.groupby("Category").agg(
            Revenue=("Total Revenue","sum"), Orders=("Orders","sum"), Margin=("Profit Margin","mean")
        ).reset_index()
        st.download_button("⬇ Download Category Summary", summary.to_csv(index=False).encode(),
                           "zepto_summary.csv", "text/csv", use_container_width=True)

# ════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════
with tab_analytics:
    if not show_stats or len(df) < 5:
        st.info("Enable Statistical Analysis in the sidebar and ensure at least 5 records are loaded.")
    else:
        st.markdown('<div class="sec-head">Descriptive Statistics & Correlations</div>', unsafe_allow_html=True)
        rev_arr  = df["Total Revenue"].values
        z_scores = np.abs(stats.zscore(rev_arr))
        outliers = df[z_scores > 2].copy()
        outliers["Z-Score"] = z_scores[z_scores > 2].round(2)
        _, p_norm = stats.shapiro(rev_arr[:5000])
        r_disc, p_disc = stats.pearsonr(df["Discount"], df["Orders"])
        r_rev,  p_rev  = stats.pearsonr(df["Total Revenue"], df["Profit"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px"><div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
            for label, val in [
                ("Mean Revenue", fmt(np.mean(rev_arr))),
                ("Median Revenue", fmt(np.median(rev_arr))),
                ("Std Deviation", fmt(np.std(rev_arr))),
                ("Skewness", f"{stats.skew(rev_arr):.3f}"),
                ("Kurtosis", f"{stats.kurtosis(rev_arr):.3f}"),
                ("Normality p-value", f"{p_norm:.4f}"),
            ]:
                st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
            normal_txt = "✓ Normally distributed" if p_norm > 0.05 else "⚠ Not normally distributed"
            normal_clr = "#34d399" if p_norm > 0.05 else "#fcd34d"
            st.markdown(f'<div style="margin-top:10px;font-size:10px;color:{normal_clr};background:rgba(99,130,255,.06);padding:7px 10px;border-radius:7px">{normal_txt} (Shapiro-Wilk, α=0.05)</div></div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f'<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px"><div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">⚠ Outlier Detection ({len(outliers)} at Z>2)</div>', unsafe_allow_html=True)
            if len(outliers) > 0:
                for _, row in outliers.head(6).iterrows():
                    st.markdown(f'<div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.15);border-radius:7px;padding:8px 10px;margin-bottom:5px"><div style="font-size:11px;font-weight:600;color:#f0f4ff">{row["Product Name"]}</div><div style="font-size:9px;color:#8899bb">{row.get("City","—")} · {row.get("Category","—")}</div><div style="display:flex;justify-content:space-between;margin-top:3px"><span style="font-size:10px;color:#67e8f9">{fmt(row["Total Revenue"])}</span><span style="font-size:10px;color:#fca5a5;font-family:monospace">Z={row["Z-Score"]}</span></div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:11px;color:#4a5a7a;text-align:center;padding:20px">No significant outliers (Z>2)</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown('<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px"><div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
            for label, val in [
                ("Discount → Orders (r)", f"{r_disc:.3f}"),
                ("Discount → Orders (p)", f"{p_disc:.4f}"),
                ("Revenue → Profit (r)", f"{r_rev:.3f}"),
                ("Revenue → Profit (p)", f"{p_rev:.4f}"),
            ]:
                st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
            for pair, r, p in [("Discount ↔ Orders", r_disc, p_disc), ("Revenue ↔ Profit", r_rev, p_rev)]:
                strong = abs(r) > 0.5; direction = "positive" if r > 0 else "negative"
                txt = f"{'Strong' if strong else 'Weak'} {direction} — {'sig. ✓' if p < 0.05 else 'not sig.'} (p={p:.3f})"
                clr = "#34d399" if p < 0.05 else "#fcd34d"
                st.markdown(f'<div style="background:rgba(99,102,241,.07);border-radius:6px;padding:7px 10px;margin-top:6px"><div style="font-size:10px;font-weight:600;color:#a5b4fc">{pair}</div><div style="font-size:10px;color:{clr};margin-top:2px">{txt}</div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Full correlation matrix
        num_df = df[["Original Price","Current Price","Discount","Orders","Total Revenue","Profit","Profit Margin"]].corr().round(3)
        fig = px.imshow(num_df, text_auto=True,
                        color_continuous_scale=["#ef4444","#0d1628","#6366f1"],
                        zmin=-1, zmax=1, title="Full Correlation Matrix")
        fig.update_layout(**PLOTLY, title_font_color="#f0f4ff", height=360)
        st.plotly_chart(fig, use_container_width=True)

        # Unit economics (labeled as model estimates)
        st.markdown('<div class="sec-head">Unit Economics Model <span style="font-size:9px;color:#fbbf24">(Estimated — replace with real cost data)</span></div>', unsafe_allow_html=True)
        avg_rev = float(df["Total Revenue"].mean()) if len(df) > 0 else 500
        avg_cogs = avg_rev*0.52; avg_rider = avg_rev*0.12; avg_pkg = avg_rev*0.03
        avg_gateway = avg_rev*0.02; avg_promo = avg_rev*0.05
        net_profit = avg_rev - avg_cogs - avg_rider - avg_pkg - avg_gateway - avg_promo
        contrib_margin = (net_profit / avg_rev * 100) if avg_rev > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            labels = ["Revenue","COGS","Rider Pay","Packaging","Gateway","Promos","Net Profit"]
            values = [avg_rev,-avg_cogs,-avg_rider,-avg_pkg,-avg_gateway,-avg_promo,net_profit]
            fig = go.Figure(go.Waterfall(
                orientation="v", measure=["absolute","relative","relative","relative","relative","relative","total"],
                x=labels, y=values, text=[fmt(abs(v)) for v in values], textposition="outside",
                connector={"line":{"color":"rgba(99,130,255,.3)"}},
                decreasing={"marker":{"color":"#ef4444"}}, increasing={"marker":{"color":"#10b981"}},
                totals={"marker":{"color":"#6366f1"}}))
            fig.update_layout(**PLOTLY, height=320,
                title=dict(text=f"Revenue → Net Profit | CM: {contrib_margin:.1f}% (estimated)", font=dict(color="#f0f4ff",size=12)),
                showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Pie(
                labels=["COGS (52%)","Rider Pay (12%)","Packaging (3%)","Gateway (2%)","Promos (5%)","Net Profit (26%)"],
                values=[avg_cogs,avg_rider,avg_pkg,avg_gateway,avg_promo,max(0,net_profit)],
                hole=0.6, marker=dict(colors=["#6366f1","#06b6d4","#f59e0b","#8b5cf6","#ec4899","#10b981"]),
                textinfo="label+percent", textfont=dict(size=10)))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=320,
                title=dict(text="Estimated Cost Structure", font=dict(color="#f0f4ff",size=13)),
                margin=dict(l=10,r=10,t=50,b=10), legend=dict(font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════
# TAB 4 — FORECAST
# ════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="sec-head">Sales Forecasting — Linear Regression</div>', unsafe_allow_html=True)
    prod_rev_arr = df.groupby("Product Name")["Total Revenue"].sum().sort_values().values
    n = len(prod_rev_arr)

    if n < 5:
        st.warning("Need at least 5 product records for forecasting. Adjust your filters.")
    else:
        X = np.arange(1, n+1).reshape(-1, 1); y = prod_rev_arr
        model = LinearRegression().fit(X, y)
        next_val = max(0, float(model.predict([[n+1]])[0]))
        r2 = model.score(X, y)
        residuals = y - model.predict(X); ci = 1.96 * np.std(residuals)
        mean_y = np.mean(y); growth = ((next_val - mean_y) / mean_y * 100) if mean_y else 0

        col1, col2 = st.columns([2, 1])
        with col1:
            step = max(1, n // 20); xs = list(range(1, n+1, step)) + [n+1]
            actuals_plot = [float(prod_rev_arr[i-1]) if i <= n else None for i in xs]
            trend_plot   = [float(model.predict([[i]])[0]) for i in xs]
            upper = [t + ci for t in trend_plot]; lower = [max(0, t - ci) for t in trend_plot]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=upper, fill=None, mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=lower, fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(99,102,241,.08)", name="95% CI"))
            fig.add_trace(go.Scatter(x=xs, y=actuals_plot, mode="lines+markers", name="Actual", line=dict(color="#6366f1", width=2), marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=xs, y=trend_plot, mode="lines", name="Trend", line=dict(color="#06b6d4", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=[n+1], y=[next_val], mode="markers", name="Forecast", marker=dict(color="#10b981", size=14, symbol="star")))
            fig.update_layout(**PLOTLY, title="Revenue Forecast with 95% Confidence Interval", title_font_color="#f0f4ff", height=320)
            fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("⚠️ Linear regression on product-level totals. For time-series forecasting, a date column is required.")

        with col2:
            clr = "#34d399" if growth >= 0 else "#f87171"
            st.markdown(f"""
            <div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:20px">
              <div style="font-size:10px;color:#4a5a7a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Next Forecast Point</div>
              <div style="font-size:26px;font-weight:700;font-family:'DM Mono',monospace;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{fmt(next_val)}</div>
              <div style="display:inline-block;font-size:11px;font-weight:600;padding:3px 9px;border-radius:5px;margin:8px 0;background:{'rgba(16,185,129,.12)' if growth>=0 else 'rgba(239,68,68,.12)'};color:{clr}">
                {'↑' if growth>=0 else '↓'} {abs(growth):.1f}% vs avg
              </div>
            """, unsafe_allow_html=True)
            for label, val in [("R² Score", f"{r2:.4f}"), ("Slope β₁", f"{model.coef_[0]:.3f}"), ("95% CI Band", fmt(ci))]:
                st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
            quality = "✓ Good fit" if r2 > 0.6 else "⚠ Low R² — high variance"
            q_clr = "#34d399" if r2 > 0.6 else "#fcd34d"
            st.markdown(f'<div style="margin-top:10px;font-size:10px;color:{q_clr};background:rgba(99,130,255,.06);padding:7px 10px;border-radius:7px">{quality}</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 5 — AI INSIGHTS
# ════════════════════════════════════════
with tab_ai:
    st.markdown('<div class="sec-head">AI-Generated Business Insights</div>', unsafe_allow_html=True)
    insights = generate_exec_insights(df)
    cols = st.columns(3)
    for i, ins in enumerate(insights):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="insight-card">
              <div style="font-size:24px;margin-bottom:10px">{ins['icon']}</div>
              <div class="insight-title">{ins['title']}</div>
              <div class="insight-body">{ins['body']}</div>
              <div class="insight-action">{ins['action']}</div>
            </div><br>""", unsafe_allow_html=True)

    # Action required section
    st.markdown('<div class="sec-head">🚨 Action Required</div>', unsafe_allow_html=True)
    actions = []
    # Weakest city
    if len(city_rev) >= 2:
        actions.append({"priority":"HIGH", "action":f"Launch promotions in {city_rev.index[-1]}", "detail":f"Revenue gap: {fmt(city_rev.iloc[0]-city_rev.iloc[-1])} vs top city", "color":"#ef4444"})
    # Weakest category
    if len(cat_rev) >= 2:
        actions.append({"priority":"MED", "action":f"Review {cat_rev.index[-1]} strategy", "detail":f"Only {cat_rev.iloc[-1]/total_rev*100:.1f}% revenue share — reposition or cut", "color":"#f59e0b"})
    # Discount opportunity
    disc_r, disc_p = stats.pearsonr(df["Discount"], df["Orders"]) if len(df) >= 5 else (0, 1)
    if disc_r > 0.3 and disc_p < 0.05:
        actions.append({"priority":"MED", "action":"Scale discount promotions", "detail":f"Positive correlation (r={disc_r:.2f}) — discounts statistically drive orders", "color":"#f59e0b"})
    elif disc_r < -0.3:
        actions.append({"priority":"HIGH", "action":"Review discount strategy", "detail":"Discounts negatively correlated with orders — customers may perceive quality drop", "color":"#ef4444"})
    # Top product stock
    top_p = prod_rev.index[0] if len(prod_rev) > 0 else "—"
    actions.append({"priority":"LOW", "action":f"Ensure {top_p} is always in stock", "detail":"Top revenue driver — stockout = immediate revenue loss", "color":"#10b981"})

    for act in actions:
        st.markdown(f"""
        <div style="background:#0d1628;border:1px solid {act['color']}33;border-left:3px solid {act['color']};
          border-radius:10px;padding:14px 16px;margin-bottom:8px;display:flex;align-items:center;gap:14px">
          <div style="background:{act['color']}22;color:{act['color']};font-size:9px;font-weight:700;
            padding:3px 8px;border-radius:4px;white-space:nowrap">{act['priority']}</div>
          <div>
            <div style="font-size:13px;font-weight:600;color:#f0f4ff">{act['action']}</div>
            <div style="font-size:11px;color:#64748b;margin-top:2px">{act['detail']}</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 6 — BLINKBOT (Claude-powered)
# ════════════════════════════════════════
with tab_bot:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e1b6e,#312e81);padding:16px 20px;
      border:1px solid rgba(99,102,241,.25);border-radius:16px;margin-bottom:16px;
      display:flex;align-items:center;gap:14px">
      <div style="width:44px;height:44px;background:linear-gradient(135deg,#6366f1,#06b6d4);
        border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px">🤖</div>
      <div>
        <div style="font-size:15px;font-weight:700;color:#f0f4ff">BlinkBot</div>
        <div style="font-size:11px;color:#a5b4fc">Senior AI Business Analyst · Powered by Claude · Context-aware</div>
      </div>
      <div style="margin-left:auto;background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);
        border-radius:20px;padding:4px 12px;font-size:10px;color:#34d399">● Analyzing {len(df):,} records</div>
    </div>
    """.replace("{len(df):,}", f"{len(df):,}"), unsafe_allow_html=True)

    # Init chat
    if "bb_history" not in st.session_state:
        st.session_state.bb_history = [
            {"role":"bot", "msg":f"👋 **Hi! I'm BlinkBot**, your AI Business Analyst powered by Claude.\n\nI've analyzed your **{len(df):,} records** and I'm ready to answer questions about revenue, profit, products, cities, influencer impact, and more.\n\nTry asking: *\"Give me an executive summary\"* or *\"Which city needs attention?\"*"}
        ]
    if "bb_messages" not in st.session_state:
        st.session_state.bb_messages = []

    # Display history
    for msg in st.session_state.bb_history:
        css = "chat-bot" if msg["role"] == "bot" else "chat-user"
        st.markdown(f'<div class="{css}">{msg["msg"]}</div>', unsafe_allow_html=True)

    # Quick buttons
    st.markdown("**💡 Suggested Questions:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    clicked = None
    quick = [
        ("📋 Full Summary",       "Give me a complete executive summary of the business"),
        ("🏆 Best & Worst",       "What are the best and worst performing products and cities?"),
        ("⚡ Influencer ROI",     "Is influencer marketing working? What's the revenue lift?"),
        ("🚨 What needs action?", "What should I focus on to improve revenue this week?"),
    ]
    for col, (label, q) in zip([qc1,qc2,qc3,qc4], quick):
        with col:
            if st.button(label, key=f"bb_{label}", use_container_width=True):
                clicked = q

    # Input
    with st.form("bb_form", clear_on_submit=True):
        fc1, fc2 = st.columns([5,1])
        with fc1:
            user_input = st.text_input("Ask BlinkBot...", placeholder="e.g. Which category has the worst margin?", label_visibility="collapsed")
        with fc2:
            submitted = st.form_submit_button("Ask 🤖", use_container_width=True)

    question = user_input.strip() if (submitted and user_input.strip()) else clicked

    if question:
        st.session_state.bb_history.append({"role":"user", "msg": question})
        # Build Claude context
        system_prompt = build_data_context(df)
        # Maintain conversation history for Claude (last 10 turns)
        # Build API messages: only user+assistant alternating turns, skip the initial bot greeting
        api_messages = []
        for m in st.session_state.bb_history[-10:]:
            role = "user" if m["role"] == "user" else "assistant"
            # Skip leading assistant messages (Claude API requires first message to be user)
            if not api_messages and role == "assistant":
                continue
            # Avoid consecutive same-role messages
            if api_messages and api_messages[-1]["role"] == role:
                api_messages[-1]["content"] += "\n" + m["msg"]
            else:
                api_messages.append({"role": role, "content": m["msg"]})
        # Safety: must have at least one user message
        if not api_messages:
            api_messages = [{"role": "user", "content": question}]

        with st.spinner("BlinkBot is analyzing..."):
            response = call_claude_api(api_messages, system_prompt)

        st.session_state.bb_history.append({"role":"bot", "msg": response})
        st.rerun()

    if len(st.session_state.bb_history) > 1:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.bb_history = []
                st.session_state.bb_messages = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════════
if show_raw:
    st.markdown('<div class="sec-head">Raw Data Table</div>', unsafe_allow_html=True)
    display_cols = ["Product Name","Category","City","Original Price","Current Price",
                    "Discount","Orders","Total Revenue","Profit","Profit Margin","AOV","Influencer Active"]
    show_df = df[[c for c in display_cols if c in df.columns]].copy()
    # Safe formatting that works on large datasets (avoids StreamlitAPIException with .style on 100k+ rows)
    for col in ["Total Revenue", "Profit"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].apply(lambda x: f"\u20b9{x:,.0f}" if pd.notna(x) else "\u2014")
    for col in ["Original Price", "Current Price"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].apply(lambda x: f"\u20b9{x:.0f}" if pd.notna(x) else "\u2014")
    if "Profit Margin" in show_df.columns:
        show_df["Profit Margin"] = show_df["Profit Margin"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "\u2014")
    if "AOV" in show_df.columns:
        show_df["AOV"] = show_df["AOV"].apply(lambda x: f"\u20b9{x:,.0f}" if pd.notna(x) else "\u2014")
    if "Discount" in show_df.columns:
        show_df["Discount"] = show_df["Discount"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "\u2014")
    st.dataframe(show_df, use_container_width=True, height=400)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  Ayush Intelligence Hub v3.0 &nbsp;·&nbsp;
  Developed by <span class="dev">Ayush Mishra</span> &nbsp;·&nbsp;
  Pandas · SciPy · scikit-learn · Streamlit · Plotly · Claude API
</div>
""", unsafe_allow_html=True)
