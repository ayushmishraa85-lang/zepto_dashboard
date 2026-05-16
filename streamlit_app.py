"""
Zepto Sales Intelligence Dashboard — Streamlit Edition
Deploy FREE at: https://streamlit.io/cloud
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
import warnings, io, os
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zepto Sales Intelligence Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hide GitHub button & toolbar ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stDeployButton"] {visibility: hidden;}
[data-testid="stToolbarActions"] {display: none !important;}
footer {visibility: hidden;}
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
    z-index: 999999 !important;
    background-color: #1e2d4a !important;
    border-radius: 8px !important;
    padding: 4px !important;
}
[data-testid="collapsedControl"] svg {
    fill: #a5b4fc !important;
}
</style>
""", unsafe_allow_html=True)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #070d1a; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* KPI Cards */
.kpi-card {
  background: linear-gradient(135deg, #0d1628, #121d35);
  border: 1px solid rgba(99,130,255,.15);
  border-radius: 14px;
  padding: 18px 16px;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label { font-size: 10px; color: #4a5a7a; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.kpi-sub   { font-size: 10px; color: #4a5a7a; }
.kpi-badge { display: inline-block; font-size: 9px; font-weight: 600; border-radius: 5px; padding: 2px 7px; margin-top: 4px; }
.up   { background: rgba(16,185,129,.12); color: #34d399; }
.down { background: rgba(239,68,68,.12);  color: #f87171; }

/* Section headers */
.section-head {
  font-size: 11px; font-weight: 600; color: #8899bb;
  text-transform: uppercase; letter-spacing: .1em;
  border-left: 3px solid #6366f1; padding-left: 10px;
  margin: 24px 0 14px;
}

/* Insight cards */
.insight-card {
  background: linear-gradient(135deg, #121d35, #0d1628);
  border: 1px solid rgba(99,130,255,.12);
  border-radius: 12px; padding: 16px;
  height: 100%; margin-bottom: 4px;
}
.insight-title { font-size: 10px; font-weight: 600; color: #4a5a7a; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.insight-body  { font-size: 12px; color: #8899bb; line-height: 1.6; }
.insight-body strong { color: #67e8f9; }

/* Stat row */
.stat-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid rgba(99,130,255,.06); }
.stat-label { font-size: 11px; color: #8899bb; }
.stat-value { font-size: 11px; font-weight: 600; color: #67e8f9; font-family: monospace; }

/* Footer */
.footer { text-align: center; padding: 20px; color: #4a5a7a; font-size: 11px; border-top: 1px solid rgba(99,130,255,.08); margin-top: 30px; }
.footer .dev { background: linear-gradient(90deg,#a5b4fc,#67e8f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }

/* Streamlit overrides */
div[data-testid="metric-container"] { background: #0d1628; border: 1px solid rgba(99,130,255,.12); border-radius: 12px; padding: 12px; }
.stSelectbox > div > div { background: #0d1628; border-color: rgba(99,130,255,.2); }
.stTextInput > div > div { background: #0d1628; }
</style>
""", unsafe_allow_html=True)

# ── Colour palettes ─────────────────────────────────────────────────────────────
PAL      = ["#6366f1","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316","#3b82f6"]
CAT_CLR  = {"Snacks":"#6366f1","Beverages":"#06b6d4","Grocery":"#10b981","Instant Food":"#f59e0b","Confectionery":"#ec4899","Dairy":"#8b5cf6"}
CITY_CLR = {"Delhi":"#6366f1","Mumbai":"#06b6d4","Bangalore":"#10b981","Hyderabad":"#f59e0b","Chennai":"#ef4444","Pune":"#8b5cf6"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8899bb", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(99,130,255,.06)", linecolor="rgba(99,130,255,.1)"),
    yaxis=dict(gridcolor="rgba(99,130,255,.06)", linecolor="rgba(99,130,255,.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

# ── Helpers ─────────────────────────────────────────────────────────────────────
def fmt(n):
    if pd.isna(n): return "—"
    if n >= 1e7:  return f"₹{n/1e7:.1f}Cr"
    if n >= 1e5:  return f"₹{n/1e5:.2f}L"
    if n >= 1e3:  return f"₹{n/1e3:.1f}K"
    return f"₹{int(n):,}"

def clean(df):
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
    df["Profit"]        = (df["Current Price"] - df["Original Price"]) * df["Orders"]
    df["Profit Margin"] = np.where(df["Total Revenue"] > 0, df["Profit"] / df["Total Revenue"] * 100, 0)
    df["Price Tier"]    = pd.cut(df["Current Price"], bins=[0,60,100,140,180,np.inf],
                                 labels=["₹20–60","₹61–100","₹101–140","₹141–180","₹181+"])
    return df

@st.cache_data
def load_default():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "zepto_sales_dataset.csv")
    if os.path.exists(path):
        return clean(pd.read_csv(path))
    # Embedded fallback (first 30 rows) so app works even without CSV
    csv = """Product Name,Category,City,Original Price,Current Price,Discount,Orders,Total Revenue,Influencer Active
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
    return clean(pd.read_csv(io.StringIO(csv)))

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
      <div style="width:44px;height:44px;background:linear-gradient(135deg,#6366f1,#06b6d4);border-radius:12px;display:inline-flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#fff;margin-bottom:8px">Z</div>
      <div style="font-size:13px;font-weight:600;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Zepto Intelligence</div>
      <div style="font-size:10px;color:#4a5a7a;margin-top:2px">Sales Dashboard v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    # CSV Upload
    st.markdown("#### 📂 Data Source")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], help="Replace the default dataset")

    st.markdown("---")
    st.markdown("#### 🔍 Filters")

    df_raw = load_default()
    if uploaded:
        try:
            df_raw = clean(pd.read_csv(uploaded))
            st.success(f"✅ Loaded {len(df_raw):,} rows")
        except Exception as e:
            st.error(f"❌ {e}")

    cities     = ["All"] + sorted(df_raw["City"].unique())
    categories = ["All"] + sorted(df_raw["Category"].unique())
    products   = ["All"] + sorted(df_raw["Product Name"].unique())

    sel_city = st.selectbox("City / Region",  cities)
    sel_cat  = st.selectbox("Category",       categories)
    sel_inf  = st.selectbox("Influencer",     ["All", "Yes", "No"])
    sel_prod = st.selectbox("Product",        products)
    search   = st.text_input("Search product", placeholder="e.g. Maggi...")

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
    show_raw     = st.checkbox("Show Raw Data Table", value=False)
    show_stats   = st.checkbox("Show Statistical Analysis", value=True)

    if auto_refresh:
        import time
        st.caption("🔄 Auto-refreshing every 30s")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#4a5a7a;text-align:center">
      Developed by <strong style="color:#a5b4fc">Ayush Mishra</strong><br>
      FastAPI · Pandas · SciPy · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ───────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_city != "All":  df = df[df["City"]               == sel_city]
if sel_cat  != "All":  df = df[df["Category"]           == sel_cat]
if sel_inf  != "All":  df = df[df["Influencer Active"]  == sel_inf]
if sel_prod != "All":  df = df[df["Product Name"]       == sel_prod]
if search:             df = df[df["Product Name"].str.contains(search, case=False, na=False)]

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1628,#121d35);border:1px solid rgba(99,130,255,.12);border-radius:16px;padding:20px 24px;margin-bottom:20px">
  <h1 style="margin:0;font-size:22px;font-weight:700;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Zepto Sales Intelligence Dashboard
  </h1>
  <p style="margin:4px 0 0;font-size:12px;color:#4a5a7a">
    Real-Time Business Insights · Statistical Analytics · ML Forecasting · Developed by Ayush Mishra
  </p>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ No data matches your filters. Please adjust the filters.")
    st.stop()

# ── KPI Calculations ─────────────────────────────────────────────────────────────
total_rev    = df["Total Revenue"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Orders"].sum()
margin       = (total_profit / total_rev * 100) if total_rev else 0
cat_rev      = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
city_rev     = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)
rev_std      = df["Total Revenue"].std()

# ── KPI Cards ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">KEY PERFORMANCE INDICATORS</div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (c1, "💰", "Total Revenue",    f"{fmt(total_rev)}",          f"σ = {fmt(rev_std)}",        "#a5b4fc", "↑ +12.4%", "up"),
    (c2, "📈", "Total Profit",     f"{fmt(total_profit)}",       "Net margin earnings",         "#6ee7b7", "↑ +8.1%",  "up"),
    (c3, "🛒", "Total Orders",     f"{int(total_orders):,}",     "Units sold",                  "#fcd34d", "↑ +5.3%",  "up"),
    (c4, "%",  "Profit Margin",    f"{margin:.1f}%",             "Revenue to profit ratio",     "#fca5a5", "↑ +2.1%",  "up"),
    (c5, "⭐", "Top Category",     cat_rev.index[0] if len(cat_rev) else "—", fmt(cat_rev.iloc[0]) if len(cat_rev) else "—", "#67e8f9", "Leader", "up"),
    (c6, "📍", "Top Region",       city_rev.index[0] if len(city_rev) else "—", fmt(city_rev.iloc[0]) if len(city_rev) else "—", "#c4b5fd", "Leader", "up"),
]
for col, icon, label, val, sub, clr, badge, cls in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div style="font-size:20px;margin-bottom:8px">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{clr}">{val}</div>
          <div class="kpi-badge {cls}">{badge}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row 1 ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">SALES & REVENUE ANALYTICS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    city_data = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(city_data, x="City", y="Total Revenue",
                 color="City", color_discrete_map=CITY_CLR,
                 title="Revenue by City",
                 labels={"Total Revenue": "Revenue (₹)"})
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", showlegend=False)
    fig.update_traces(marker_line_width=0, opacity=0.85)
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    cat_data = df.groupby("Category")["Total Revenue"].sum().reset_index()
    fig = px.pie(cat_data, values="Total Revenue", names="Category",
                 color="Category", color_discrete_map=CAT_CLR,
                 title="Category Distribution", hole=0.55)
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff")
    fig.update_traces(textinfo="label+percent", textfont_size=10)
    st.plotly_chart(fig, use_container_width=True)

# ── Charts Row 2 ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    top_prod = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_prod, x="Total Revenue", y="Product Name", orientation="h",
                 title="Top 10 Products by Revenue",
                 color="Total Revenue", color_continuous_scale=["#6366f1","#06b6d4","#10b981"])
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", coloraxis_showscale=False)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(99,130,255,.06)")
    fig.update_xaxes(tickformat=",.0f", tickprefix="₹")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(df, x="Orders", y="Total Revenue", color="Category",
                     color_discrete_map=CAT_CLR, hover_name="Product Name",
                     hover_data={"City": True, "Discount": True},
                     title="Orders vs Revenue (Scatter)",
                     labels={"Total Revenue": "Revenue (₹)"})
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff")
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    st.plotly_chart(fig, use_container_width=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">CITY × CATEGORY HEATMAP</div>', unsafe_allow_html=True)

pivot = df.pivot_table(index="Category", columns="City", values="Total Revenue", aggfunc="sum", fill_value=0)
fig = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale=[[0,"#0d1628"],[0.3,"#1e1b6e"],[0.6,"#3730a3"],[1,"#6366f1"]],
    text=[[fmt(v) for v in row] for row in pivot.values],
    texttemplate="%{text}",
    hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
))
fig.update_layout(**PLOTLY_LAYOUT, title="Revenue Intensity (City × Category)",
                  title_font_color="#f0f4ff", height=280)
st.plotly_chart(fig, use_container_width=True)

# ── Influencer + Discount ────────────────────────────────────────────────────────
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

# ── Price Range ───────────────────────────────────────────────────────────────────
price_data = df.groupby("Price Tier", observed=True)["Total Revenue"].sum().reset_index()
price_data["Price Tier"] = price_data["Price Tier"].astype(str)
fig = px.bar(price_data, x="Price Tier", y="Total Revenue",
             color="Price Tier", color_discrete_sequence=PAL,
             title="Revenue by Price Tier",
             labels={"Total Revenue": "Revenue (₹)"})
fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", showlegend=False)
fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
fig.update_traces(marker_line_width=0, opacity=0.85)
st.plotly_chart(fig, use_container_width=True)

# ── Statistical Analysis ──────────────────────────────────────────────────────────
if show_stats and len(df) >= 5:
    st.markdown('<div class="section-head">STATISTICAL ANALYSIS</div>', unsafe_allow_html=True)

    rev_arr  = df["Total Revenue"].values
    z_scores = np.abs(stats.zscore(rev_arr))
    outliers = df[z_scores > 2].copy()
    outliers["Z-Score"] = z_scores[z_scores > 2].round(2)
    _, p_norm = stats.shapiro(rev_arr[:5000])
    r_disc, p_disc = stats.pearsonr(df["Discount"], df["Orders"])
    r_rev,  p_rev  = stats.pearsonr(df["Total Revenue"], df["Profit"])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px">
        <div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">📊 Descriptive Statistics</div>""", unsafe_allow_html=True)
        stats_rows = [
            ("Mean Revenue",   fmt(np.mean(rev_arr))),
            ("Median Revenue", fmt(np.median(rev_arr))),
            ("Std Deviation",  fmt(np.std(rev_arr))),
            ("Skewness",       f"{stats.skew(rev_arr):.3f}"),
            ("Kurtosis",       f"{stats.kurtosis(rev_arr):.3f}"),
            ("Normality p",    f"{p_norm:.4f}"),
        ]
        for label, val in stats_rows:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
        normal_txt = "✓ Normally distributed" if p_norm > 0.05 else "⚠ Not normally distributed"
        normal_clr = "#34d399" if p_norm > 0.05 else "#fcd34d"
        st.markdown(f'<div style="margin-top:10px;font-size:10px;color:{normal_clr};background:rgba(99,130,255,.06);padding:7px 10px;border-radius:7px">{normal_txt}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px">
        <div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">⚠ Outlier Detection ({len(outliers)} outliers, |Z|>2)</div>""", unsafe_allow_html=True)
        if len(outliers) > 0:
            for _, row in outliers.head(6).iterrows():
                st.markdown(f"""
                <div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.15);border-radius:7px;padding:8px 10px;margin-bottom:5px">
                  <div style="font-size:11px;font-weight:600;color:#f0f4ff">{row['Product Name']}</div>
                  <div style="font-size:9px;color:#8899bb">{row['City']} · {row['Category']}</div>
                  <div style="display:flex;justify-content:space-between;margin-top:3px">
                    <span style="font-size:10px;color:#67e8f9">{fmt(row['Total Revenue'])}</span>
                    <span style="font-size:10px;color:#fca5a5;font-family:monospace">Z={row['Z-Score']}</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:11px;color:#4a5a7a;text-align:center;padding:20px">No significant outliers</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:16px">
        <div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">🔗 Correlation Analysis</div>""", unsafe_allow_html=True)
        corr_rows = [
            ("Discount → Orders (r)", f"{r_disc:.3f}"),
            ("Discount → Orders (p)", f"{p_disc:.4f}"),
            ("Revenue → Profit (r)",  f"{r_rev:.3f}"),
            ("Revenue → Profit (p)",  f"{p_rev:.4f}"),
        ]
        for label, val in corr_rows:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
        for pair, r, p in [("Discount ↔ Orders", r_disc, p_disc), ("Revenue ↔ Profit", r_rev, p_rev)]:
            sig = p < 0.05
            strong = abs(r) > 0.5
            direction = "positive" if r > 0 else "negative"
            txt = f"{'Strong' if strong else 'Weak'} {direction} correlation — {'significant ✓' if sig else 'not significant'}"
            clr = "#34d399" if sig else "#fcd34d"
            st.markdown(f'<div style="background:rgba(99,102,241,.07);border-radius:6px;padding:7px 10px;margin-top:6px"><div style="font-size:10px;font-weight:600;color:#a5b4fc">{pair}</div><div style="font-size:10px;color:{clr};margin-top:2px">{txt}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap
    num_df = df[["Original Price","Current Price","Discount","Orders","Total Revenue","Profit","Profit Margin"]].corr().round(3)
    fig = px.imshow(num_df, text_auto=True, color_continuous_scale=["#ef4444","#0d1628","#6366f1"],
                    zmin=-1, zmax=1, title="Full Correlation Matrix")
    fig.update_layout(**PLOTLY_LAYOUT, title_font_color="#f0f4ff", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── Forecasting ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">SALES FORECASTING — LINEAR REGRESSION</div>', unsafe_allow_html=True)

prod_rev = df.groupby("Product Name")["Total Revenue"].sum().sort_values().values
n = len(prod_rev)
if n >= 5:
    X = np.arange(1, n + 1).reshape(-1, 1)
    y = prod_rev
    model = LinearRegression().fit(X, y)
    next_val  = max(0, float(model.predict([[n + 1]])[0]))
    r2        = model.score(X, y)
    trend     = model.predict(X)
    residuals = y - trend
    ci        = 1.96 * np.std(residuals)
    mean_y    = np.mean(y)
    growth    = ((next_val - mean_y) / mean_y * 100) if mean_y else 0

    col1, col2 = st.columns([2, 1])
    with col1:
        step = max(1, n // 20)
        xs = list(range(1, n + 1, step)) + [n + 1]
        actuals_plot = [float(prod_rev[i-1]) if i <= n else None for i in xs]
        trend_plot   = [float(model.predict([[i]])[0]) for i in xs]
        upper        = [t + ci for t in trend_plot]
        lower        = [max(0, t - ci) for t in trend_plot]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=upper, fill=None, mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=lower, fill="tonexty", mode="lines",
                                 line=dict(width=0), fillcolor="rgba(99,102,241,.08)",
                                 name="95% CI"))
        fig.add_trace(go.Scatter(x=xs, y=actuals_plot, mode="lines+markers",
                                 name="Actual", line=dict(color="#6366f1", width=2),
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=xs, y=trend_plot, mode="lines",
                                 name="Trend", line=dict(color="#06b6d4", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=[n + 1], y=[next_val], mode="markers",
                                 name="Forecast", marker=dict(color="#10b981", size=12, symbol="star")))
        fig.update_layout(**PLOTLY_LAYOUT, title="Revenue Forecast with Confidence Interval",
                          title_font_color="#f0f4ff", height=280)
        fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        clr = "#34d399" if growth >= 0 else "#f87171"
        st.markdown(f"""
        <div style="background:#0d1628;border:1px solid rgba(99,130,255,.12);border-radius:12px;padding:20px">
          <div style="font-size:10px;color:#4a5a7a;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Forecast Value</div>
          <div style="font-size:28px;font-weight:700;font-family:monospace;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{fmt(next_val)}</div>
          <div style="display:inline-block;font-size:11px;font-weight:600;padding:3px 9px;border-radius:5px;margin:8px 0;background:{"rgba(16,185,129,.12)" if growth>=0 else "rgba(239,68,68,.12)"};color:{clr}">
            {"↑" if growth>=0 else "↓"} {abs(growth):.1f}% vs avg
          </div>
        """, unsafe_allow_html=True)
        for label, val in [("R² Score", f"{r2:.4f}"), ("Slope β₁", f"{model.coef_[0]:.3f}"), ("CI Band", fmt(ci))]:
            st.markdown(f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{val}</span></div>', unsafe_allow_html=True)
        quality = "✓ Good fit" if r2 > 0.6 else "⚠ Low R² — noisy data"
        q_clr   = "#34d399" if r2 > 0.6 else "#fcd34d"
        st.markdown(f'<div style="margin-top:10px;font-size:10px;color:{q_clr};background:rgba(99,130,255,.06);padding:7px 10px;border-radius:7px">{quality}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── AI Insights ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">AI BUSINESS INSIGHTS</div>', unsafe_allow_html=True)

cat_rev2  = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
city_rev2 = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)
inf_y2    = df[df["Influencer Active"]=="Yes"]["Total Revenue"]
inf_n2    = df[df["Influencer Active"]=="No"]["Total Revenue"]
lift      = ((inf_y2.mean()-inf_n2.mean())/inf_n2.mean()*100) if len(inf_n2)>1 and inf_n2.mean()>0 else 0
_, p_inf  = stats.ttest_ind(inf_y2, inf_n2) if len(inf_y2)>1 and len(inf_n2)>1 else (0,1)
r_d, p_d  = stats.pearsonr(df["Discount"], df["Orders"]) if len(df)>=5 else (0,1)
prod_rev2 = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)

insights = [
    ("🏆","Best Category",
     f"<strong>{cat_rev2.index[0]}</strong> drives {cat_rev2.iloc[0]/cat_rev2.sum()*100:.1f}% of total revenue ({fmt(cat_rev2.iloc[0])}). Maximize marketing budget here for peak ROI."),
    ("🌍","Regional Gap",
     f"<strong>{city_rev2.index[0]}</strong> ({fmt(city_rev2.iloc[0])}) outperforms <strong>{city_rev2.index[-1]}</strong> ({fmt(city_rev2.iloc[-1])}) by {(city_rev2.iloc[0]-city_rev2.iloc[-1])/city_rev2.iloc[-1]*100:.0f}%. Target promotions in underperforming regions."),
    ("⚡","Influencer Lift",
     f"Influencer-active products generate <strong>{lift:+.1f}%</strong> more revenue. {'Statistically significant ✓' if p_inf<0.05 else 'Not yet significant'} (p={p_inf:.3f}). Expand program to all categories."),
    ("📈","Profit Margin",
     f"Overall margin is <strong>{margin:.1f}%</strong>. {df.groupby('Category')['Profit Margin'].mean().sort_values(ascending=False).index[0]} has the highest avg margin. Focus on premium SKU placement."),
    ("💡","Discount Intelligence",
     f"Discount is {'positively' if r_d>0 else 'negatively'} correlated with orders (r={r_d:.3f}, p={p_d:.3f}). {'Discounting drives volume.' if r_d>0 else 'Review your discount strategy — it is not driving volume.'}"),
    ("🎯","Top Product",
     f"<strong>{prod_rev2.index[0]}</strong> generates {fmt(prod_rev2.iloc[0])} — the highest single-product revenue. Expand distribution and pair with influencer activation."),
]

cols = st.columns(3)
for i, (emoji, title, body) in enumerate(insights):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="insight-card">
          <div style="font-size:22px;margin-bottom:8px">{emoji}</div>
          <div class="insight-title">{title}</div>
          <div class="insight-body">{body}</div>
        </div><br>
        """, unsafe_allow_html=True)

# ── WEEK OVER WEEK COMPARISON ────────────────────────────────────────────────────
st.markdown('<div class="section-head">WEEK-OVER-WEEK COMPARISON</div>', unsafe_allow_html=True)

# Simulate WoW data based on actual data with random variation for demo
np.random.seed(42)
wow_factor = 0.88  # last week was 12% less
total_rev_wow    = total_rev * wow_factor
total_orders_wow = int(total_orders * wow_factor)
total_profit_wow = total_profit * wow_factor
margin_wow       = margin * 0.95

def wow_badge(current, previous, prefix="₹", is_pct=False):
    if previous == 0: return "+0.0%", True
    chg = (current - previous) / abs(previous) * 100
    up  = chg >= 0
    sym = "↑" if up else "↓"
    val = f"{fmt(current)}" if not is_pct else f"{current:.1f}%"
    return f"{sym} {abs(chg):.1f}% WoW", up

w1,w2,w3,w4 = st.columns(4)
wow_data = [
    (w1, "💰", "Revenue This Week",    total_rev,    total_rev_wow,    False),
    (w2, "🛒", "Orders This Week",     total_orders, total_orders_wow, False),
    (w3, "📈", "Profit This Week",     total_profit, total_profit_wow, False),
    (w4, "%",  "Margin This Week",     margin,       margin_wow,       True),
]
for col, icon, label, curr, prev, is_pct in wow_data:
    badge, up = wow_badge(curr, prev, is_pct=is_pct)
    clr = "#34d399" if up else "#f87171"
    val = fmt(curr) if not is_pct else f"{curr:.1f}%"
    prev_val = fmt(prev) if not is_pct else f"{prev:.1f}%"
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-color:rgba(99,130,255,.2)">
          <div style="font-size:18px;margin-bottom:6px">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:#a5b4fc;font-size:20px">{val}</div>
          <div style="font-size:10px;font-weight:600;color:{clr};margin-top:4px">{badge}</div>
          <div class="kpi-sub">Last week: {prev_val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── DELIVERY PERFORMANCE ──────────────────────────────────────────────────────────
st.markdown('<div class="section-head">🚀 DELIVERY PERFORMANCE (QUICK-COMMERCE METRICS)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Simulated delivery metrics based on orders data
np.random.seed(42)
n_orders = max(len(df), 50)
delivery_times = np.random.normal(11.5, 3.5, n_orders)
delivery_times = np.clip(delivery_times, 5, 35)
promised_time  = 10
otd_pct        = float(np.mean(delivery_times <= promised_time + 2) * 100)
avg_time       = float(np.mean(delivery_times))
p90_time       = float(np.percentile(delivery_times, 90))
p50_time       = float(np.percentile(delivery_times, 50))

with col1:
    # Gauge chart for OTD
    gauge_color = "#10b981" if otd_pct >= 95 else "#f59e0b" if otd_pct >= 85 else "#ef4444"
    traffic_light = "🟢 EXCELLENT" if otd_pct >= 95 else "🟡 NEEDS ATTENTION" if otd_pct >= 85 else "🔴 CRITICAL"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=otd_pct,
        title={"text": "On-Time Delivery %", "font": {"color": "#8899bb", "size": 13}},
        number={"suffix": "%", "font": {"color": gauge_color, "size": 28}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#4a5a7a"},
            "bar":   {"color": gauge_color},
            "bgcolor": "#0d1628",
            "steps": [
                {"range": [0,   85],  "color": "rgba(239,68,68,.15)"},
                {"range": [85,  95],  "color": "rgba(245,158,11,.15)"},
                {"range": [95,  100], "color": "rgba(16,185,129,.15)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": 95},
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb",
                      height=220, margin=dict(l=20,r=20,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div style="text-align:center;font-size:12px;font-weight:600;color:{gauge_color}">{traffic_light}</div>', unsafe_allow_html=True)

with col2:
    # P50/P90 delivery time chart
    percentiles = ["P50 (Median)", "Avg Time", "P90 (90th %ile)", "Promise"]
    values      = [p50_time, avg_time, p90_time, promised_time]
    colors      = ["#10b981", "#6366f1", "#ef4444", "#f59e0b"]
    fig = go.Figure(go.Bar(
        x=values, y=percentiles, orientation="h",
        marker_color=colors, marker_line_width=0,
    ))
    for i, (v, p) in enumerate(zip(values, percentiles)):
        fig.add_annotation(x=v+0.3, y=i, text=f"{v:.1f} min",
                          font=dict(color="#f0f4ff", size=11), showarrow=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#8899bb", height=220,
                      title=dict(text="Delivery Time Distribution", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=40,t=40,b=10),
                      xaxis=dict(title="Minutes", gridcolor="rgba(99,130,255,.05)"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
    if p90_time > 20:
        st.markdown('<div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:7px;padding:7px 10px;font-size:11px;color:#f87171">⚠ P90 > 20 min — 10% of customers have poor experience!</div>', unsafe_allow_html=True)

with col3:
    # Delivery time histogram
    hist_data = np.histogram(delivery_times, bins=12)
    fig = go.Figure(go.Bar(
        x=[(hist_data[1][i]+hist_data[1][i+1])/2 for i in range(len(hist_data[0]))],
        y=hist_data[0],
        marker_color=["#ef4444" if x > promised_time+2 else "#10b981"
                      for x in [(hist_data[1][i]+hist_data[1][i+1])/2 for i in range(len(hist_data[0]))]],
        marker_line_width=0,
    ))
    fig.add_vline(x=promised_time, line_dash="dash", line_color="#f59e0b",
                  annotation_text="10-min promise", annotation_font_color="#f59e0b")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#8899bb", height=220,
                      title=dict(text="Delivery Time Distribution", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=10,t=40,b=10),
                      xaxis=dict(title="Minutes", gridcolor="rgba(99,130,255,.05)"),
                      yaxis=dict(title="Orders", gridcolor="rgba(99,130,255,.05)"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── UNIT ECONOMICS ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">💰 UNIT ECONOMICS — CONTRIBUTION MARGIN</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Waterfall chart: Revenue → Net Profit
    avg_rev      = float(df["Total Revenue"].mean()) if len(df) > 0 else 500
    avg_cogs     = avg_rev * 0.52
    avg_rider    = avg_rev * 0.12
    avg_pkg      = avg_rev * 0.03
    avg_gateway  = avg_rev * 0.02
    avg_promo    = avg_rev * 0.05
    net_profit   = avg_rev - avg_cogs - avg_rider - avg_pkg - avg_gateway - avg_promo
    contrib_margin = (net_profit / avg_rev * 100) if avg_rev > 0 else 0

    labels = ["Revenue", "COGS", "Rider Pay", "Packaging", "Gateway Fee", "Promos", "Net Profit"]
    values = [avg_rev, -avg_cogs, -avg_rider, -avg_pkg, -avg_gateway, -avg_promo, net_profit]
    colors = ["#10b981","#ef4444","#ef4444","#f59e0b","#f59e0b","#f59e0b",
              "#10b981" if net_profit > 0 else "#ef4444"]

    fig = go.Figure(go.Waterfall(
        name="Unit Economics",
        orientation="v",
        measure=["absolute","relative","relative","relative","relative","relative","total"],
        x=labels,
        y=values,
        text=[fmt(abs(v)) for v in values],
        textposition="outside",
        connector={"line": {"color": "rgba(99,130,255,.3)"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#6366f1"}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#8899bb", height=320,
                      title=dict(text=f"Revenue → Net Profit Waterfall (Avg per batch) | CM: {contrib_margin:.1f}%",
                                font=dict(color="#f0f4ff", size=12)),
                      margin=dict(l=10,r=10,t=50,b=10),
                      yaxis=dict(gridcolor="rgba(99,130,255,.05)"),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Cost breakdown donut
    cost_labels = ["COGS (52%)", "Rider Pay (12%)", "Packaging (3%)", "Gateway (2%)", "Promos (5%)", "Net Profit (26%)"]
    cost_vals   = [avg_cogs, avg_rider, avg_pkg, avg_gateway, avg_promo, max(0, net_profit)]
    cost_colors = ["#6366f1","#06b6d4","#f59e0b","#8b5cf6","#ec4899","#10b981"]

    fig = go.Figure(go.Pie(
        labels=cost_labels, values=cost_vals,
        hole=0.6, marker=dict(colors=cost_colors),
        textinfo="label+percent", textfont=dict(size=10),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=320,
                      title=dict(text="Cost Structure Breakdown", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=10,t=50,b=10),
                      legend=dict(font=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ORDER DEFECT RATE (FUNNEL) ────────────────────────────────────────────────────
st.markdown('<div class="section-head">🔍 ORDER QUALITY — DEFECT RATE ANALYSIS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    total_ord      = int(total_orders)
    expired        = int(total_ord * 0.018)
    missing        = int(total_ord * 0.024)
    cancelled_oos  = int(total_ord * 0.031)
    total_defects  = expired + missing + cancelled_oos
    perfect_orders = total_ord - total_defects
    odr_pct        = total_defects / total_ord * 100 if total_ord > 0 else 0

    funnel_labels = ["Total Orders", "After Expired/Damaged", "After Missing Items", "After OOS Cancels", "✅ Perfect Orders"]
    funnel_vals   = [total_ord,
                     total_ord - expired,
                     total_ord - expired - missing,
                     total_ord - expired - missing - cancelled_oos,
                     perfect_orders]

    fig = go.Figure(go.Funnel(
        y=funnel_labels, x=funnel_vals,
        textinfo="value+percent initial",
        marker=dict(color=["#6366f1","#8b5cf6","#f59e0b","#ef4444","#10b981"]),
        connector=dict(line=dict(color="rgba(99,130,255,.2)", width=1)),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#8899bb", height=300,
                      title=dict(text=f"Order Quality Funnel | ODR: {odr_pct:.1f}%", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    defect_cats   = ["Expired/Damaged", "Missing Items", "Cancelled (OOS)"]
    defect_counts = [expired, missing, cancelled_oos]
    defect_colors = ["#ef4444","#f59e0b","#8b5cf6"]

    fig = go.Figure(go.Bar(
        x=defect_cats, y=defect_counts,
        marker_color=defect_colors, marker_line_width=0,
        text=defect_counts, textposition="outside",
        textfont=dict(color="#f0f4ff"),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#8899bb", height=300,
                      title=dict(text="Defect Breakdown by Category", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=10,t=50,b=10),
                      yaxis=dict(gridcolor="rgba(99,130,255,.05)"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── INVENTORY INTELLIGENCE ────────────────────────────────────────────────────────
st.markdown('<div class="section-head">📦 INVENTORY INTELLIGENCE — STOCK ALERTS</div>', unsafe_allow_html=True)

# Calculate velocity (orders per product) and simulate stock levels
prod_velocity = df.groupby("Product Name")["Orders"].sum().sort_values(ascending=False)
np.random.seed(123)
stock_data = []
for prod, velocity in prod_velocity.head(10).items():
    stock_left   = int(np.random.randint(5, 200))
    daily_sales  = int(velocity * 0.3)
    days_left    = stock_left / daily_sales if daily_sales > 0 else 99
    risk         = "🔴 CRITICAL" if days_left < 1 else "🟡 LOW" if days_left < 2 else "🟢 OK"
    risk_color   = "#ef4444" if days_left < 1 else "#f59e0b" if days_left < 2 else "#10b981"
    reorder      = "⚡ ORDER NOW" if days_left < 1 else "📋 Plan Reorder" if days_left < 2 else "✅ Sufficient"
    stock_data.append({
        "Product": prod,
        "Stock Left": stock_left,
        "Daily Sales": daily_sales,
        "Days Cover": round(days_left, 1),
        "Risk": risk,
        "Action": reorder,
        "_color": risk_color,
    })

stock_df = pd.DataFrame(stock_data)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div style="font-size:11px;font-weight:600;color:#a5b4fc;margin-bottom:10px">⚡ Top 10 High-Risk Inventory Items</div>', unsafe_allow_html=True)
    for _, row in stock_df.iterrows():
        clr = row["_color"]
        bg  = "rgba(239,68,68,.08)" if "CRITICAL" in row["Risk"] else "rgba(245,158,11,.08)" if "LOW" in row["Risk"] else "rgba(16,185,129,.06)"
        bc  = "rgba(239,68,68,.3)"  if "CRITICAL" in row["Risk"] else "rgba(245,158,11,.3)"  if "LOW" in row["Risk"] else "rgba(16,185,129,.2)"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {bc};border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between">
          <div>
            <div style="font-size:12px;font-weight:600;color:#f0f4ff">{row['Product']}</div>
            <div style="font-size:10px;color:#8899bb;margin-top:2px">Stock: {row['Stock Left']} units &nbsp;·&nbsp; Daily sales: {row['Daily Sales']} units &nbsp;·&nbsp; Covers: {row['Days Cover']} days</div>
          </div>
          <div style="text-align:right">
            <div style="font-size:11px;font-weight:700;color:{clr}">{row['Risk']}</div>
            <div style="font-size:10px;color:{clr};margin-top:2px">{row['Action']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Stockout risk gauge
    critical_count = len(stock_df[stock_df["Risk"].str.contains("CRITICAL")])
    low_count      = len(stock_df[stock_df["Risk"].str.contains("LOW")])
    ok_count       = len(stock_df[stock_df["Risk"].str.contains("OK")])

    fig = go.Figure(go.Pie(
        labels=["Critical 🔴", "Low Stock 🟡", "OK 🟢"],
        values=[critical_count, low_count, ok_count],
        hole=0.65,
        marker=dict(colors=["#ef4444","#f59e0b","#10b981"]),
        textinfo="label+value",
        textfont=dict(size=11),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb",
                      height=280,
                      title=dict(text="Stock Risk Distribution", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=10,t=50,b=10),
                      legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:8px;padding:12px;text-align:center;margin-top:8px">
      <div style="font-size:22px;font-weight:700;color:#ef4444">{critical_count}</div>
      <div style="font-size:10px;color:#8899bb">Products need immediate reorder</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CUSTOMER RETENTION COHORT ─────────────────────────────────────────────────────
st.markdown('<div class="section-head">👥 CUSTOMER RETENTION — NEW vs REPEAT</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Simulated cohort data
    weeks   = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]
    new_c   = [1200, 980, 1100, 870, 1050, 920]
    repeat_c = [800, 920, 1050, 1100, 1200, 1280]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="New Customers",    x=weeks, y=new_c,    marker_color="#6366f1", marker_line_width=0))
    fig.add_trace(go.Bar(name="Repeat Customers", x=weeks, y=repeat_c, marker_color="#10b981", marker_line_width=0))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#8899bb", height=280, barmode="group",
                      title=dict(text="New vs Repeat Customers (Weekly)", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=10,t=50,b=10),
                      legend=dict(font=dict(size=10)),
                      yaxis=dict(gridcolor="rgba(99,130,255,.05)"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Retention heatmap
    cohort_weeks  = ["Week 1", "Week 2", "Week 3", "Week 4"]
    retention_wks = ["W+0", "W+1", "W+2", "W+3"]
    retention_matrix = [
        [100, 68, 52, 41],
        [100, 71, 55, 43],
        [100, 65, 48, 38],
        [100, 73, 58, 46],
    ]
    fig = go.Figure(go.Heatmap(
        z=retention_matrix,
        x=retention_wks,
        y=cohort_weeks,
        colorscale=[[0,"#0d1628"],[0.4,"#312e81"],[0.7,"#4338ca"],[1,"#6366f1"]],
        text=[[f"{v}%" for v in row] for row in retention_matrix],
        texttemplate="%{text}",
        hovertemplate="Cohort: %{y}<br>Week: %{x}<br>Retention: %{text}<extra></extra>",
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#8899bb", height=280,
                      title=dict(text="Cohort Retention Table (%)", font=dict(color="#f0f4ff", size=13)),
                      margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── BLINKBOT AI CHATBOT ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('''<div class="section-head">🤖 BLINKBOT — AI BUSINESS ANALYST</div>''', unsafe_allow_html=True)

# BlinkBot CSS
st.markdown("""
<style>
.blinkbot-container {
    background: linear-gradient(135deg, #0d1628, #121d35);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 16px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 20px;
}
.blinkbot-header {
    background: linear-gradient(135deg, #1e1b6e, #312e81);
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid rgba(99,102,241,.2);
}
.bot-avatar {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 700;
}
.chat-message-bot {
    background: linear-gradient(135deg, rgba(99,102,241,.1), rgba(6,182,212,.05));
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 12px 12px 12px 0;
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #e2e8f0;
    line-height: 1.6;
}
.chat-message-user {
    background: rgba(30,41,59,.8);
    border: 1px solid rgba(99,130,255,.15);
    border-radius: 12px 12px 0 12px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #cbd5e1;
    text-align: right;
}
.quick-btn {
    display: inline-block;
    background: rgba(99,102,241,.12);
    border: 1px solid rgba(99,102,241,.3);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 11px;
    color: #a5b4fc;
    margin: 3px;
    cursor: pointer;
}
.alert-box {
    background: rgba(239,68,68,.08);
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 12px;
    color: #fca5a5;
}
.insight-box {
    background: rgba(16,185,129,.08);
    border: 1px solid rgba(16,185,129,.25);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 12px;
    color: #6ee7b7;
}
</style>
""", unsafe_allow_html=True)

# ── BlinkBot Brain ────────────────────────────────────────────────────────────────
def blinkbot_analyze(question, df):
    """Core BlinkBot analysis engine."""
    q = question.lower().strip()
    
    if df is None or len(df) == 0:
        return "⚠️ No data loaded yet. Please upload a CSV file to get started!", "warning"
    
    # Pre-compute key metrics
    total_rev     = df["Total Revenue"].sum() if "Total Revenue" in df.columns else 0
    total_orders  = df["Orders"].sum()        if "Orders"        in df.columns else 0
    total_profit  = df["Profit"].sum()        if "Profit"        in df.columns else 0
    margin        = (total_profit/total_rev*100) if total_rev > 0 else 0
    
    cat_rev  = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)  if "Category"     in df.columns else None
    city_rev = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)       if "City"         in df.columns else None
    prod_rev = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False) if "Product Name" in df.columns else None
    
    def fmt(n):
        if n >= 1e7:  return f"₹{n/1e7:.2f} Crore"
        if n >= 1e5:  return f"₹{n/1e5:.2f} Lakh"
        if n >= 1e3:  return f"₹{n/1e3:.1f}K"
        return f"₹{int(n):,}"
    
    # ── REVENUE QUESTIONS ──
    if any(w in q for w in ["total revenue", "revenue", "how much", "earnings", "sales total"]):
        top_cat  = cat_rev.index[0]  if cat_rev  is not None else "N/A"
        top_city = city_rev.index[0] if city_rev is not None else "N/A"
        return f"""**📊 Direct Answer:**
Your total revenue is **{fmt(total_rev)}** across **{len(df):,} transactions**.

**📈 Core Metrics:**
- 🏆 Best performing category: **{top_cat}** contributing {f"{cat_rev.iloc[0]/total_rev*100:.1f}%" if cat_rev is not None else "N/A"}
- 📍 Top city: **{top_city}** with {fmt(city_rev.iloc[0]) if city_rev is not None else "N/A"}
- 💰 Total profit embedded: **{fmt(total_profit)}** ({margin:.1f}% margin)

**💡 Business Recommendation:**
Double down on **{top_cat}** in **{top_city}** — this combination is your highest-ROI opportunity right now. Consider allocating 30% more marketing budget here.""", "success"

    # ── PROFIT QUESTIONS ──
    elif any(w in q for w in ["profit", "margin", "earnings after", "net"]):
        best_margin_cat = df.groupby("Category")["Profit Margin"].mean().sort_values(ascending=False) if "Profit Margin" in df.columns and "Category" in df.columns else None
        worst_margin_cat = best_margin_cat.index[-1] if best_margin_cat is not None else "N/A"
        best_margin_name = best_margin_cat.index[0]  if best_margin_cat is not None else "N/A"
        return f"""**💰 Direct Answer:**
Your total profit is **{fmt(total_profit)}** with an overall margin of **{margin:.1f}%**.

**📈 Core Metrics:**
- 🏆 Highest margin category: **{best_margin_name}** ({f"{best_margin_cat.iloc[0]:.1f}%" if best_margin_cat is not None else "N/A"})
- ⚠️ Lowest margin category: **{worst_margin_cat}** — needs pricing review
- 📊 Revenue to profit conversion: every ₹100 earned = ₹{margin:.0f} kept

**💡 Business Recommendation:**
Focus on growing **{best_margin_name}** sales volume — it gives you the best return. For **{worst_margin_cat}**, either raise prices by 5–8% or reduce procurement costs.""", "success"

    # ── BEST PRODUCT ──
    elif any(w in q for w in ["best product", "top product", "highest selling", "best selling", "number one"]):
        if prod_rev is not None:
            top3 = prod_rev.head(3)
            return f"""**🏆 Direct Answer:**
Your #1 best-selling product is **{top3.index[0]}** generating **{fmt(top3.iloc[0])}** in revenue.

**📈 Top 3 Products:**
1. 🥇 **{top3.index[0]}** — {fmt(top3.iloc[0])}
2. 🥈 **{top3.index[1] if len(top3)>1 else "N/A"}** — {fmt(top3.iloc[1]) if len(top3)>1 else "N/A"}
3. 🥉 **{top3.index[2] if len(top3)>2 else "N/A"}** — {fmt(top3.iloc[2]) if len(top3)>2 else "N/A"}

**💡 Business Recommendation:**
Ensure **{top3.index[0]}** is always in stock (never goes below 20 units). Consider creating a bundle offer with your #2 and #3 products to boost average order value.""", "success"
        return "Product data not available in your dataset.", "warning"

    # ── WORST PRODUCT ──
    elif any(w in q for w in ["worst product", "lowest", "weakest product", "poor performing product", "bad product"]):
        if prod_rev is not None:
            worst = prod_rev.tail(3).sort_values()
            return f"""**⚠️ Operational Alert — Underperforming Products:**
Your lowest-revenue product is **{worst.index[0]}** generating only **{fmt(worst.iloc[0])}**.

**📉 Bottom 3 Products:**
1. 🔴 **{worst.index[0]}** — {fmt(worst.iloc[0])}
2. 🟡 **{worst.index[1] if len(worst)>1 else "N/A"}** — {fmt(worst.iloc[1]) if len(worst)>1 else "N/A"}
3. 🟡 **{worst.index[2] if len(worst)>2 else "N/A"}** — {fmt(worst.iloc[2]) if len(worst)>2 else "N/A"}

**💡 Business Recommendation:**
Run a **30-day promotion** on these products — a 10% discount often triggers enough volume to move them up the rankings. If sales don't improve, consider discontinuing the lowest one to free up shelf space.""", "warning"
        return "Product data not available.", "warning"

    # ── CITY/REGION QUESTIONS ──
    elif any(w in q for w in ["city", "region", "location", "where", "best city", "top city", "worst city", "which city"]):
        if city_rev is not None:
            top_city  = city_rev.index[0];  top_v  = city_rev.iloc[0]
            bot_city  = city_rev.index[-1]; bot_v  = city_rev.iloc[-1]
            gap       = (top_v - bot_v) / bot_v * 100 if bot_v > 0 else 0
            alert = (f"\n\n**⚠️ Operational Alert:** {bot_city} is underperforming by **{gap:.0f}%** compared to {top_city}. This gap is significant and needs immediate attention.") if gap > 50 else ""
            return f"""**📍 Direct Answer:**
**{top_city}** is your strongest market with **{fmt(top_v)}** in revenue.

**🗺️ City Performance Ranking:**
{chr(10).join([f"{i+1}. {'🟢' if i==0 else '🟡' if i<len(city_rev)-1 else '🔴'} **{city}** — {fmt(rev)}" for i, (city, rev) in enumerate(city_rev.items())])}{alert}

**💡 Business Recommendation:**
Replicate whatever is working in **{top_city}** (pricing, product mix, influencer activity) and apply it to **{bot_city}**. Start with a targeted influencer campaign in {bot_city} for your top 3 products.""", "success"
        return "City data not found in your dataset.", "warning"

    # ── CATEGORY QUESTIONS ──
    elif any(w in q for w in ["category", "segment", "type", "product type", "best category"]):
        if cat_rev is not None:
            return f"""**🏷️ Direct Answer:**
**{cat_rev.index[0]}** is your top-performing category with **{fmt(cat_rev.iloc[0])}** revenue ({cat_rev.iloc[0]/total_rev*100:.1f}% of total).

**📊 Category Breakdown:**
{chr(10).join([f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '▫️'} **{cat}** — {fmt(rev)} ({rev/total_rev*100:.1f}%)" for i, (cat, rev) in enumerate(cat_rev.items())])}

**💡 Business Recommendation:**
**{cat_rev.index[-1]}** is your weakest category at only {cat_rev.iloc[-1]/total_rev*100:.1f}% share. Either invest in growing it with promotions or focus your inventory budget on **{cat_rev.index[0]}** for better returns.""", "success"
        return "Category data not available.", "warning"

    # ── INFLUENCER QUESTIONS ──
    elif any(w in q for w in ["influencer", "marketing", "promotion", "campaign"]):
        if "Influencer Active" in df.columns:
            inf_y = df[df["Influencer Active"]=="Yes"]["Total Revenue"].mean()
            inf_n = df[df["Influencer Active"]=="No"]["Total Revenue"].mean()
            lift  = (inf_y - inf_n) / inf_n * 100 if inf_n > 0 else 0
            inf_y_orders = df[df["Influencer Active"]=="Yes"]["Orders"].mean()
            inf_n_orders = df[df["Influencer Active"]=="No"]["Orders"].mean()
            order_lift = (inf_y_orders - inf_n_orders) / inf_n_orders * 100 if inf_n_orders > 0 else 0
            return f"""**⚡ Direct Answer:**
Influencer marketing is generating a **{lift:+.1f}% revenue lift** compared to non-influencer products.

**📊 Core Metrics:**
- 💰 Avg revenue WITH influencer: **{fmt(inf_y)}** per listing
- 💰 Avg revenue WITHOUT influencer: **{fmt(inf_n)}** per listing
- 📦 Order volume lift: **{order_lift:+.1f}%** more orders with influencers
- 🎯 Active influencer products: **{len(df[df["Influencer Active"]=="Yes"])}** of {len(df)} total

**💡 Business Recommendation:**
{"Influencer marketing is clearly working — scale it up! Activate influencers for ALL products in your top category for maximum ROI." if lift > 5 else "The influencer lift is small. Review your influencer targeting — focus on micro-influencers in specific cities rather than broad campaigns."}""", "success"
        return "Influencer data not available in your dataset.", "warning"

    # ── ORDERS QUESTIONS ──
    elif any(w in q for w in ["orders", "order count", "how many orders", "total orders", "volume"]):
        avg_order_val = total_rev / total_orders if total_orders > 0 else 0
        top_city_orders = df.groupby("City")["Orders"].sum().sort_values(ascending=False) if "City" in df.columns else None
        return f"""**🛒 Direct Answer:**
You have processed **{int(total_orders):,} total orders** across all products and cities.

**📊 Core Metrics:**
- 💰 Average order value: **{fmt(avg_order_val)}**
- 📍 Highest order volume city: **{top_city_orders.index[0] if top_city_orders is not None else "N/A"}** ({int(top_city_orders.iloc[0]):,} orders)
- 📦 Total revenue per order: **{fmt(avg_order_val)}**

**💡 Business Recommendation:**
Focus on increasing your average order value from **{fmt(avg_order_val)}** to **{fmt(avg_order_val*1.15)}** by introducing bundle deals or a "Frequently Bought Together" feature. A 15% AOV increase = 15% more revenue with zero extra acquisition cost.""", "success"

    # ── DISCOUNT QUESTIONS ──
    elif any(w in q for w in ["discount", "offer", "deal", "promo", "price reduction"]):
        if "Discount" in df.columns:
            disc_impact = df.groupby("Discount").agg(avg_rev=("Total Revenue","mean"), avg_orders=("Orders","mean")).reset_index()
            best_disc   = disc_impact.loc[disc_impact["avg_rev"].idxmax()]
            return f"""**🏷️ Direct Answer:**
The most effective discount level is **{int(best_disc["Discount"])}%** — generating the highest average revenue of **{fmt(best_disc["avg_rev"])}** per product.

**📊 Discount Performance:**
{chr(10).join([f"- **{int(r.Discount)}% discount** → Avg Revenue: {fmt(r.avg_rev)} | Avg Orders: {r.avg_orders:.0f}" for _, r in disc_impact.iterrows()])}

**💡 Business Recommendation:**
Stick to **{int(best_disc["Discount"])}% discounts** as your standard promotional rate. Avoid going higher unless clearing dead stock — deep discounts often train customers to wait for sales instead of buying at full price.""", "success"
        return "Discount data not available.", "warning"

    # ── SUMMARY / OVERVIEW ──
    elif any(w in q for w in ["summary", "overview", "brief", "tell me about", "analyze", "analyse", "what can you tell", "insights"]):
        top_cat  = cat_rev.index[0]  if cat_rev  is not None else "N/A"
        top_city = city_rev.index[0] if city_rev is not None else "N/A"
        bot_city = city_rev.index[-1] if city_rev is not None else "N/A"
        top_prod = prod_rev.index[0]  if prod_rev is not None else "N/A"
        return f"""**📋 Executive Summary — Your Business at a Glance:**

**Revenue & Profit:**
- 💰 Total Revenue: **{fmt(total_rev)}** | Profit: **{fmt(total_profit)}** ({margin:.1f}% margin)
- 🛒 Total Orders: **{int(total_orders):,}** | Avg Order Value: **{fmt(total_rev/total_orders if total_orders>0 else 0)}**

**Top Performers:**
- 🏆 Best Category: **{top_cat}** ({f"{cat_rev.iloc[0]/total_rev*100:.1f}%" if cat_rev is not None else "N/A"} of revenue)
- 📍 Best City: **{top_city}** ({fmt(city_rev.iloc[0]) if city_rev is not None else "N/A"})
- ⭐ Best Product: **{top_prod}** ({fmt(prod_rev.iloc[0]) if prod_rev is not None else "N/A"})

**⚠️ Alerts:**
- {f"**{bot_city}** is your weakest region — needs immediate attention" if city_rev is not None else "No city data"}
- {f"**{cat_rev.index[-1]}** category is underperforming at {cat_rev.iloc[-1]/total_rev*100:.1f}% share" if cat_rev is not None else ""}

**💡 Top Recommendation:**
Concentrate resources on **{top_cat} in {top_city}** — this is your growth engine. Simultaneously, investigate why **{bot_city}** is lagging and run a targeted campaign there this month.""", "success"

    # ── STOCK/INVENTORY ──
    elif any(w in q for w in ["stock", "inventory", "reorder", "shortage", "out of stock"]):
        if prod_rev is not None:
            top5 = prod_rev.head(5)
            return f"""**📦 Direct Answer:**
Based on your sales velocity, here are the **5 products most at risk of stockout**:

{chr(10).join([f"{i+1}. 🔴 **{prod}** — High demand ({fmt(rev)} revenue) — Keep minimum 50 units" for i, (prod, rev) in enumerate(top5.items())])}

**💡 Business Recommendation:**
Set up **automatic reorder alerts** when any top-5 product falls below 20 units. For **{top5.index[0]}**, consider keeping a safety stock of 100 units given its revenue importance.""", "warning"
        return "Product data not available.", "warning"

    # ── GREETING ──
    elif any(w in q for w in ["hello", "hi", "hey", "good morning", "good afternoon", "namaste", "hii"]):
        top_cat  = cat_rev.index[0]  if cat_rev  is not None else "your top category"
        top_city = city_rev.index[0] if city_rev is not None else "your top city"
        return f"""👋 **Hello! I'm BlinkBot** — your AI Business Intelligence Analyst.

I have already analyzed your **{len(df):,} transaction records** and I'm ready to help!

**Here's what I found at first glance:**
- 💰 Total Revenue: **{fmt(total_rev)}**
- 🏆 Top Category: **{top_cat}**
- 📍 Top City: **{top_city}**

**Ask me anything like:**
- *"What is my total profit?"*
- *"Which city is performing worst?"*
- *"Should I increase discounts?"*
- *"Give me a full business summary"*""", "success"

    # ── DEFAULT ──
    else:
        cols_available = ", ".join([c for c in df.columns])
        return f"""🤔 I'm not sure I understood that specific question. Let me help you better!

**I can answer questions about:**
- 💰 Revenue & Profit analysis
- 🏆 Best/worst performing products
- 📍 City & regional performance  
- 🏷️ Category breakdowns
- ⚡ Influencer marketing impact
- 🛒 Order volume & trends
- 🏷️ Discount effectiveness
- 📦 Inventory & stock alerts

**Your data has these columns:** {cols_available}

**Try asking:** *"Give me a business summary"* or *"Which product is performing best?"*""", "info"


# ── BlinkBot UI ───────────────────────────────────────────────────────────────────
st.markdown('''
<div class="blinkbot-container">
  <div class="blinkbot-header">
    <div class="bot-avatar">🤖</div>
    <div>
      <div style="font-size:15px;font-weight:700;color:#f0f4ff">BlinkBot</div>
      <div style="font-size:11px;color:#a5b4fc">Senior AI Business Analyst • Always Online</div>
    </div>
    <div style="margin-left:auto;background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);border-radius:20px;padding:4px 10px;font-size:10px;color:#34d399">● Live</div>
  </div>
</div>
''', unsafe_allow_html=True)

# Chat history
if "blinkbot_history" not in st.session_state:
    st.session_state.blinkbot_history = []
    # Welcome message
    st.session_state.blinkbot_history.append({
        "role": "bot",
        "msg": f"👋 **Hi! I'm BlinkBot**, your AI Business Analyst. I've analyzed your **{len(df):,} records**. Ask me anything about your business — revenue, profit, best cities, top products, or say *'Give me a summary'* to get started!"
    })

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.blinkbot_history:
        if msg["role"] == "bot":
            st.markdown(f'''<div class="chat-message-bot">{msg["msg"]}</div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="chat-message-user">💬 {msg["msg"]}</div>''', unsafe_allow_html=True)

# Quick question buttons
st.markdown("**💡 Quick Questions:**", unsafe_allow_html=True)
qcol1, qcol2, qcol3, qcol4 = st.columns(4)
quick_questions = {
    "📊 Revenue Summary":    "Give me a full business summary",
    "🏆 Best Product":       "Which product is performing best?",
    "📍 City Analysis":      "Which city is performing worst?",
    "⚡ Influencer Impact":  "How is influencer marketing performing?",
}
clicked_quick = None
for col, (label, question) in zip([qcol1, qcol2, qcol3, qcol4], quick_questions.items()):
    with col:
        if st.button(label, use_container_width=True):
            clicked_quick = question

# Text input
with st.form(key="blinkbot_form", clear_on_submit=True):
    fcol1, fcol2 = st.columns([5, 1])
    with fcol1:
        user_input = st.text_input(
            "Ask BlinkBot...",
            placeholder="e.g. What is my total profit? Which city is weakest? Should I discount more?",
            label_visibility="collapsed"
        )
    with fcol2:
        submitted = st.form_submit_button("Ask 🤖", use_container_width=True)

# Process input
question_to_answer = None
if submitted and user_input.strip():
    question_to_answer = user_input.strip()
elif clicked_quick:
    question_to_answer = clicked_quick

if question_to_answer:
    st.session_state.blinkbot_history.append({"role": "user", "msg": question_to_answer})
    response, resp_type = blinkbot_analyze(question_to_answer, df)
    st.session_state.blinkbot_history.append({"role": "bot", "msg": response})
    st.rerun()

# Clear chat button
if len(st.session_state.blinkbot_history) > 1:
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.blinkbot_history = []
        st.rerun()


# ── Raw Data Table ────────────────────────────────────────────────────────────────

# ── BLINKBOT AI CHATBOT ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('''<div class="section-head">🤖 BLINKBOT — AI BUSINESS ANALYST</div>''', unsafe_allow_html=True)

# BlinkBot CSS
st.markdown("""
<style>
.blinkbot-container {
    background: linear-gradient(135deg, #0d1628, #121d35);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 16px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 20px;
}
.blinkbot-header {
    background: linear-gradient(135deg, #1e1b6e, #312e81);
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid rgba(99,102,241,.2);
}
.bot-avatar {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 700;
}
.chat-message-bot {
    background: linear-gradient(135deg, rgba(99,102,241,.1), rgba(6,182,212,.05));
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 12px 12px 12px 0;
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #e2e8f0;
    line-height: 1.6;
}
.chat-message-user {
    background: rgba(30,41,59,.8);
    border: 1px solid rgba(99,130,255,.15);
    border-radius: 12px 12px 0 12px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #cbd5e1;
    text-align: right;
}
.quick-btn {
    display: inline-block;
    background: rgba(99,102,241,.12);
    border: 1px solid rgba(99,102,241,.3);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 11px;
    color: #a5b4fc;
    margin: 3px;
    cursor: pointer;
}
.alert-box {
    background: rgba(239,68,68,.08);
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 12px;
    color: #fca5a5;
}
.insight-box {
    background: rgba(16,185,129,.08);
    border: 1px solid rgba(16,185,129,.25);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 12px;
    color: #6ee7b7;
}
</style>
""", unsafe_allow_html=True)

# ── BlinkBot Brain ────────────────────────────────────────────────────────────────
def blinkbot_analyze(question, df):
    """Core BlinkBot analysis engine."""
    q = question.lower().strip()
    
    if df is None or len(df) == 0:
        return "⚠️ No data loaded yet. Please upload a CSV file to get started!", "warning"
    
    # Pre-compute key metrics
    total_rev     = df["Total Revenue"].sum() if "Total Revenue" in df.columns else 0
    total_orders  = df["Orders"].sum()        if "Orders"        in df.columns else 0
    total_profit  = df["Profit"].sum()        if "Profit"        in df.columns else 0
    margin        = (total_profit/total_rev*100) if total_rev > 0 else 0
    
    cat_rev  = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)  if "Category"     in df.columns else None
    city_rev = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)       if "City"         in df.columns else None
    prod_rev = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False) if "Product Name" in df.columns else None
    
    def fmt(n):
        if n >= 1e7:  return f"₹{n/1e7:.2f} Crore"
        if n >= 1e5:  return f"₹{n/1e5:.2f} Lakh"
        if n >= 1e3:  return f"₹{n/1e3:.1f}K"
        return f"₹{int(n):,}"
    
    # ── REVENUE QUESTIONS ──
    if any(w in q for w in ["total revenue", "revenue", "how much", "earnings", "sales total"]):
        top_cat  = cat_rev.index[0]  if cat_rev  is not None else "N/A"
        top_city = city_rev.index[0] if city_rev is not None else "N/A"
        return f"""**📊 Direct Answer:**
Your total revenue is **{fmt(total_rev)}** across **{len(df):,} transactions**.

**📈 Core Metrics:**
- 🏆 Best performing category: **{top_cat}** contributing {f"{cat_rev.iloc[0]/total_rev*100:.1f}%" if cat_rev is not None else "N/A"}
- 📍 Top city: **{top_city}** with {fmt(city_rev.iloc[0]) if city_rev is not None else "N/A"}
- 💰 Total profit embedded: **{fmt(total_profit)}** ({margin:.1f}% margin)

**💡 Business Recommendation:**
Double down on **{top_cat}** in **{top_city}** — this combination is your highest-ROI opportunity right now. Consider allocating 30% more marketing budget here.""", "success"

    # ── PROFIT QUESTIONS ──
    elif any(w in q for w in ["profit", "margin", "earnings after", "net"]):
        best_margin_cat = df.groupby("Category")["Profit Margin"].mean().sort_values(ascending=False) if "Profit Margin" in df.columns and "Category" in df.columns else None
        worst_margin_cat = best_margin_cat.index[-1] if best_margin_cat is not None else "N/A"
        best_margin_name = best_margin_cat.index[0]  if best_margin_cat is not None else "N/A"
        return f"""**💰 Direct Answer:**
Your total profit is **{fmt(total_profit)}** with an overall margin of **{margin:.1f}%**.

**📈 Core Metrics:**
- 🏆 Highest margin category: **{best_margin_name}** ({f"{best_margin_cat.iloc[0]:.1f}%" if best_margin_cat is not None else "N/A"})
- ⚠️ Lowest margin category: **{worst_margin_cat}** — needs pricing review
- 📊 Revenue to profit conversion: every ₹100 earned = ₹{margin:.0f} kept

**💡 Business Recommendation:**
Focus on growing **{best_margin_name}** sales volume — it gives you the best return. For **{worst_margin_cat}**, either raise prices by 5–8% or reduce procurement costs.""", "success"

    # ── BEST PRODUCT ──
    elif any(w in q for w in ["best product", "top product", "highest selling", "best selling", "number one"]):
        if prod_rev is not None:
            top3 = prod_rev.head(3)
            return f"""**🏆 Direct Answer:**
Your #1 best-selling product is **{top3.index[0]}** generating **{fmt(top3.iloc[0])}** in revenue.

**📈 Top 3 Products:**
1. 🥇 **{top3.index[0]}** — {fmt(top3.iloc[0])}
2. 🥈 **{top3.index[1] if len(top3)>1 else "N/A"}** — {fmt(top3.iloc[1]) if len(top3)>1 else "N/A"}
3. 🥉 **{top3.index[2] if len(top3)>2 else "N/A"}** — {fmt(top3.iloc[2]) if len(top3)>2 else "N/A"}

**💡 Business Recommendation:**
Ensure **{top3.index[0]}** is always in stock (never goes below 20 units). Consider creating a bundle offer with your #2 and #3 products to boost average order value.""", "success"
        return "Product data not available in your dataset.", "warning"

    # ── WORST PRODUCT ──
    elif any(w in q for w in ["worst product", "lowest", "weakest product", "poor performing product", "bad product"]):
        if prod_rev is not None:
            worst = prod_rev.tail(3).sort_values()
            return f"""**⚠️ Operational Alert — Underperforming Products:**
Your lowest-revenue product is **{worst.index[0]}** generating only **{fmt(worst.iloc[0])}**.

**📉 Bottom 3 Products:**
1. 🔴 **{worst.index[0]}** — {fmt(worst.iloc[0])}
2. 🟡 **{worst.index[1] if len(worst)>1 else "N/A"}** — {fmt(worst.iloc[1]) if len(worst)>1 else "N/A"}
3. 🟡 **{worst.index[2] if len(worst)>2 else "N/A"}** — {fmt(worst.iloc[2]) if len(worst)>2 else "N/A"}

**💡 Business Recommendation:**
Run a **30-day promotion** on these products — a 10% discount often triggers enough volume to move them up the rankings. If sales don't improve, consider discontinuing the lowest one to free up shelf space.""", "warning"
        return "Product data not available.", "warning"

    # ── CITY/REGION QUESTIONS ──
    elif any(w in q for w in ["city", "region", "location", "where", "best city", "top city", "worst city", "which city"]):
        if city_rev is not None:
            top_city  = city_rev.index[0];  top_v  = city_rev.iloc[0]
            bot_city  = city_rev.index[-1]; bot_v  = city_rev.iloc[-1]
            gap       = (top_v - bot_v) / bot_v * 100 if bot_v > 0 else 0
            alert = (f"\n\n**⚠️ Operational Alert:** {bot_city} is underperforming by **{gap:.0f}%** compared to {top_city}. This gap is significant and needs immediate attention.") if gap > 50 else ""
            return f"""**📍 Direct Answer:**
**{top_city}** is your strongest market with **{fmt(top_v)}** in revenue.

**🗺️ City Performance Ranking:**
{chr(10).join([f"{i+1}. {'🟢' if i==0 else '🟡' if i<len(city_rev)-1 else '🔴'} **{city}** — {fmt(rev)}" for i, (city, rev) in enumerate(city_rev.items())])}{alert}

**💡 Business Recommendation:**
Replicate whatever is working in **{top_city}** (pricing, product mix, influencer activity) and apply it to **{bot_city}**. Start with a targeted influencer campaign in {bot_city} for your top 3 products.""", "success"
        return "City data not found in your dataset.", "warning"

    # ── CATEGORY QUESTIONS ──
    elif any(w in q for w in ["category", "segment", "type", "product type", "best category"]):
        if cat_rev is not None:
            return f"""**🏷️ Direct Answer:**
**{cat_rev.index[0]}** is your top-performing category with **{fmt(cat_rev.iloc[0])}** revenue ({cat_rev.iloc[0]/total_rev*100:.1f}% of total).

**📊 Category Breakdown:**
{chr(10).join([f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else '▫️'} **{cat}** — {fmt(rev)} ({rev/total_rev*100:.1f}%)" for i, (cat, rev) in enumerate(cat_rev.items())])}

**💡 Business Recommendation:**
**{cat_rev.index[-1]}** is your weakest category at only {cat_rev.iloc[-1]/total_rev*100:.1f}% share. Either invest in growing it with promotions or focus your inventory budget on **{cat_rev.index[0]}** for better returns.""", "success"
        return "Category data not available.", "warning"

    # ── INFLUENCER QUESTIONS ──
    elif any(w in q for w in ["influencer", "marketing", "promotion", "campaign"]):
        if "Influencer Active" in df.columns:
            inf_y = df[df["Influencer Active"]=="Yes"]["Total Revenue"].mean()
            inf_n = df[df["Influencer Active"]=="No"]["Total Revenue"].mean()
            lift  = (inf_y - inf_n) / inf_n * 100 if inf_n > 0 else 0
            inf_y_orders = df[df["Influencer Active"]=="Yes"]["Orders"].mean()
            inf_n_orders = df[df["Influencer Active"]=="No"]["Orders"].mean()
            order_lift = (inf_y_orders - inf_n_orders) / inf_n_orders * 100 if inf_n_orders > 0 else 0
            return f"""**⚡ Direct Answer:**
Influencer marketing is generating a **{lift:+.1f}% revenue lift** compared to non-influencer products.

**📊 Core Metrics:**
- 💰 Avg revenue WITH influencer: **{fmt(inf_y)}** per listing
- 💰 Avg revenue WITHOUT influencer: **{fmt(inf_n)}** per listing
- 📦 Order volume lift: **{order_lift:+.1f}%** more orders with influencers
- 🎯 Active influencer products: **{len(df[df["Influencer Active"]=="Yes"])}** of {len(df)} total

**💡 Business Recommendation:**
{"Influencer marketing is clearly working — scale it up! Activate influencers for ALL products in your top category for maximum ROI." if lift > 5 else "The influencer lift is small. Review your influencer targeting — focus on micro-influencers in specific cities rather than broad campaigns."}""", "success"
        return "Influencer data not available in your dataset.", "warning"

    # ── ORDERS QUESTIONS ──
    elif any(w in q for w in ["orders", "order count", "how many orders", "total orders", "volume"]):
        avg_order_val = total_rev / total_orders if total_orders > 0 else 0
        top_city_orders = df.groupby("City")["Orders"].sum().sort_values(ascending=False) if "City" in df.columns else None
        return f"""**🛒 Direct Answer:**
You have processed **{int(total_orders):,} total orders** across all products and cities.

**📊 Core Metrics:**
- 💰 Average order value: **{fmt(avg_order_val)}**
- 📍 Highest order volume city: **{top_city_orders.index[0] if top_city_orders is not None else "N/A"}** ({int(top_city_orders.iloc[0]):,} orders)
- 📦 Total revenue per order: **{fmt(avg_order_val)}**

**💡 Business Recommendation:**
Focus on increasing your average order value from **{fmt(avg_order_val)}** to **{fmt(avg_order_val*1.15)}** by introducing bundle deals or a "Frequently Bought Together" feature. A 15% AOV increase = 15% more revenue with zero extra acquisition cost.""", "success"

    # ── DISCOUNT QUESTIONS ──
    elif any(w in q for w in ["discount", "offer", "deal", "promo", "price reduction"]):
        if "Discount" in df.columns:
            disc_impact = df.groupby("Discount").agg(avg_rev=("Total Revenue","mean"), avg_orders=("Orders","mean")).reset_index()
            best_disc   = disc_impact.loc[disc_impact["avg_rev"].idxmax()]
            return f"""**🏷️ Direct Answer:**
The most effective discount level is **{int(best_disc["Discount"])}%** — generating the highest average revenue of **{fmt(best_disc["avg_rev"])}** per product.

**📊 Discount Performance:**
{chr(10).join([f"- **{int(r.Discount)}% discount** → Avg Revenue: {fmt(r.avg_rev)} | Avg Orders: {r.avg_orders:.0f}" for _, r in disc_impact.iterrows()])}

**💡 Business Recommendation:**
Stick to **{int(best_disc["Discount"])}% discounts** as your standard promotional rate. Avoid going higher unless clearing dead stock — deep discounts often train customers to wait for sales instead of buying at full price.""", "success"
        return "Discount data not available.", "warning"

    # ── SUMMARY / OVERVIEW ──
    elif any(w in q for w in ["summary", "overview", "brief", "tell me about", "analyze", "analyse", "what can you tell", "insights"]):
        top_cat  = cat_rev.index[0]  if cat_rev  is not None else "N/A"
        top_city = city_rev.index[0] if city_rev is not None else "N/A"
        bot_city = city_rev.index[-1] if city_rev is not None else "N/A"
        top_prod = prod_rev.index[0]  if prod_rev is not None else "N/A"
        return f"""**📋 Executive Summary — Your Business at a Glance:**

**Revenue & Profit:**
- 💰 Total Revenue: **{fmt(total_rev)}** | Profit: **{fmt(total_profit)}** ({margin:.1f}% margin)
- 🛒 Total Orders: **{int(total_orders):,}** | Avg Order Value: **{fmt(total_rev/total_orders if total_orders>0 else 0)}**

**Top Performers:**
- 🏆 Best Category: **{top_cat}** ({f"{cat_rev.iloc[0]/total_rev*100:.1f}%" if cat_rev is not None else "N/A"} of revenue)
- 📍 Best City: **{top_city}** ({fmt(city_rev.iloc[0]) if city_rev is not None else "N/A"})
- ⭐ Best Product: **{top_prod}** ({fmt(prod_rev.iloc[0]) if prod_rev is not None else "N/A"})

**⚠️ Alerts:**
- {f"**{bot_city}** is your weakest region — needs immediate attention" if city_rev is not None else "No city data"}
- {f"**{cat_rev.index[-1]}** category is underperforming at {cat_rev.iloc[-1]/total_rev*100:.1f}% share" if cat_rev is not None else ""}

**💡 Top Recommendation:**
Concentrate resources on **{top_cat} in {top_city}** — this is your growth engine. Simultaneously, investigate why **{bot_city}** is lagging and run a targeted campaign there this month.""", "success"

    # ── STOCK/INVENTORY ──
    elif any(w in q for w in ["stock", "inventory", "reorder", "shortage", "out of stock"]):
        if prod_rev is not None:
            top5 = prod_rev.head(5)
            return f"""**📦 Direct Answer:**
Based on your sales velocity, here are the **5 products most at risk of stockout**:

{chr(10).join([f"{i+1}. 🔴 **{prod}** — High demand ({fmt(rev)} revenue) — Keep minimum 50 units" for i, (prod, rev) in enumerate(top5.items())])}

**💡 Business Recommendation:**
Set up **automatic reorder alerts** when any top-5 product falls below 20 units. For **{top5.index[0]}**, consider keeping a safety stock of 100 units given its revenue importance.""", "warning"
        return "Product data not available.", "warning"

    # ── GREETING ──
    elif any(w in q for w in ["hello", "hi", "hey", "good morning", "good afternoon", "namaste", "hii"]):
        top_cat  = cat_rev.index[0]  if cat_rev  is not None else "your top category"
        top_city = city_rev.index[0] if city_rev is not None else "your top city"
        return f"""👋 **Hello! I'm BlinkBot** — your AI Business Intelligence Analyst.

I have already analyzed your **{len(df):,} transaction records** and I'm ready to help!

**Here's what I found at first glance:**
- 💰 Total Revenue: **{fmt(total_rev)}**
- 🏆 Top Category: **{top_cat}**
- 📍 Top City: **{top_city}**

**Ask me anything like:**
- *"What is my total profit?"*
- *"Which city is performing worst?"*
- *"Should I increase discounts?"*
- *"Give me a full business summary"*""", "success"

    # ── DEFAULT ──
    else:
        cols_available = ", ".join([c for c in df.columns])
        return f"""🤔 I'm not sure I understood that specific question. Let me help you better!

**I can answer questions about:**
- 💰 Revenue & Profit analysis
- 🏆 Best/worst performing products
- 📍 City & regional performance  
- 🏷️ Category breakdowns
- ⚡ Influencer marketing impact
- 🛒 Order volume & trends
- 🏷️ Discount effectiveness
- 📦 Inventory & stock alerts

**Your data has these columns:** {cols_available}

**Try asking:** *"Give me a business summary"* or *"Which product is performing best?"*""", "info"


# ── BlinkBot UI ───────────────────────────────────────────────────────────────────
st.markdown('''
<div class="blinkbot-container">
  <div class="blinkbot-header">
    <div class="bot-avatar">🤖</div>
    <div>
      <div style="font-size:15px;font-weight:700;color:#f0f4ff">BlinkBot</div>
      <div style="font-size:11px;color:#a5b4fc">Senior AI Business Analyst • Always Online</div>
    </div>
    <div style="margin-left:auto;background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);border-radius:20px;padding:4px 10px;font-size:10px;color:#34d399">● Live</div>
  </div>
</div>
''', unsafe_allow_html=True)

# Chat history
if "blinkbot_history" not in st.session_state:
    st.session_state.blinkbot_history = []
    # Welcome message
    st.session_state.blinkbot_history.append({
        "role": "bot",
        "msg": f"👋 **Hi! I'm BlinkBot**, your AI Business Analyst. I've analyzed your **{len(df):,} records**. Ask me anything about your business — revenue, profit, best cities, top products, or say *'Give me a summary'* to get started!"
    })

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.blinkbot_history:
        if msg["role"] == "bot":
            st.markdown(f'''<div class="chat-message-bot">{msg["msg"]}</div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="chat-message-user">💬 {msg["msg"]}</div>''', unsafe_allow_html=True)

# Quick question buttons
st.markdown("**💡 Quick Questions:**", unsafe_allow_html=True)
qcol1, qcol2, qcol3, qcol4 = st.columns(4)
quick_questions = {
    "📊 Revenue Summary":    "Give me a full business summary",
    "🏆 Best Product":       "Which product is performing best?",
    "📍 City Analysis":      "Which city is performing worst?",
    "⚡ Influencer Impact":  "How is influencer marketing performing?",
}
clicked_quick = None
for col, (label, question) in zip([qcol1, qcol2, qcol3, qcol4], quick_questions.items()):
    with col:
        if st.button(label, use_container_width=True,key=f"btn_{label}"):
            clicked_quick = question

# Text input
with st.form(key="blinkbot_form_1", clear_on_submit=True):
    fcol1, fcol2 = st.columns([5, 1])
    with fcol1:
        user_input = st.text_input(
            "Ask BlinkBot...",
            placeholder="e.g. What is my total profit? Which city is weakest? Should I discount more?",
            label_visibility="collapsed"
        )
    with fcol2:
        submitted = st.form_submit_button("Ask 🤖", use_container_width=True)

# Process input
question_to_answer = None
if submitted and user_input.strip():
    question_to_answer = user_input.strip()
elif clicked_quick:
    question_to_answer = clicked_quick

if question_to_answer:
    st.session_state.blinkbot_history.append({"role": "user", "msg": question_to_answer})
    response, resp_type = blinkbot_analyze(question_to_answer, df)
    st.session_state.blinkbot_history.append({"role": "bot", "msg": response})
    st.rerun()

# Clear chat button
if len(st.session_state.blinkbot_history) > 1:
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.blinkbot_history = []
        st.rerun()


# ── Raw Data Table ────────────────────────────────────────────────────────────────
if show_raw:
    st.markdown('<div class="section-head">RAW DATA TABLE</div>', unsafe_allow_html=True)
    display_cols = ["Product Name","Category","City","Original Price","Current Price","Discount","Orders","Total Revenue","Profit","Profit Margin","Influencer Active"]
    show_df = df[[c for c in display_cols if c in df.columns]].copy()
    show_df["Profit Margin"] = show_df["Profit Margin"].round(1).astype(str) + "%"
    st.dataframe(show_df, use_container_width=True, height=350)

    csv_out = df.to_csv(index=False)
    st.download_button("⬇ Download Filtered CSV", csv_out, "zepto_filtered.csv", "text/csv")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Zepto Sales Intelligence Dashboard v2.0 &nbsp;·&nbsp;
  Developed by <span class="dev">Ayush Mishra</span> &nbsp;·&nbsp;
  FastAPI · Pandas · SciPy · scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)

# ── Auto refresh ─────────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
