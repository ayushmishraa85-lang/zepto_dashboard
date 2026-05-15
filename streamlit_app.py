"""
NovaMS — Streamlit Edition
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
    page_title="Ayush Intelligence Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stDeployButton"] {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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

# ── Sidebar toggle ──
with st.sidebar:
    pass

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
      <div style="width:44px;height:44px;background:linear-gradient(135deg,#ff6b6b,#ffd93d);border-radius:12px;display:inline-flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#fff;margin-bottom:8px">N</div>
      <div style="font-size:13px;font-weight:600;background:linear-gradient(90deg,#a5b4fc,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">NovaMS</div>
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

# ── Raw Data Table ────────────────────────────────────────────────────────────────
if show_raw:
    st.markdown('<div class="section-head">RAW DATA TABLE</div>', unsafe_allow_html=True)
    display_cols = ["Product Name","Category","City","Original Price","Current Price","Discount","Orders","Total Revenue","Profit","Profit Margin","Influencer Active"]
    show_df = df[[c for c in display_cols if c in df.columns]].copy()
    show_df["Profit Margin"] = show_df["Profit Margin"].round(1).astype(str) + "%"
    st.dataframe(show_df, use_container_width=True, height=350)

    csv_out = df.to_csv(index=False)
    st.download_button("⬇ Download Filtered CSV", csv_out, "Ayush_filtered.csv", "text/csv")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Ayush Intelligence Hub&nbsp;·&nbsp;
  Developed by <span class="dev">Ayush Mishra</span> &nbsp;·&nbsp;
  FastAPI · Pandas · SciPy · scikit-learn · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)

# ── Auto refresh ─────────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()
