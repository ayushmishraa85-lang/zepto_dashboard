"""
Zepto Sales Intelligence Dashboard — FastAPI Backend
Handles: data ingestion, cleaning, statistical analytics, ML forecasting, outlier detection
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io
import json
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Zepto Sales Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory data store ────────────────────────────────────────────────────
_df: Optional[pd.DataFrame] = None

DATA_PATH = "../data/zepto_sales_dataset.csv"


# ─── Data Pipeline ───────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]

    # Drop fully empty rows
    df.dropna(how="all", inplace=True)

    # Numeric coercion
    for col in ["Original Price", "Current Price", "Discount", "Orders", "Total Revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric NaN with column median
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # String columns: fill with mode
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].fillna(df[col].mode()[0]).str.strip()

    # Derived columns
    df["Profit"]        = (df["Current Price"] - df["Original Price"]) * df["Orders"]
    df["Profit Margin"] = np.where(df["Total Revenue"] > 0,
                                   (df["Profit"] / df["Total Revenue"]) * 100, 0)
    df["Price Tier"]    = pd.cut(df["Current Price"],
                                 bins=[0, 60, 100, 140, 180, np.inf],
                                 labels=["₹20–60", "₹61–100", "₹101–140", "₹141–180", "₹181+"])

    return df


def get_df(city=None, category=None, influencer=None, search=None) -> pd.DataFrame:
    global _df
    if _df is None:
        _df = load_data()
    df = _df.copy()
    if city       and city       != "All": df = df[df["City"]               == city]
    if category   and category   != "All": df = df[df["Category"]           == category]
    if influencer and influencer != "All": df = df[df["Influencer Active"]  == influencer]
    if search:
        df = df[df["Product Name"].str.contains(search, case=False, na=False)]
    return df


# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _df
    try:
        _df = load_data()
        print(f"✅ Loaded {len(_df)} rows from CSV.")
    except Exception as e:
        print(f"⚠️  Could not load CSV on startup: {e}")


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "rows": len(_df) if _df is not None else 0}


# ─── Upload ──────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global _df
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted.")
    contents = await file.read()
    try:
        raw = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        _df = clean_data(raw)
        return {"message": f"Loaded {len(_df)} rows from '{file.filename}'", "rows": len(_df)}
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")


# ─── KPI Summary ─────────────────────────────────────────────────────────────

@app.get("/kpis")
def kpis(city: str = "All", category: str = "All",
         influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    if df.empty:
        return {}

    total_rev    = float(df["Total Revenue"].sum())
    total_profit = float(df["Profit"].sum())
    total_orders = int(df["Orders"].sum())
    margin       = (total_profit / total_rev * 100) if total_rev else 0
    avg_price    = float(df["Current Price"].mean())

    cat_rev  = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
    city_rev = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)
    prod_rev = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)

    # Standard deviation of revenue — tells how "spread" the numbers are
    rev_std  = float(df["Total Revenue"].std())
    rev_mean = float(df["Total Revenue"].mean())

    return {
        "total_revenue":    total_rev,
        "total_profit":     total_profit,
        "total_orders":     total_orders,
        "profit_margin":    round(margin, 2),
        "avg_price":        round(avg_price, 2),
        "record_count":     len(df),
        "revenue_mean":     round(rev_mean, 2),
        "revenue_std":      round(rev_std, 2),
        "top_category":     cat_rev.index[0]  if len(cat_rev)  else "—",
        "top_city":         city_rev.index[0] if len(city_rev) else "—",
        "top_product":      prod_rev.index[0] if len(prod_rev) else "—",
        "top_category_rev": float(cat_rev.iloc[0])  if len(cat_rev)  else 0,
        "top_city_rev":     float(city_rev.iloc[0]) if len(city_rev) else 0,
    }


# ─── Revenue by City ─────────────────────────────────────────────────────────

@app.get("/city-revenue")
def city_revenue(city: str = "All", category: str = "All",
                 influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    result = (df.groupby("City")["Total Revenue"]
                .sum().sort_values(ascending=False)
                .reset_index()
                .rename(columns={"Total Revenue": "revenue"}))
    return result.to_dict(orient="records")


# ─── Revenue by Category ─────────────────────────────────────────────────────

@app.get("/category-revenue")
def category_revenue(city: str = "All", category: str = "All",
                     influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    result = (df.groupby("Category")["Total Revenue"]
                .sum().sort_values(ascending=False)
                .reset_index()
                .rename(columns={"Total Revenue": "revenue"}))
    return result.to_dict(orient="records")


# ─── Top Products ─────────────────────────────────────────────────────────────

@app.get("/top-products")
def top_products(n: int = 10, city: str = "All", category: str = "All",
                 influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    result = (df.groupby("Product Name")["Total Revenue"]
                .sum().sort_values(ascending=False)
                .head(n).reset_index()
                .rename(columns={"Total Revenue": "revenue"}))
    return result.to_dict(orient="records")


# ─── Scatter Data ─────────────────────────────────────────────────────────────

@app.get("/scatter")
def scatter(city: str = "All", category: str = "All",
            influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    return df[["Product Name", "Orders", "Total Revenue", "Category"]].rename(
        columns={"Product Name": "name", "Orders": "orders",
                 "Total Revenue": "revenue", "Category": "category"}
    ).to_dict(orient="records")


# ─── Heatmap ─────────────────────────────────────────────────────────────────

@app.get("/heatmap")
def heatmap(city: str = "All", category: str = "All",
            influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    pivot = (df.pivot_table(index="Category", columns="City",
                            values="Total Revenue", aggfunc="sum", fill_value=0))
    return {
        "categories": pivot.index.tolist(),
        "cities":     pivot.columns.tolist(),
        "matrix":     pivot.values.tolist(),
    }


# ─── Influencer Impact ───────────────────────────────────────────────────────

@app.get("/influencer-impact")
def influencer_impact(city: str = "All", category: str = "All", search: str = ""):
    df = get_df(city, category, None, search or None)
    result = (df.groupby(["Category", "Influencer Active"])["Total Revenue"]
                .mean().reset_index()
                .rename(columns={"Total Revenue": "avg_revenue",
                                 "Influencer Active": "influencer"}))
    return result.to_dict(orient="records")


# ─── Discount Analysis ────────────────────────────────────────────────────────

@app.get("/discount-analysis")
def discount_analysis(city: str = "All", category: str = "All",
                      influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    result = (df.groupby("Discount")
                .agg(avg_revenue=("Total Revenue", "mean"),
                     avg_orders=("Orders", "mean"),
                     count=("Orders", "count"))
                .reset_index())
    return result.to_dict(orient="records")


# ─── Price Range Buckets ─────────────────────────────────────────────────────

@app.get("/price-ranges")
def price_ranges(city: str = "All", category: str = "All",
                 influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    result = (df.groupby("Price Tier", observed=True)["Total Revenue"]
                .sum().reset_index()
                .rename(columns={"Price Tier": "tier", "Total Revenue": "revenue"}))
    result["tier"] = result["tier"].astype(str)
    return result.to_dict(orient="records")


# ─── Statistical Analysis (Z-Scores & Outliers) ───────────────────────────────

@app.get("/statistics")
def statistics(city: str = "All", category: str = "All",
               influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    if len(df) < 3:
        return {}

    rev = df["Total Revenue"].values
    z_scores = np.abs(stats.zscore(rev))

    outliers = df[z_scores > 2][["Product Name", "City", "Category", "Total Revenue"]].copy()
    outliers["z_score"] = z_scores[z_scores > 2].round(2)
    outliers = outliers.sort_values("z_score", ascending=False)

    # Shapiro-Wilk normality test (sample ≤ 5000)
    sample = rev[:5000]
    _, p_value = stats.shapiro(sample) if len(sample) >= 3 else (0, 1)

    return {
        "mean":          round(float(np.mean(rev)), 2),
        "median":        round(float(np.median(rev)), 2),
        "std":           round(float(np.std(rev)), 2),
        "variance":      round(float(np.var(rev)), 2),
        "skewness":      round(float(stats.skew(rev)), 3),
        "kurtosis":      round(float(stats.kurtosis(rev)), 3),
        "p_value_normality": round(float(p_value), 4),
        "is_normal":     bool(p_value > 0.05),
        "outlier_count": int(len(outliers)),
        "outliers":      outliers.head(10).to_dict(orient="records"),
    }


# ─── Correlation Analysis ────────────────────────────────────────────────────

@app.get("/correlation")
def correlation(city: str = "All", category: str = "All",
                influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    if len(df) < 5:
        return {}

    numeric = df[["Original Price", "Current Price", "Discount",
                  "Orders", "Total Revenue", "Profit", "Profit Margin"]]
    corr = numeric.corr(method="pearson").round(3)

    # Discount → Orders Pearson r + p-value
    r, p = stats.pearsonr(df["Discount"], df["Orders"])
    # Revenue → Profit
    r2, p2 = stats.pearsonr(df["Total Revenue"], df["Profit"])

    return {
        "matrix":              corr.to_dict(),
        "columns":             corr.columns.tolist(),
        "discount_orders_r":   round(float(r), 3),
        "discount_orders_p":   round(float(p), 4),
        "revenue_profit_r":    round(float(r2), 3),
        "revenue_profit_p":    round(float(p2), 4),
    }


# ─── Forecasting (Linear Regression) ────────────────────────────────────────

@app.get("/forecast")
def forecast(city: str = "All", category: str = "All",
             influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    if len(df) < 5:
        return {}

    prod_rev = (df.groupby("Product Name")["Total Revenue"]
                  .sum().sort_values().values)
    n = len(prod_rev)
    X = np.arange(1, n + 1).reshape(-1, 1)
    y = prod_rev

    model = LinearRegression()
    model.fit(X, y)
    next_val   = float(model.predict([[n + 1]])[0])
    r_squared  = float(model.score(X, y))
    trend_line = model.predict(X).tolist()

    # Confidence interval (±1.96 * std of residuals)
    residuals = y - model.predict(X)
    margin    = 1.96 * float(np.std(residuals))

    mean_y = float(np.mean(y))
    growth = ((next_val - mean_y) / mean_y * 100) if mean_y else 0

    return {
        "next_value":      round(max(0, next_val), 2),
        "growth_pct":      round(growth, 2),
        "r_squared":       round(r_squared, 4),
        "slope":           round(float(model.coef_[0]), 4),
        "intercept":       round(float(model.intercept_), 4),
        "confidence_band": round(margin, 2),
        "actuals":         [round(float(v), 2) for v in y],
        "trend_line":      [round(float(v), 2) for v in trend_line],
        "labels":          list(range(1, n + 1)),
    }


# ─── AI Insights ─────────────────────────────────────────────────────────────

@app.get("/insights")
def insights(city: str = "All", category: str = "All",
             influencer: str = "All", search: str = ""):
    df = get_df(city, category, influencer, search or None)
    if df.empty:
        return []

    out = []

    # Best category
    cat_rev  = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
    best_cat = cat_rev.index[0]
    best_val = float(cat_rev.iloc[0])
    pct_total = (best_val / float(cat_rev.sum()) * 100) if cat_rev.sum() else 0
    out.append({
        "emoji": "🏆", "title": "Best Performing Category",
        "body": f"{best_cat} drives {pct_total:.1f}% of total revenue (₹{best_val:,.0f}). "
                f"It outperforms all other categories and should be the primary marketing focus."
    })

    # Regional gap
    city_rev  = df.groupby("City")["Total Revenue"].sum().sort_values(ascending=False)
    top_city  = city_rev.index[0];  top_v = float(city_rev.iloc[0])
    bot_city  = city_rev.index[-1]; bot_v = float(city_rev.iloc[-1])
    gap_pct   = ((top_v - bot_v) / bot_v * 100) if bot_v else 0
    out.append({
        "emoji": "🌍", "title": "Regional Performance Gap",
        "body": f"{top_city} (₹{top_v:,.0f}) outperforms {bot_city} (₹{bot_v:,.0f}) "
                f"by {gap_pct:.0f}%. Targeted promotions in {bot_city} could close this gap."
    })

    # Influencer lift
    inf_y = df[df["Influencer Active"] == "Yes"]["Total Revenue"]
    inf_n = df[df["Influencer Active"] == "No"]["Total Revenue"]
    if len(inf_y) > 1 and len(inf_n) > 1:
        lift = ((inf_y.mean() - inf_n.mean()) / inf_n.mean() * 100) if inf_n.mean() else 0
        t_stat, p_val = stats.ttest_ind(inf_y, inf_n)
        sig = "statistically significant (p < 0.05)" if p_val < 0.05 else "not statistically significant"
        out.append({
            "emoji": "⚡", "title": "Influencer Revenue Lift",
            "body": f"Influencer-active products generate {lift:+.1f}% revenue vs inactive. "
                    f"This is {sig} (p={p_val:.3f}). Expand influencer program to all categories."
        })

    # Profit margin insight
    total_rev  = float(df["Total Revenue"].sum())
    total_prof = float(df["Profit"].sum())
    margin     = (total_prof / total_rev * 100) if total_rev else 0
    best_margin_cat = df.groupby("Category")["Profit Margin"].mean().sort_values(ascending=False)
    out.append({
        "emoji": "📈", "title": "Profitability Trend",
        "body": f"Overall profit margin is {margin:.1f}%. "
                f"{best_margin_cat.index[0]} has the highest average margin ({best_margin_cat.iloc[0]:.1f}%). "
                f"Focus on premium SKU placement to push margins higher."
    })

    # Discount correlation
    r, p = stats.pearsonr(df["Discount"], df["Orders"])
    direction = "positively" if r > 0 else "negatively"
    out.append({
        "emoji": "💡", "title": "Discount–Orders Correlation",
        "body": f"Discount % is {direction} correlated with order volume (r={r:.3f}, p={p:.3f}). "
                f"{'Discounting drives volume — use strategically for slow movers.' if r > 0 else 'Higher discounts do not reliably increase orders — review your pricing strategy.'}"
    })

    # Top product spotlight
    prod_rev = df.groupby("Product Name")["Total Revenue"].sum().sort_values(ascending=False)
    out.append({
        "emoji": "🎯", "title": "Top Product Spotlight",
        "body": f"{prod_rev.index[0]} generates ₹{prod_rev.iloc[0]:,.0f} — the highest single-product revenue. "
                f"Pair with influencer activation and expand to underperforming regions for compounding growth."
    })

    return out


# ─── Raw Data (paginated) ────────────────────────────────────────────────────

@app.get("/data")
def raw_data(city: str = "All", category: str = "All",
             influencer: str = "All", search: str = "",
             page: int = 1, page_size: int = 50):
    df = get_df(city, category, influencer, search or None)
    total = len(df)
    start = (page - 1) * page_size
    chunk = df.iloc[start: start + page_size].copy()
    chunk["Price Tier"] = chunk["Price Tier"].astype(str)
    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     (total + page_size - 1) // page_size,
        "data":      chunk.to_dict(orient="records"),
    }


# ─── Metadata (filter options) ───────────────────────────────────────────────

@app.get("/meta")
def meta():
    global _df
    if _df is None:
        return {}
    return {
        "cities":     sorted(_df["City"].unique().tolist()),
        "categories": sorted(_df["Category"].unique().tolist()),
        "products":   sorted(_df["Product Name"].unique().tolist()),
    }
