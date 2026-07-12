"""
Ayush Intelligence Hub Dashboard — Streamlit Edition
Phase 4: Anthropic LLM streaming · natural-language BlinkBot · chart detection
Phase 5: CSV + Excel (.xlsx) upload support with validation
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

# Columns every dataset MUST contain for the dashboard's math to work.
# "Influencer Active" is intentionally excluded — it's treated as optional
# throughout the app (every consumer already checks `if "Influencer Active" in df.columns`).
REQUIRED_COLUMNS = [
    "Product Name", "Category", "City",
    "Original Price", "Current Price", "Discount",
    "Orders", "Total Revenue",
]

# Columns that identify the alternate "order-level" export schema used by
# some quick-commerce platforms (one row per order, not per product/city
# aggregate). If ALL of these are present but the REQUIRED_COLUMNS above are
# NOT, _map_alternate_schema() below converts the file automatically instead
# of rejecting it.
_ALT_SCHEMA_MARKERS = ["Product", "Quantity", "Unit_Price", "Revenue"]


def _map_alternate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and convert the order-level export schema:

        Order_ID, Platform, Order_Date, City, Category, Product, Quantity,
        Unit_Price, Discount, Delivery_Fee, Delivery_Time_Min,
        Customer_Rating, Payment_Method, Revenue, Cost, Profit

    into the dashboard's canonical schema (REQUIRED_COLUMNS), so it validates
    and flows through clean() unchanged.

    In this export, `Discount` is an absolute ₹ amount off the line total
    (Quantity * Unit_Price), NOT a percentage, and:
        Revenue = Quantity * Unit_Price - Discount + Delivery_Fee
        Profit  = Revenue - Cost

    This function is a no-op if the file already matches REQUIRED_COLUMNS,
    or if it doesn't match the alternate schema either.
    """
    df.columns = [str(c).strip() for c in df.columns]

    already_canonical = all(c in df.columns for c in REQUIRED_COLUMNS)
    is_alt_schema      = all(c in df.columns for c in _ALT_SCHEMA_MARKERS)

    if already_canonical or not is_alt_schema:
        return df

    df = df.copy()
    qty          = pd.to_numeric(df["Quantity"],   errors="coerce").fillna(1)
    unit_price   = pd.to_numeric(df["Unit_Price"], errors="coerce").fillna(0)
    discount_amt = pd.to_numeric(df["Discount"],   errors="coerce").fillna(0)  # absolute ₹, not %
    line_total   = qty * unit_price

    df["Product Name"]   = df["Product"]
    df["Orders"]          = qty
    df["Total Revenue"]   = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Original Price"]  = unit_price
    # Effective per-unit price the customer actually paid, after discount.
    df["Current Price"]   = np.where(
        qty > 0, (line_total - discount_amt) / qty, unit_price
    ).round(2)
    # Convert absolute discount to a % of the line total so it plays nicely
    # with the dashboard's discount-tier binning / correlation logic.
    df["Discount"] = np.where(
        line_total > 0, discount_amt / line_total * 100, 0
    ).round(1)

    return df


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


def _read_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    """
    Detect the uploaded file's type by extension and parse it into a
    raw (un-cleaned) DataFrame.

    Supports .csv and .xlsx. Raises ValueError with a user-friendly
    message for anything else, or if parsing fails outright.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            raise ValueError(
                f"Couldn't parse **{uploaded_file.name}** as CSV. "
                f"Check that it's a valid comma-separated file. ({e})"
            )

    elif filename.endswith(".xlsx"):
        try:
            # First sheet by default — most exports from Zepto/Blinkit/
            # Swiggy Instamart-style tools ship a single-sheet workbook.
            return pd.read_excel(uploaded_file, engine="openpyxl")
        except ImportError:
            raise ValueError(
                "Reading .xlsx files requires the `openpyxl` package. "
                "Install it with `pip install openpyxl` and retry."
            )
        except Exception as e:
            raise ValueError(
                f"Couldn't parse **{uploaded_file.name}** as an Excel file. "
                f"Make sure it's a valid, non-password-protected .xlsx workbook. ({e})"
            )

    else:
        raise ValueError(
            "Unsupported file type. Please upload a **.csv** or **.xlsx** file."
        )


def _validate_columns(df: pd.DataFrame, filename: str) -> None:
    """Raise ValueError if any required column is missing from df."""
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"**{filename}** is missing required column(s): "
            + ", ".join(f"`{m}`" for m in missing)
            + ". Expected columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
            + " (optional: `Influencer Active`)."
        )
    if df.dropna(how="all").empty:
        raise ValueError(f"**{filename}** was read successfully but contains no data rows.")


def load_user_file(uploaded_file) -> pd.DataFrame:
    """
    End-to-end loader for a user-uploaded CSV or XLSX file:
      1. Detect type & parse (_read_uploaded_dataframe)
      2. Map alternate/order-level export schemas onto the canonical one
         (_map_alternate_schema) — no-op if already canonical
      3. Validate required columns (_validate_columns)
      4. Clean, impute, and engineer features (clean)

    Works with exports from Zepto, Blinkit, Swiggy Instamart, or any
    quick-commerce platform — either the aggregate schema (Product Name,
    Orders, Total Revenue, ...) or the order-level schema (Product,
    Quantity, Unit_Price, Revenue, ...). Column order and extra columns
    don't matter.

    Raises ValueError on any recoverable problem; the caller is expected
    to catch it and fall back to the default dataset.
    """
    raw_df = _read_uploaded_dataframe(uploaded_file)
    raw_df = _map_alternate_schema(raw_df)
    _validate_columns(raw_df, uploaded_file.name)
    return clean(raw_df)
