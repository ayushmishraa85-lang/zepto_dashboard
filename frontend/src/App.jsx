// src/App.jsx
import { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, ReferenceLine,
} from "recharts";
import { useDashboard } from "./hooks/useDashboard";
import KPICard        from "./components/KPICard";
import ChartCard      from "./components/ChartCard";
import Sidebar        from "./components/Sidebar";
import StatisticsPanel from "./components/StatisticsPanel";

// ── Colour palettes ──────────────────────────────────────────────────────────
const PAL     = ["#6366f1","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316","#3b82f6"];
const CAT_C   = { Snacks:"#6366f1", Beverages:"#06b6d4", Grocery:"#10b981", "Instant Food":"#f59e0b", Confectionery:"#ec4899", Dairy:"#8b5cf6" };
const CITY_C  = { Delhi:"#6366f1", Mumbai:"#06b6d4", Bangalore:"#10b981", Hyderabad:"#f59e0b", Chennai:"#ef4444", Pune:"#8b5cf6" };

// ── Helpers ──────────────────────────────────────────────────────────────────
function fmtINR(n) {
  if (n === undefined || n === null) return "—";
  if (typeof n === "string") return n;
  if (n >= 1e7)  return "₹" + (n / 1e7).toFixed(1) + "Cr";
  if (n >= 1e5)  return "₹" + (n / 1e5).toFixed(2) + "L";
  if (n >= 1e3)  return "₹" + (n / 1e3).toFixed(1) + "K";
  return "₹" + Math.round(n).toLocaleString("en-IN");
}

const TT_STYLE = {
  contentStyle: { background: "#0d1628", border: "1px solid rgba(99,130,255,.2)", borderRadius: 8, fontSize: 11 },
  labelStyle:   { color: "#8899bb" },
  cursor:       { fill: "rgba(99,102,241,.07)" },
};

// ── Heatmap Component ─────────────────────────────────────────────────────────
function HeatmapGrid({ data }) {
  if (!data?.categories || !data?.matrix) return <div style={{ color: "#4a5a7a", padding: 20, textAlign: "center" }}>No data</div>;
  const { categories, cities, matrix } = data;
  const flat = matrix.flat();
  const maxV = Math.max(...flat, 1);

  function cellColor(v) {
    const t = v / maxV;
    const r = Math.round(60  + t * 150);
    const g = Math.round(30  + t * 80);
    const b = Math.round(140 + t * 60);
    return { bg: `rgba(${r+40},${g+30},${b+40},${.1 + t * .75})`, txt: t > .4 ? "#eef2ff" : "#8899bb" };
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "separate", borderSpacing: 3, width: "100%", minWidth: 480 }}>
        <thead>
          <tr>
            <th style={{ fontSize: 9, color: "#4a5a7a", padding: "4px 8px", textAlign: "left" }} />
            {cities.map(c => <th key={c} style={{ fontSize: 9, color: "#4a5a7a", fontWeight: 500, textAlign: "center", padding: "4px 3px" }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {categories.map((cat, ri) => (
            <tr key={cat}>
              <td style={{ fontSize: 9, color: "#4a5a7a", fontWeight: 500, padding: "0 8px 0 0", whiteSpace: "nowrap" }}>{cat}</td>
              {cities.map((city, ci) => {
                const v = matrix[ri]?.[ci] || 0;
                const { bg, txt } = cellColor(v);
                return (
                  <td key={city} title={`${city} – ${cat}: ${fmtINR(v)}`}
                    style={{ background: bg, color: txt, borderRadius: 4, padding: "6px 3px", textAlign: "center", fontSize: 8, fontWeight: 600, cursor: "default", transition: "transform .15s" }}
                    onMouseEnter={e => e.currentTarget.style.transform = "scale(1.1)"}
                    onMouseLeave={e => e.currentTarget.style.transform = ""}
                  >
                    {v > 0 ? fmtINR(v) : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Forecast Chart ────────────────────────────────────────────────────────────
function ForecastSection({ data }) {
  if (!data) return null;
  const pts = (data.labels || []).map((x, i) => ({
    x,
    actual:   data.actuals?.[i] ?? null,
    trend:    data.trend_line?.[i] ?? null,
    upperBand: (data.trend_line?.[i] ?? 0) + (data.confidence_band || 0),
    lowerBand: Math.max(0, (data.trend_line?.[i] ?? 0) - (data.confidence_band || 0)),
  }));
  // Add forecast point
  pts.push({
    x: (data.labels?.length || 0) + 1,
    actual: null,
    trend:  data.next_value,
    upperBand: data.next_value + (data.confidence_band || 0),
    lowerBand: Math.max(0, data.next_value - (data.confidence_band || 0)),
  });

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, margin: "8px 0 14px" }}>
        <div style={{ fontSize: 30, fontWeight: 700, fontFamily: "monospace", background: "linear-gradient(90deg,#a5b4fc,#67e8f9)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          {fmtINR(data.next_value)}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <span style={{ fontSize: 10, fontWeight: 600, padding: "3px 8px", borderRadius: 5, background: data.growth_pct >= 0 ? "rgba(16,185,129,.12)" : "rgba(239,68,68,.12)", color: data.growth_pct >= 0 ? "#34d399" : "#f87171" }}>
            {data.growth_pct >= 0 ? "↑" : "↓"} {Math.abs(data.growth_pct).toFixed(1)}% vs avg
          </span>
          <span style={{ fontSize: 10, padding: "3px 8px", borderRadius: 5, background: "rgba(6,182,212,.1)", color: "#67e8f9" }}>
            R² = {data.r_squared}
          </span>
        </div>
      </div>
      <div style={{ fontSize: 10, color: "#4a5a7a", marginBottom: 12 }}>
        ±{fmtINR(data.confidence_band)} confidence band &nbsp;|&nbsp; slope = {data.slope} &nbsp;|&nbsp; intercept = {fmtINR(data.intercept)}
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={pts.slice(-20)}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,130,255,.05)" />
          <XAxis dataKey="x" tick={{ fill: "#8899bb", fontSize: 9 }} />
          <YAxis tickFormatter={fmtINR} tick={{ fill: "#8899bb", fontSize: 9 }} />
          <Tooltip formatter={(v) => fmtINR(v)} {...TT_STYLE} />
          <Legend wrapperStyle={{ fontSize: 10, color: "#8899bb" }} />
          <Line type="monotone" dataKey="actual"    stroke="#6366f1" strokeWidth={2} dot={{ r: 2 }} name="Actual"    connectNulls={false} />
          <Line type="monotone" dataKey="trend"     stroke="#06b6d4" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 2 }} name="Trend" />
          <Line type="monotone" dataKey="upperBand" stroke="#8b5cf6" strokeWidth={1} strokeDasharray="2 4" dot={false} name="Upper CI" />
          <Line type="monotone" dataKey="lowerBand" stroke="#8b5cf6" strokeWidth={1} strokeDasharray="2 4" dot={false} name="Lower CI" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const { filters, updateFilter, data, loading, error, backendOk,
          refresh, startAutoRefresh, stopAutoRefresh } = useDashboard();

  const [clock, setClock]       = useState("");
  const [autoOn, setAutoOn]     = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");

  // Clock
  useEffect(() => {
    const id = setInterval(() => {
      const n = new Date();
      setClock(n.toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true }));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const toggleAuto = () => {
    if (autoOn) { stopAutoRefresh(); setAutoOn(false); }
    else        { startAutoRefresh(30000); setAutoOn(true); }
  };

  const handleUpload = async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      const { api } = await import("./utils/api");
      const res = await api.uploadCSV(file);
      setUploadMsg(res.message);
      setTimeout(() => setUploadMsg(""), 4000);
      refresh();
    } catch (e) {
      setUploadMsg("Upload failed: " + e.message);
    } finally {
      setUploading(false);
    }
  };

  const { kpis, cityRevenue, categoryRevenue, topProducts, scatter,
          heatmap, influencer, discount, priceRanges,
          statistics, correlation, forecast, insights, meta } = data;

  return (
    <div style={{ fontFamily: "'Inter', system-ui, sans-serif", background: "#070d1a", color: "#f0f4ff", minHeight: "100vh", fontSize: 13 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: #090f1e; }
        ::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 2px; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
      `}</style>

      {/* ── HEADER ── */}
      <header style={{
        position: "sticky", top: 0, zIndex: 200,
        background: "rgba(7,13,26,.97)", backdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(99,130,255,.1)",
        height: 54, display: "flex", alignItems: "center",
        gap: 12, padding: "0 20px",
      }}>
        <div style={{ width: 36, height: 36, background: "linear-gradient(135deg,#6366f1,#06b6d4)", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: 18, flexShrink: 0, boxShadow: "0 0 16px rgba(99,102,241,.4)" }}>Z</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 600, background: "linear-gradient(90deg,#a5b4fc,#67e8f9,#a5b4fc)", backgroundSize: "200%", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", animation: "shimmer 4s linear infinite" }}>
            Zepto Sales Intelligence Dashboard v2.0
          </div>
          <div style={{ fontSize: 10, color: "#4a5a7a", marginTop: 1 }}>Real-Time Business Insights · FastAPI + React · Statistical Analytics</div>
        </div>

        {/* Backend status */}
        <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, padding: "4px 10px", borderRadius: 6, background: backendOk ? "rgba(16,185,129,.1)" : "rgba(239,68,68,.1)", border: `1px solid ${backendOk ? "rgba(16,185,129,.3)" : "rgba(239,68,68,.3)"}`, color: backendOk ? "#34d399" : "#f87171" }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: backendOk ? "#10b981" : "#ef4444", animation: backendOk ? "pulse 2s infinite" : "" }} />
          {backendOk ? "Backend Live" : "Backend Offline"}
        </div>

        {uploadMsg && <div style={{ fontSize: 10, color: "#34d399", background: "rgba(16,185,129,.1)", padding: "4px 10px", borderRadius: 6, border: "1px solid rgba(16,185,129,.3)" }}>{uploadMsg}</div>}

        <div style={{ fontSize: 10, color: "#4a5a7a", fontFamily: "monospace", background: "#0d1628", border: "1px solid rgba(99,130,255,.1)", padding: "5px 10px", borderRadius: 7 }}>{clock}</div>

        <button onClick={refresh} disabled={loading} style={{ display: "flex", alignItems: "center", gap: 5, background: "#121d35", border: "1px solid rgba(99,130,255,.1)", borderRadius: 7, padding: "6px 12px", color: "#8899bb", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>
          {loading ? "⏳" : "↻"} Refresh
        </button>

        <div title="Ayush Mishra" style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg,#8b5cf6,#ec4899)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, cursor: "pointer", boxShadow: "0 0 12px rgba(139,92,246,.3)" }}>AM</div>
      </header>

      <div style={{ display: "flex" }}>
        {/* ── SIDEBAR ── */}
        <Sidebar
          filters={filters}
          onFilter={updateFilter}
          meta={meta}
          onUpload={handleUpload}
          autoRefresh={autoOn}
          onAutoRefresh={toggleAuto}
        />

        {/* ── MAIN ── */}
        <main style={{ flex: 1, padding: 20, minWidth: 0 }}>

          {!backendOk && (
            <div style={{ background: "rgba(245,158,11,.1)", border: "1px solid rgba(245,158,11,.3)", borderRadius: 10, padding: 16, marginBottom: 16, fontSize: 12, color: "#fcd34d" }}>
              ⚠ <strong>Backend not reachable.</strong> Start the FastAPI server: <code style={{ fontFamily: "monospace", background: "rgba(0,0,0,.3)", padding: "2px 6px", borderRadius: 4 }}>cd backend && uvicorn main:app --reload</code>. The dashboard will auto-connect once running.
            </div>
          )}

          {error && (
            <div style={{ background: "rgba(239,68,68,.1)", border: "1px solid rgba(239,68,68,.3)", borderRadius: 10, padding: 14, marginBottom: 16, fontSize: 12, color: "#f87171" }}>
              ❌ {error}
            </div>
          )}

          {/* ── KPIs ── */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#8899bb", letterSpacing: ".08em", textTransform: "uppercase" }}>Key Performance Indicators</div>
            <div style={{ fontSize: 9, fontWeight: 600, padding: "3px 10px", borderRadius: 20, background: "rgba(16,185,129,.12)", color: "#34d399", border: "1px solid rgba(16,185,129,.25)", display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#10b981", animation: "pulse 2s infinite" }} />
              {kpis?.record_count || 0} records
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
            <KPICard type="revenue"  value={kpis?.total_revenue}  sub={`σ = ${fmtINR(kpis?.revenue_std)}`}  change="+12.4%" up />
            <KPICard type="profit"   value={kpis?.total_profit}   sub="Net margin earnings"                  change="+8.1%"  up />
            <KPICard type="orders"   value={kpis?.total_orders}   sub="Units sold"                           change="+5.3%"  up />
            <KPICard type="margin"   value={kpis?.profit_margin != null ? kpis.profit_margin.toFixed(1) + "%" : "—"} sub="Revenue to profit" change="+2.1%" up />
            <KPICard type="top_cat"  value={kpis?.top_category}   sub={fmtINR(kpis?.top_category_rev)}       up={false} />
            <KPICard type="top_city" value={kpis?.top_city}       sub={fmtINR(kpis?.top_city_rev)}           up={false} />
          </div>

          {/* ── CHARTS ROW 1 ── */}
          <SectionHead title="Sales & Revenue Analytics" badge="Interactive" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>

            <ChartCard title="Revenue by City" subtitle="Total revenue contribution per region">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={cityRevenue} margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,130,255,.05)" />
                  <XAxis dataKey="City" tick={{ fill: "#8899bb", fontSize: 10 }} />
                  <YAxis tickFormatter={fmtINR} tick={{ fill: "#8899bb", fontSize: 9 }} />
                  <Tooltip formatter={(v) => fmtINR(v)} {...TT_STYLE} />
                  <Bar dataKey="revenue" radius={[5, 5, 0, 0]} name="Revenue">
                    {cityRevenue.map((entry, i) => <Cell key={i} fill={CITY_C[entry.City] || PAL[i % PAL.length]} fillOpacity={0.85} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Category Distribution" subtitle="Revenue share across product categories">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={categoryRevenue} dataKey="revenue" nameKey="Category" cx="50%" cy="50%" outerRadius={80} innerRadius={45} paddingAngle={3} label={({ Category, percent }) => `${Category} ${(percent * 100).toFixed(0)}%`} labelLine={false}>
                    {categoryRevenue.map((entry, i) => <Cell key={i} fill={CAT_C[entry.Category] || PAL[i % PAL.length]} />)}
                  </Pie>
                  <Tooltip formatter={(v) => fmtINR(v)} {...TT_STYLE} />
                </PieChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* ── CHARTS ROW 2 ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>

            <ChartCard title="Top 10 Products" subtitle="Best-performing products by total revenue">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={topProducts} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,130,255,.05)" horizontal={false} />
                  <XAxis type="number" tickFormatter={fmtINR} tick={{ fill: "#8899bb", fontSize: 9 }} />
                  <YAxis type="category" dataKey="Product Name" tick={{ fill: "#94a3b8", fontSize: 9 }} width={80} />
                  <Tooltip formatter={(v) => fmtINR(v)} {...TT_STYLE} />
                  <Bar dataKey="revenue" radius={[0, 4, 4, 0]} name="Revenue">
                    {topProducts.map((_, i) => <Cell key={i} fill={PAL[i % PAL.length]} fillOpacity={0.85} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Orders vs Revenue (Scatter)" subtitle="Each point = one product-city combination">
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,130,255,.05)" />
                  <XAxis dataKey="orders" name="Orders"  tick={{ fill: "#8899bb", fontSize: 9 }} label={{ value: "Orders", fill: "#4a5a7a", fontSize: 10, position: "insideBottom", offset: -2 }} />
                  <YAxis dataKey="revenue" name="Revenue" tickFormatter={fmtINR} tick={{ fill: "#8899bb", fontSize: 9 }} />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ payload }) => {
                    if (!payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={{ background: "#0d1628", border: "1px solid rgba(99,130,255,.2)", borderRadius: 8, padding: "8px 12px", fontSize: 11 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4, color: "#f0f4ff" }}>{d.name}</div>
                        <div style={{ color: "#8899bb" }}>Orders: {d.orders}</div>
                        <div style={{ color: "#67e8f9" }}>Revenue: {fmtINR(d.revenue)}</div>
                        <div style={{ color: "#4a5a7a" }}>{d.category}</div>
                      </div>
                    );
                  }} />
                  <Scatter data={scatter} name="Products" fill="#6366f1" fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* ── HEATMAP ── */}
          <ChartCard title="City × Category Revenue Heatmap" subtitle="Revenue intensity — deeper colour = higher value" style={{ marginBottom: 14 }}>
            <HeatmapGrid data={heatmap} />
          </ChartCard>

          {/* ── INFLUENCER + DISCOUNT ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>

            <ChartCard title="Influencer Impact by Category" subtitle="Avg revenue: active vs inactive influencer">
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={
                  (() => {
                    const cats = [...new Set(influencer.map(d => d.Category))];
                    return cats.map(cat => {
                      const y = influencer.find(d => d.Category === cat && d.influencer === "Yes");
                      const n = influencer.find(d => d.Category === cat && d.influencer === "No");
                      return { Category: cat, "Influencer Active": y?.avg_revenue || 0, "No Influencer": n?.avg_revenue || 0 };
                    });
                  })()
                }>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,130,255,.05)" />
                  <XAxis dataKey="Category" tick={{ fill: "#8899bb", fontSize: 9 }} />
                  <YAxis tickFormatter={fmtINR} tick={{ fill: "#8899bb", fontSize: 9 }} />
                  <Tooltip formatter={(v) => fmtINR(v)} {...TT_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 10, color: "#8899bb" }} />
                  <Bar dataKey="Influencer Active" fill="#6366f1" fillOpacity={0.8} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="No Influencer"    fill="#4a5a7a" fillOpacity={0.6} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Discount vs Avg Revenue" subtitle="Does higher discount drive more revenue?">
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={discount}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,130,255,.05)" />
                  <XAxis dataKey="Discount" tickFormatter={v => `${v}%`} tick={{ fill: "#8899bb", fontSize: 10 }} />
                  <YAxis tickFormatter={fmtINR} tick={{ fill: "#8899bb", fontSize: 9 }} />
                  <Tooltip formatter={(v, n) => [fmtINR(v), n]} {...TT_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 10, color: "#8899bb" }} />
                  <Bar dataKey="avg_revenue" name="Avg Revenue" radius={[5, 5, 0, 0]}>
                    <Cell fill="#10b981" fillOpacity={0.8} />
                    <Cell fill="#f59e0b" fillOpacity={0.8} />
                    <Cell fill="#ef4444" fillOpacity={0.8} />
                  </Bar>
                  <Bar dataKey="avg_orders" name="Avg Orders" fill="#6366f1" fillOpacity={0.5} radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* ── STATISTICAL ANALYSIS ── */}
          <SectionHead title="Statistical Analysis" badge="Z-Scores · Outlier Detection · Correlation" />
          <StatisticsPanel stats={statistics} correlation={correlation} />

          {/* ── PRICE RANGES ── */}
          <ChartCard title="Revenue by Price Tier" subtitle="Products grouped by current price range" style={{ marginBottom: 14 }}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={priceRanges}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,130,255,.05)" />
                <XAxis dataKey="tier" tick={{ fill: "#8899bb", fontSize: 10 }} />
                <YAxis tickFormatter={fmtINR} tick={{ fill: "#8899bb", fontSize: 9 }} />
                <Tooltip formatter={(v) => fmtINR(v)} {...TT_STYLE} />
                <Bar dataKey="revenue" name="Revenue" radius={[6, 6, 0, 0]}>
                  {priceRanges.map((_, i) => <Cell key={i} fill={PAL[i % PAL.length]} fillOpacity={0.8} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* ── FORECAST ── */}
          <SectionHead title="Sales Forecasting" badge="Linear Regression · Confidence Bands · R²" />
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14, marginBottom: 14 }}>
            <ChartCard title="Next Period Revenue Forecast" subtitle="Linear regression with ±1.96σ confidence interval">
              <ForecastSection data={forecast} />
            </ChartCard>
            <ChartCard title="Model Stats" subtitle="Linear regression diagnostics">
              {forecast && (
                <div>
                  {[
                    { label: "Next Predicted Value", val: fmtINR(forecast.next_value), hi: true },
                    { label: "Growth vs Mean",       val: `${forecast.growth_pct >= 0 ? "+" : ""}${forecast.growth_pct?.toFixed(1)}%` },
                    { label: "R² (Fit Quality)",     val: forecast.r_squared, hi: true },
                    { label: "Slope (β₁)",           val: forecast.slope },
                    { label: "Intercept (β₀)",       val: fmtINR(forecast.intercept), hi: true },
                    { label: "±CI Band",             val: fmtINR(forecast.confidence_band) },
                  ].map(({ label, val, hi }) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid rgba(99,130,255,.06)" }}>
                      <span style={{ fontSize: 11, color: "#8899bb" }}>{label}</span>
                      <span style={{ fontSize: 11, fontWeight: 600, color: hi ? "#67e8f9" : "#f0f4ff", fontFamily: "monospace" }}>{val}</span>
                    </div>
                  ))}
                  <div style={{ marginTop: 12, padding: "8px 10px", borderRadius: 7, background: forecast.r_squared > 0.6 ? "rgba(16,185,129,.08)" : "rgba(245,158,11,.08)", fontSize: 10, color: forecast.r_squared > 0.6 ? "#34d399" : "#fcd34d" }}>
                    {forecast.r_squared > 0.6 ? "✓ Good model fit — trend is reliable" : "⚠ Low R² — data may be non-linear or noisy"}
                  </div>
                </div>
              )}
            </ChartCard>
          </div>

          {/* ── AI INSIGHTS ── */}
          <SectionHead title="AI Business Insights" badge="Auto-Generated · Statistical Backing" />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 20 }}>
            {(insights || []).map((ins, i) => (
              <div key={i} style={{
                background: "linear-gradient(135deg,#121d35,#0d1628)",
                border: "1px solid rgba(99,130,255,.1)",
                borderRadius: 12, padding: 16,
                transition: "border-color .2s, transform .2s",
                animation: `fadeUp .4s ${i * 0.07}s both`,
              }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(99,102,241,.3)"; e.currentTarget.style.transform = "translateY(-2px)"; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(99,130,255,.1)"; e.currentTarget.style.transform = ""; }}
              >
                <div style={{ fontSize: 22, marginBottom: 8 }}>{ins.emoji}</div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "#4a5a7a", textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 6 }}>{ins.title}</div>
                <div style={{ fontSize: 11, color: "#8899bb", lineHeight: 1.6 }}>{ins.body}</div>
              </div>
            ))}
          </div>

          {/* ── FOOTER ── */}
          <div style={{ textAlign: "center", padding: "24px 20px", borderTop: "1px solid rgba(99,130,255,.1)", color: "#4a5a7a", fontSize: 11, letterSpacing: ".04em" }}>
            <div>Zepto Sales Intelligence Dashboard v2.0 &nbsp;·&nbsp; Developed by{" "}
              <span style={{ background: "linear-gradient(90deg,#a5b4fc,#67e8f9)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", fontWeight: 700 }}>Ayush Mishra</span>
            </div>
            <div style={{ marginTop: 6, fontSize: 9 }}>FastAPI · Pandas · SciPy · scikit-learn · React · Recharts</div>
          </div>
        </main>
      </div>
    </div>
  );
}

// ── Section Header helper ──────────────────────────────────────────────────────
function SectionHead({ title, badge }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, marginTop: 4 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "#8899bb", letterSpacing: ".08em", textTransform: "uppercase" }}>{title}</div>
      {badge && <div style={{ fontSize: 9, fontWeight: 600, padding: "3px 9px", borderRadius: 20, background: "rgba(6,182,212,.12)", color: "#06b6d4", border: "1px solid rgba(6,182,212,.25)" }}>{badge}</div>}
    </div>
  );
}
