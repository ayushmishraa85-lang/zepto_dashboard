// src/components/StatisticsPanel.jsx
import React from "react";
import ChartCard from "./ChartCard";

function StatRow({ label, value, highlight }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "8px 0", borderBottom: "1px solid rgba(99,130,255,.07)",
    }}>
      <span style={{ fontSize: 11, color: "#8899bb" }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 600, color: highlight ? "#67e8f9" : "#f0f4ff", fontFamily: "monospace" }}>{value}</span>
    </div>
  );
}

export default function StatisticsPanel({ stats, correlation }) {
  if (!stats || !correlation) return null;

  const corrCols = correlation.columns || [];
  const corrMatrix = correlation.matrix || {};

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginBottom: 14 }}>

      {/* Descriptive Stats */}
      <ChartCard title="📊 Descriptive Statistics" subtitle="Revenue distribution metrics">
        <StatRow label="Mean Revenue"    value={`₹${stats.mean?.toLocaleString("en-IN") || 0}`}   highlight />
        <StatRow label="Median Revenue"  value={`₹${stats.median?.toLocaleString("en-IN") || 0}`} />
        <StatRow label="Std Deviation"   value={`₹${stats.std?.toLocaleString("en-IN") || 0}`}    highlight />
        <StatRow label="Variance"        value={`₹${stats.variance?.toLocaleString("en-IN") || 0}`} />
        <StatRow label="Skewness"        value={stats.skewness} highlight />
        <StatRow label="Kurtosis"        value={stats.kurtosis} />
        <StatRow label="Normality p-val" value={stats.p_value_normality} highlight />
        <div style={{ marginTop: 10, fontSize: 10, padding: "6px 10px", borderRadius: 6, background: stats.is_normal ? "rgba(16,185,129,.1)" : "rgba(245,158,11,.1)", color: stats.is_normal ? "#34d399" : "#fcd34d" }}>
          {stats.is_normal ? "✓ Revenue is normally distributed" : "⚠ Revenue is not normally distributed (skewed)"}
        </div>
      </ChartCard>

      {/* Outliers */}
      <ChartCard title="⚠ Outlier Detection" subtitle={`${stats.outlier_count || 0} outliers (|Z| > 2)`}>
        <div style={{ fontSize: 10, color: "#4a5a7a", marginBottom: 8 }}>Products with anomalous revenue (Z-score {">"} 2σ)</div>
        <div style={{ maxHeight: 220, overflowY: "auto" }}>
          {(stats.outliers || []).map((o, i) => (
            <div key={i} style={{
              background: "rgba(239,68,68,.06)", border: "1px solid rgba(239,68,68,.15)",
              borderRadius: 7, padding: "8px 10px", marginBottom: 6,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#f0f4ff", marginBottom: 2 }}>{o["Product Name"]}</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <span style={{ fontSize: 9, color: "#8899bb" }}>{o["City"]} · {o["Category"]}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                <span style={{ fontSize: 10, color: "#67e8f9" }}>₹{o["Total Revenue"]?.toLocaleString("en-IN")}</span>
                <span style={{ fontSize: 10, color: "#fca5a5", fontFamily: "monospace" }}>Z = {o["z_score"]}</span>
              </div>
            </div>
          ))}
          {(!stats.outliers || stats.outliers.length === 0) && (
            <div style={{ fontSize: 11, color: "#4a5a7a", textAlign: "center", paddingTop: 20 }}>No significant outliers detected</div>
          )}
        </div>
      </ChartCard>

      {/* Correlation Key Metrics */}
      <ChartCard title="🔗 Correlation Insights" subtitle="Pearson r between key variables">
        <StatRow label="Discount → Orders (r)"    value={correlation.discount_orders_r} highlight />
        <StatRow label="Discount → Orders (p)"    value={correlation.discount_orders_p} />
        <StatRow label="Revenue → Profit (r)"     value={correlation.revenue_profit_r}  highlight />
        <StatRow label="Revenue → Profit (p)"     value={correlation.revenue_profit_p}  />
        <div style={{ marginTop: 12, fontSize: 10, color: "#4a5a7a", marginBottom: 6 }}>Interpretation</div>
        {[
          { pair: "Discount ↔ Orders", r: correlation.discount_orders_r, p: correlation.discount_orders_p },
          { pair: "Revenue ↔ Profit",  r: correlation.revenue_profit_r,  p: correlation.revenue_profit_p  },
        ].map(({ pair, r, p }) => {
          const sig   = p < 0.05;
          const strong = Math.abs(r) > 0.5;
          const dir   = r > 0 ? "positive" : "negative";
          return (
            <div key={pair} style={{ background: "rgba(99,102,241,.08)", borderRadius: 6, padding: "7px 10px", marginBottom: 6 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "#a5b4fc", marginBottom: 2 }}>{pair}</div>
              <div style={{ fontSize: 10, color: "#8899bb" }}>
                {strong ? "Strong" : "Weak"} {dir} correlation
                {sig ? " — statistically significant" : " — not significant"}
              </div>
            </div>
          );
        })}
      </ChartCard>
    </div>
  );
}
