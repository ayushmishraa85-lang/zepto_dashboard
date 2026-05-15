// src/components/KPICard.jsx
import React from "react";

const CONFIGS = {
  revenue:  { label: "Total Revenue",   icon: "💰", grad: "linear-gradient(135deg,#6366f1,#8b5cf6)", glow: "rgba(99,102,241,.2)",  clr: "#a5b4fc" },
  profit:   { label: "Total Profit",    icon: "📈", grad: "linear-gradient(135deg,#10b981,#06b6d4)", glow: "rgba(16,185,129,.2)",  clr: "#6ee7b7" },
  orders:   { label: "Total Orders",    icon: "🛒", grad: "linear-gradient(135deg,#f59e0b,#ef4444)", glow: "rgba(245,158,11,.2)",  clr: "#fcd34d" },
  margin:   { label: "Profit Margin",   icon: "%",  grad: "linear-gradient(135deg,#ef4444,#f59e0b)", glow: "rgba(239,68,68,.2)",   clr: "#fca5a5" },
  top_cat:  { label: "Top Category",    icon: "⭐", grad: "linear-gradient(135deg,#06b6d4,#3b82f6)", glow: "rgba(6,182,212,.2)",   clr: "#67e8f9" },
  top_city: { label: "Top Region",      icon: "📍", grad: "linear-gradient(135deg,#8b5cf6,#ec4899)", glow: "rgba(139,92,246,.2)",  clr: "#c4b5fd" },
};

function fmt(n) {
  if (typeof n === "string") return n;
  if (n >= 1e7)  return "₹" + (n / 1e7).toFixed(1)  + "Cr";
  if (n >= 1e5)  return "₹" + (n / 1e5).toFixed(1)  + "L";
  if (n >= 1e3)  return "₹" + (n / 1e3).toFixed(1)  + "K";
  if (n >= 1)    return n % 1 === 0 ? n.toLocaleString("en-IN") : n.toFixed(1);
  return String(n);
}

export default function KPICard({ type, value, sub, change, up = true }) {
  const cfg = CONFIGS[type] || CONFIGS.revenue;
  return (
    <div style={{
      background: "#0d1628", border: "1px solid rgba(99,130,255,.12)",
      borderRadius: 14, padding: "16px 14px", position: "relative",
      overflow: "hidden", cursor: "pointer",
      transition: "transform .25s, box-shadow .25s",
      animation: "fadeUp .4s both",
    }}
      onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = `0 12px 40px rgba(0,0,0,.3), 0 0 24px ${cfg.glow}`; }}
      onMouseLeave={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = ""; }}
    >
      {/* top gradient bar */}
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: cfg.grad }} />
      {/* background glow circle */}
      <div style={{ position: "absolute", top: "-30%", right: "-5%", width: 70, height: 70, borderRadius: "50%", background: cfg.glow, pointerEvents: "none" }} />

      <div style={{ width: 34, height: 34, borderRadius: 9, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 10, fontSize: 16, background: cfg.glow.replace(".2", ".12") }}>
        {cfg.icon}
      </div>
      <div style={{ fontSize: 9, color: "#4a5a7a", textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 3 }}>{cfg.label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, lineHeight: 1, marginBottom: 5, color: cfg.clr }}>{fmt(value)}</div>
      {change !== undefined && (
        <div style={{ display: "inline-flex", alignItems: "center", gap: 2, fontSize: 9, fontWeight: 600, borderRadius: 4, padding: "2px 6px", background: up ? "rgba(16,185,129,.1)" : "rgba(239,68,68,.1)", color: up ? "#34d399" : "#f87171" }}>
          {up ? "↑" : "↓"} {change}
        </div>
      )}
      {sub && <div style={{ fontSize: 9, color: "#4a5a7a", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}
