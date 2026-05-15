// src/components/Sidebar.jsx
import React, { useState } from "react";

const NAV = [
  { icon: "▦", label: "Overview" },
  { icon: "📊", label: "Revenue" },
  { icon: "🗺", label: "Regional" },
  { icon: "📦", label: "Products" },
  { icon: "🔬", label: "Statistics" },
  { icon: "🔮", label: "Forecast" },
];

export default function Sidebar({ filters, onFilter, meta, onUpload, autoRefresh, onAutoRefresh }) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeNav, setActiveNav] = useState(0);

  const sel = (key, val) => onFilter(key, val);

  return (
    <aside style={{
      width: collapsed ? 56 : 210,
      flexShrink: 0,
      background: "rgba(11,18,32,.97)",
      borderRight: "1px solid rgba(99,130,255,.1)",
      padding: collapsed ? "14px 8px" : "16px 12px",
      overflowY: "auto",
      maxHeight: "calc(100vh - 54px)",
      position: "sticky", top: 54,
      transition: "width .3s",
    }}>

      {/* Upload */}
      {!collapsed && (
        <div style={{ marginBottom: 18 }}>
          <label style={{
            display: "flex", flexDirection: "column", alignItems: "center",
            gap: 4, border: "1.5px dashed rgba(99,102,241,.35)",
            borderRadius: 9, padding: "12px 8px", cursor: "pointer",
            color: "#6366f1", fontSize: 11, textAlign: "center",
            transition: "all .2s",
          }}
            onMouseEnter={e => { e.currentTarget.style.background = "rgba(99,102,241,.07)"; e.currentTarget.style.borderColor = "#6366f1"; }}
            onMouseLeave={e => { e.currentTarget.style.background = ""; e.currentTarget.style.borderColor = "rgba(99,102,241,.35)"; }}
          >
            <span style={{ fontSize: 20 }}>⬆</span>
            Upload CSV
            <input type="file" accept=".csv" style={{ display: "none" }} onChange={e => onUpload(e.target.files[0])} />
          </label>
        </div>
      )}

      {/* Nav */}
      <div style={{ marginBottom: 18 }}>
        {!collapsed && <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: ".1em", textTransform: "uppercase", color: "#4a5a7a", marginBottom: 7 }}>Navigation</div>}
        {NAV.map((n, i) => (
          <div key={i} onClick={() => setActiveNav(i)} style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: collapsed ? "8px 6px" : "7px 9px",
            borderRadius: 8, cursor: "pointer",
            color: activeNav === i ? "#a5b4fc" : "#8899bb",
            fontSize: 11, fontWeight: 500,
            background: activeNav === i ? "rgba(99,102,241,.18)" : "",
            border: activeNav === i ? "1px solid rgba(99,102,241,.28)" : "1px solid transparent",
            marginBottom: 2, transition: "all .2s",
          }}
            onMouseEnter={e => { if (activeNav !== i) e.currentTarget.style.background = "rgba(99,130,255,.08)"; }}
            onMouseLeave={e => { if (activeNav !== i) e.currentTarget.style.background = ""; }}
          >
            <span>{n.icon}</span>
            {!collapsed && <span>{n.label}</span>}
          </div>
        ))}
      </div>

      {/* Filters */}
      {!collapsed && (
        <div style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: ".1em", textTransform: "uppercase", color: "#4a5a7a", marginBottom: 8 }}>Filters</div>

          {[
            { key: "city",       label: "City / Region", options: ["All", ...(meta?.cities || [])] },
            { key: "category",   label: "Category",      options: ["All", ...(meta?.categories || [])] },
            { key: "influencer", label: "Influencer",    options: ["All", "Yes", "No"] },
          ].map(({ key, label, options }) => (
            <div key={key} style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 10, color: "#4a5a7a", marginBottom: 4 }}>{label}</div>
              <select value={filters[key]} onChange={e => sel(key, e.target.value)} style={{
                width: "100%", background: "#121d35", border: "1px solid rgba(99,130,255,.1)",
                borderRadius: 6, padding: "6px 8px", color: "#f0f4ff",
                fontFamily: "inherit", fontSize: 11, outline: "none",
              }}>
                {options.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
          ))}

          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 10, color: "#4a5a7a", marginBottom: 4 }}>Search Product</div>
            <input
              type="text" placeholder="e.g. Maggi..."
              value={filters.search} onChange={e => sel("search", e.target.value)}
              style={{
                width: "100%", background: "#121d35",
                border: "1px solid rgba(99,130,255,.1)",
                borderRadius: 6, padding: "6px 8px", color: "#f0f4ff",
                fontFamily: "inherit", fontSize: 11, outline: "none",
              }}
            />
          </div>
        </div>
      )}

      {/* Settings */}
      {!collapsed && (
        <div style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: ".1em", textTransform: "uppercase", color: "#4a5a7a", marginBottom: 8 }}>Settings</div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "5px 0" }}>
            <span style={{ fontSize: 11, color: "#8899bb" }}>Auto Refresh</span>
            <div onClick={onAutoRefresh} style={{
              width: 34, height: 19, background: autoRefresh ? "#6366f1" : "#1e2d4a",
              borderRadius: 10, cursor: "pointer", position: "relative",
              border: "1px solid rgba(99,130,255,.1)", transition: "background .3s",
            }}>
              <div style={{
                width: 13, height: 13, background: "#fff", borderRadius: "50%",
                position: "absolute", top: 2, left: autoRefresh ? 18 : 2, transition: "left .3s",
              }} />
            </div>
          </div>
        </div>
      )}

      {/* Collapse */}
      <button onClick={() => setCollapsed(c => !c)} style={{
        width: "100%", background: "#121d35",
        border: "1px solid rgba(99,130,255,.1)", borderRadius: 7,
        padding: 8, color: "#4a5a7a", cursor: "pointer",
        fontSize: 11, display: "flex", alignItems: "center",
        justifyContent: "center", gap: 6, fontFamily: "inherit",
      }}>
        {collapsed ? "▶" : "◀ Collapse"}
      </button>
    </aside>
  );
}
