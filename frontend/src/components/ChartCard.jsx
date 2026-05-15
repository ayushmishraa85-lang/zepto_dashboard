// src/components/ChartCard.jsx
import React from "react";

export default function ChartCard({ title, subtitle, children, span = 1, style = {} }) {
  return (
    <div style={{
      background: "#0d1628",
      border: "1px solid rgba(99,130,255,.1)",
      borderRadius: 14,
      padding: 18,
      gridColumn: span > 1 ? `span ${span}` : undefined,
      transition: "border-color .2s",
      ...style,
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = "rgba(99,102,241,.25)"}
      onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(99,130,255,.1)"}
    >
      <div style={{ fontSize: 13, fontWeight: 600, color: "#f0f4ff", marginBottom: 2 }}>{title}</div>
      {subtitle && <div style={{ fontSize: 10, color: "#4a5a7a", marginBottom: 14 }}>{subtitle}</div>}
      {children}
    </div>
  );
}
