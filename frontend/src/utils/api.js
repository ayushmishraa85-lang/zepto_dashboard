// src/utils/api.js
// Centralised API client. Change BASE_URL to point at your FastAPI server.

export const BASE_URL = "http://localhost:8000";

function buildQuery(filters = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") params.set(k, v);
  });
  return params.toString() ? `?${params.toString()}` : "";
}

async function get(path, filters = {}) {
  const res = await fetch(`${BASE_URL}${path}${buildQuery(filters)}`);
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  health:           ()          => get("/health"),
  meta:             ()          => get("/meta"),
  kpis:             (f)         => get("/kpis", f),
  cityRevenue:      (f)         => get("/city-revenue", f),
  categoryRevenue:  (f)         => get("/category-revenue", f),
  topProducts:      (f, n = 10) => get("/top-products", { ...f, n }),
  scatter:          (f)         => get("/scatter", f),
  heatmap:          (f)         => get("/heatmap", f),
  influencerImpact: (f)         => get("/influencer-impact", f),
  discountAnalysis: (f)         => get("/discount-analysis", f),
  priceRanges:      (f)         => get("/price-ranges", f),
  statistics:       (f)         => get("/statistics", f),
  correlation:      (f)         => get("/correlation", f),
  forecast:         (f)         => get("/forecast", f),
  insights:         (f)         => get("/insights", f),
  rawData:          (f, page)   => get("/data", { ...f, page }),

  uploadCSV: async (file) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE_URL}/upload`, { method: "POST", body: form });
    if (!res.ok) throw new Error("Upload failed");
    return res.json();
  },
};
