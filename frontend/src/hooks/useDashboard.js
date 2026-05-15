// src/hooks/useDashboard.js
import { useState, useEffect, useCallback, useRef } from "react";
import { api } from "../utils/api";

export function useDashboard() {
  const [filters, setFilters] = useState({
    city: "All", category: "All", influencer: "All", search: "",
  });
  const [data, setData] = useState({
    kpis: null, cityRevenue: [], categoryRevenue: [], topProducts: [],
    scatter: [], heatmap: null, influencer: [], discount: [],
    priceRanges: [], statistics: null, correlation: null,
    forecast: null, insights: [], meta: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [backendOk, setBackendOk] = useState(false);
  const autoRef = useRef(null);

  const fetchAll = useCallback(async (f = filters) => {
    setLoading(true);
    setError(null);
    try {
      const [
        kpis, cityRevenue, categoryRevenue, topProducts, scatter,
        heatmap, influencer, discount, priceRanges,
        statistics, correlation, forecast, insights, meta,
      ] = await Promise.all([
        api.kpis(f),
        api.cityRevenue(f),
        api.categoryRevenue(f),
        api.topProducts(f),
        api.scatter(f),
        api.heatmap(f),
        api.influencerImpact(f),
        api.discountAnalysis(f),
        api.priceRanges(f),
        api.statistics(f),
        api.correlation(f),
        api.forecast(f),
        api.insights(f),
        api.meta(),
      ]);
      setData({ kpis, cityRevenue, categoryRevenue, topProducts, scatter,
                heatmap, influencer, discount, priceRanges,
                statistics, correlation, forecast, insights, meta });
      setBackendOk(true);
    } catch (e) {
      setError(e.message);
      setBackendOk(false);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => { fetchAll(filters); }, [filters]);

  const updateFilter = (key, value) =>
    setFilters(prev => ({ ...prev, [key]: value }));

  const startAutoRefresh = (ms = 30000) => {
    stopAutoRefresh();
    autoRef.current = setInterval(() => fetchAll(filters), ms);
  };
  const stopAutoRefresh = () => {
    if (autoRef.current) { clearInterval(autoRef.current); autoRef.current = null; }
  };

  return { filters, updateFilter, data, loading, error, backendOk,
           refresh: () => fetchAll(filters), startAutoRefresh, stopAutoRefresh };
}
