// src/App.js
import React, { useEffect, useState } from "react";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL || "http://localhost:8001";

function App() {
  const [health, setHealth] = useState(null);
  const [riskStats, setRiskStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError("");

        // 1) Health check
        const healthRes = await fetch(`${BACKEND_URL}/api/health`);
        if (!healthRes.ok) throw new Error("Health check failed");
        const healthJson = await healthRes.json();

        // 2) Risk distribution
        const riskRes = await fetch(
          `${BACKEND_URL}/api/analytics/risk-distribution`
        );
        if (!riskRes.ok) throw new Error("Risk analytics failed");
        const riskJson = await riskRes.json();

        setHealth(healthJson);
        setRiskStats(riskJson);
      } catch (err) {
        console.error(err);
        setError(
          err.message || "Something went wrong talking to the backend."
        );
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      {/* Top bar */}
      <header className="border-b border-slate-700 px-8 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            SecureScore AI
          </h1>
          <p className="text-sm text-slate-400">
            Fraud Detection &amp; Risk Scoring Dashboard
          </p>
        </div>
        <div className="text-sm text-slate-400">
          Backend:{" "}
          <span className={health?.status === "healthy" ? "text-emerald-400" : "text-red-400"}>
            {health ? health.status : "Checking..."}
          </span>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 px-8 py-6">
        {loading && (
          <div className="flex items-center justify-center h-full text-slate-300">
            Loading data from the model…
          </div>
        )}

        {!loading && error && (
          <div className="max-w-xl mx-auto bg-red-900/30 border border-red-500/60 rounded-xl px-4 py-3 text-sm">
            <div className="font-medium text-red-300 mb-1">Error</div>
            <p className="text-red-200">{error}</p>
            <p className="mt-2 text-xs text-red-300/80">
              Make sure the backend is running on {BACKEND_URL} and that
              <code className="ml-1">/api/health</code> and
              <code className="ml-1">/api/analytics/risk-distribution</code> are available.
            </p>
          </div>
        )}

        {!loading && !error && (
          <div className="grid gap-6 md:grid-cols-3">
            {/* KPI cards */}
            <div className="col-span-1 bg-slate-800/60 border border-slate-700 rounded-2xl p-5 shadow-lg shadow-black/30">
              <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">
                Total Predictions
              </div>
              <div className="text-2xl font-semibold">
                {riskStats?.total_predictions ?? 0}
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Number of transactions scored by the model.
              </div>
            </div>

            <div className="col-span-1 bg-slate-800/60 border border-slate-700 rounded-2xl p-5 shadow-lg shadow-black/30">
              <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">
                High Risk Transactions
              </div>
              <div className="text-2xl font-semibold text-rose-400">
                {riskStats?.high_risk_count ?? 0}
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Flagged as potentially fraudulent.
              </div>
            </div>

            <div className="col-span-1 bg-slate-800/60 border border-slate-700 rounded-2xl p-5 shadow-lg shadow-black/30">
              <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">
                Avg Risk Score
              </div>
              <div className="text-2xl font-semibold text-amber-300">
                {riskStats?.average_risk_score?.toFixed
                  ? riskStats.average_risk_score.toFixed(1)
                  : riskStats?.average_risk_score ?? 0}
              </div>
              <div className="text-xs text-slate-400 mt-2">
                0–100 scale based on fraud probability.
              </div>
            </div>

            {/* Recent predictions / JSON view */}
            <div className="md:col-span-3 bg-slate-800/60 border border-slate-700 rounded-2xl p-5 shadow-lg shadow-black/30">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-slate-100">
                  Raw Analytics Response
                </h2>
              </div>
              <pre className="text-xs bg-slate-900/70 rounded-xl p-3 overflow-x-auto text-slate-200">
                {JSON.stringify(riskStats, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </main>

      <footer className="px-8 py-4 text-xs text-slate-500 border-t border-slate-800">
        Connected to <code className="mx-1">{BACKEND_URL}</code> •
        &nbsp;Built for fintech / fraud detection interviews.
      </footer>
    </div>
  );
}

export default App;
