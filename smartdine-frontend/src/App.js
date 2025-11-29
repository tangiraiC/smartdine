// src/App.jsx

import React, { useState } from "react";
import QueryForm from "./components/QueryForm";
import ResultGrid from "./components/ResultGrid";
import { fetchRecommendations } from "./api/recommendations";

function App() {
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleQuerySubmit({ queryText, k }) {
    setLoading(true);
    setStatus("Loading...");
    setResults([]);

    try {
      const recs = await fetchRecommendations(queryText, k);
      setResults(recs);
      setStatus(`Got ${recs.length} results for "${queryText}" (k=${k})`);
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${String(err.message || err)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 960, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1>SmartDine Recommendations</h1>

      <QueryForm onSubmit={handleQuerySubmit} loading={loading} />

      <div style={{ marginTop: "1rem", minHeight: "1.5rem" }}>{status}</div>

      <ResultGrid results={results} />
    </div>
  );
}

export default App;
