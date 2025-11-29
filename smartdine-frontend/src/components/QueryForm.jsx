// src/components/QueryForm.jsx

import React, { useState } from "react";

function QueryForm({ initialQuery = "pizza", initialK = 5, onSubmit, loading }) {
  const [queryText, setQueryText] = useState(initialQuery);
  const [k, setK] = useState(initialK);

  function handleSubmit(e) {
    e.preventDefault();
    if (!onSubmit) return;
    onSubmit({ queryText, k: Number(k) || 5 });
  }

  return (
    <form
      onSubmit={handleSubmit}
      style={{ display: "flex", gap: "1rem", alignItems: "flex-end", flexWrap: "wrap" }}
    >
      <div style={{ display: "flex", flexDirection: "column" }}>
        <label htmlFor="queryText">Query</label>
        <input
          id="queryText"
          type="text"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          style={{ padding: "0.5rem", minWidth: 240 }}
          placeholder="e.g., pizza, sushi, vegan..."
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", width: 80 }}>
        <label htmlFor="k">k</label>
        <input
          id="k"
          type="number"
          min={1}
          max={50}
          value={k}
          onChange={(e) => setK(e.target.value)}
          style={{ padding: "0.5rem" }}
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        style={{
          padding: "0.6rem 1.2rem",
          borderRadius: 6,
          border: "none",
          backgroundColor: "#2563eb",
          color: "white",
          cursor: "pointer",
        }}
      >
        {loading ? "Loading..." : "Get Recommendations"}
      </button>
    </form>
  );
}

export default QueryForm;
