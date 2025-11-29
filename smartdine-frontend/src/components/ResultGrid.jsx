// src/components/ResultGrid.jsx

import React from "react";
import ResultCard from "./ResultCard";

function ResultGrid({ results }) {
  if (!results || results.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        marginTop: "1.5rem",
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "1rem",
      }}
    >
      {results.map((r, idx) => (
        <ResultCard key={r.business_id + "-" + idx} result={r} />
      ))}
    </div>
  );
}

export default ResultGrid;
