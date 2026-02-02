import React from "react";
import ResultCard from "./ResultCard";

function ResultGrid({ results }) {
  if (!results || results.length === 0) {
    return (
      <div className="mt-6 text-sm text-slate-500">
        No results yet. Try a query like <span className="font-medium">pizza</span>,{" "}
        <span className="font-medium">sushi</span>, or{" "}
        <span className="font-medium">vegan</span>.
      </div>
    );
  }

  return (
    <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {results.map((r, idx) => (
        <ResultCard key={r.business_id + "-" + idx} result={r} />
      ))}
    </div>
  );
}

export default ResultGrid;
