import React, { useState } from "react";

function QueryForm({ initialQuery = "pizza", initialK = 5, onSubmit, loading }) {
  const [queryText, setQueryText] = useState(initialQuery);
  const [k, setK] = useState(initialK);
  // Slider value: 0 (Pure Discovery) -> 0.5 (Default) -> 1 (Search Focused) -> 2 (Strict Search)
  // We map this to backend weights: Text Weight = value * 20 (approx)
  const [modeValue, setModeValue] = useState(0.5);

  function handleSubmit(e) {
    e.preventDefault();
    if (!onSubmit) return;

    // Map slider to weights
    // Slider 0.0 -> Text 0.0, Joint 1.0 (Discovery)
    // Slider 0.5 -> Text 10.0, Joint 1.0 (Normal) - wait, let's make it smoother
    // Let's rely on a multiplier.
    // Base Text Weight = 20 * modeValue.
    // Joint Weight = 1.0

    // Examples:
    // 0.0 -> Text 0
    // 0.1 -> Text 2
    // 0.5 -> Text 10 (Default)
    // 1.0 -> Text 20

    const textWeight = modeValue * 20;

    onSubmit({
      queryText,
      k: Number(k) || 5,
      weights: {
        joint_score: 1.0,
        text_similarity: textWeight
      }
    });
  }

  const getModeLabel = (val) => {
    if (val < 0.2) return "Discovery (Model Only)";
    if (val < 0.4) return "Balanced";
    if (val < 0.7) return "Search (Default)";
    return "Strict Match";
  };

  return (
    <div className="relative z-10 rounded-2xl bg-white p-1 shadow-lg ring-1 ring-black/5">
      <form
        onSubmit={handleSubmit}
        className="flex flex-col gap-6 rounded-xl border border-slate-100 bg-white p-5 sm:flex-row sm:items-end sm:gap-4"
      >
        {/* Query Input */}
        <div className="flex-1 min-w-[200px]">
          <label htmlFor="queryText" className="mb-1.5 block text-xs font-bold uppercase tracking-wider text-slate-500">
            What are you craving?
          </label>
          <input
            id="queryText"
            type="text"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            className="w-full rounded-lg border-slate-200 bg-slate-50 px-4 py-3 text-base font-medium text-slate-800 placeholder-slate-400 focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-100 transition-all shadow-sm"
            placeholder="e.g., spicy ramen, vegan burger..."
          />
        </div>

        {/* Discovery Mode Slider */}
        <div className="flex w-full flex-col sm:w-64">
          <div className="mb-1.5 flex items-center justify-between">
            <label htmlFor="mode" className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Mode
            </label>
            <span className="text-[10px] font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">
              {getModeLabel(modeValue)}
            </span>
          </div>
          <input
            id="mode"
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={modeValue}
            onChange={(e) => setModeValue(Number(e.target.value))}
            className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-blue-600"
          />
          <div className="mt-1 flex justify-between text-[10px] font-medium text-slate-400">
            <span>Explore</span>
            <span>Search</span>
          </div>
        </div>

        {/* K Count - Hidden on small screens or minimized */}
        <div className="hidden sm:flex flex-col w-16">
          <label htmlFor="k" className="mb-1.5 block text-xs font-bold uppercase tracking-wider text-slate-500">
            Count
          </label>
          <input
            id="k"
            type="number"
            min={1}
            max={50}
            value={k}
            onChange={(e) => setK(e.target.value)}
            className="w-full rounded-lg border-slate-200 bg-slate-50 px-2 py-3 text-center text-sm font-semibold focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="h-[50px] rounded-lg bg-blue-600 px-8 text-sm font-bold text-white shadow-md transition-all hover:bg-blue-700 hover:shadow-lg hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-70 disabled:hover:transform-none sm:w-auto w-full"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Searching
            </span>
          ) : (
            "Go"
          )}
        </button>
      </form>
    </div>
  );
}

export default QueryForm;
