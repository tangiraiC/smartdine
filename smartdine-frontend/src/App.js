import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import BusinessDetail from "./components/BusinessDetail";
import QueryForm from "./components/QueryForm";
import ResultGrid from "./components/ResultGrid";
import { fetchRecommendations } from "./api/recommendations";

function Home() {
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  // Key to force re-render of QueryForm to reset its internal state if needed
  const [formKey, setFormKey] = useState(0);

  async function handleQuerySubmit({ queryText, k, weights }) {
    setLoading(true);
    setStatus("Loading...");
    setResults([]);

    try {
      // Pass weights if available
      const recs = await fetchRecommendations(queryText, k, weights);
      setResults(recs);
      setStatus(`Got ${recs.length} results for "${queryText}" (k=${k})`);
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${String(err.message || err)}`);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setResults([]);
    setStatus("");
    setFormKey(prev => prev + 1); // Optional: if we want to clear the input text too
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      {/* Hero Header */}
      <header className="relative overflow-hidden bg-gradient-to-r from-blue-700 via-indigo-700 to-violet-700 pb-24 pt-10 shadow-xl">
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-150 contrast-150 mix-blend-overlay"></div>
        <div className="relative mx-auto max-w-5xl px-6 text-center text-white">
          <button
            onClick={handleReset}
            className="text-4xl font-extrabold tracking-tight sm:text-5xl drop-shadow-sm hover:opacity-90 transition-opacity focus:outline-none"
          >
            SmartDine
          </button>
          <p className="mx-auto mt-4 max-w-xl text-lg text-blue-100/90 font-medium">
            Next-Gen Restaurant Discovery
          </p>
        </div>
      </header>

      {/* Main Content Area - overlaps header */}
      <main className="relative mx-auto -mt-16 max-w-5xl px-4 pb-12 sm:px-6">
        <div className="space-y-6">

          {/* Multimodal ML Banner */}
          <div className="rounded-xl border border-blue-200 bg-blue-50/90 p-4 text-blue-900 backdrop-blur-sm shadow-sm relative overflow-hidden">
            <div className="flex items-start gap-3 relative z-10">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <div>
                <h3 className="text-sm font-bold uppercase tracking-wide text-blue-700">Experimental AI Project</h3>
                <p className="mt-1 text-sm leading-relaxed text-blue-800">
                  This system uses a <strong>Multimodal Machine Learning</strong> approach to power recommendations.
                  It fuses <strong>User Embeddings</strong> (Collaborative Filtering) with <strong>Text & Image Embeddings</strong> (Content-Based)
                  in a shared vector space to understand both <em>what you like</em> and <em>what you search for</em>.
                </p>
              </div>
            </div>
            {/* Decorative background element */}
            <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-blue-500/10 blur-xl"></div>
          </div>

          <QueryForm key={formKey} onSubmit={handleQuerySubmit} loading={loading} />

          {/* Status Message */}
          <div className="min-h-[1.5rem] px-2 text-sm font-medium text-slate-500">
            {status}
          </div>

          <ResultGrid results={results} />
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/business/:id" element={<BusinessDetail />} />
      </Routes>
    </Router>
  );
}

export default App;
