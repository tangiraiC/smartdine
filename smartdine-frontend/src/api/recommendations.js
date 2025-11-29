// src/api/recommendations.js

const API_URL = "http://127.0.0.1:8000/api/recommendations/";

export async function fetchRecommendations(queryText, k) {
  const resp = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query_text: queryText, k: Number(k) }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${errText}`);
  }

  const data = await resp.json();
  return data.results || [];
}
