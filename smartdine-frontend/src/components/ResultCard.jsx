// src/components/ResultCard.jsx

import React from "react";

function ResultCard({ result }) {
  const {
    business_id,
    score,
    representative_image_url,
    name,
    avg_rating,
    num_reviews,
  } = result;

  // Fallback image if none exists
  const imgSrc = representative_image_url
    ? representative_image_url
    : "https://via.placeholder.com/300x200?text=No+Image";

  return (
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: "0.5rem",
        boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
      }}
    >
      <img
        src={imgSrc}
        alt={business_id}
        style={{
          width: "100%",
          borderRadius: 6,
          marginBottom: "0.5rem",
          objectFit: "cover",
          height: 160,
        }}
      />

      <div style={{ fontWeight: 600 }}>Score: {score.toFixed(3)}</div>

      <div
        style={{
          fontSize: "0.8rem",
          color: "#4b5563",
          wordBreak: "break-all",
          marginTop: "0.25rem",
        }}
      >
        ID: {business_id}
      </div>
    </div>
  );
}

export default ResultCard;
