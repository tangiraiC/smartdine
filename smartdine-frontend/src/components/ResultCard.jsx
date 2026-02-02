import React from "react";
import { Link } from "react-router-dom";

function renderStars(avgRating) {
  if (avgRating == null) return null;
  const fullStars = Math.floor(avgRating);
  const hasHalf = avgRating - fullStars >= 0.5;
  const stars = [];

  for (let i = 0; i < 5; i++) {
    if (i < fullStars) {
      stars.push(
        <span key={i} className="text-amber-400">
          ★
        </span>
      );
    } else if (i === fullStars && hasHalf) {
      stars.push(
        <span key={i} className="text-amber-400">
          ☆
        </span>
      );
    } else {
      stars.push(
        <span key={i} className="text-slate-300">
          ★
        </span>
      );
    }
  }

  return <div className="text-xs">{stars}</div>;
}

function ResultCard({ result }) {
  const {
    business_id,
    score,
    representative_image_url,
    name,
    avg_rating,
    num_reviews,
    components,
  } = result;

  let imgSrc = representative_image_url;

  // If it's a relative path starting with /, prepend the backend origin
  if (imgSrc && imgSrc.startsWith("/")) {
    imgSrc = `http://127.0.0.1:8000${imgSrc}`;
  }

  imgSrc = imgSrc || "https://via.placeholder.com/400x260?text=SmartDine";

  // Calculate percentages for the mini-bars (clamped 0-100)
  // Max expected joint ~6.0, Max expected text ~1.0 (raw) * 20 (weight) = 20.0
  // Adjust scaling for visualization
  const jointScore = components?.joint_score || 0;
  const textScore = components?.text_similarity || 0;

  // Weights (approx) used for display scaling
  // Text can be high (e.g. 10 * 0.8 = 8.0). Joint usually 5.0.
  // Let's just scale relative to total score for a "contribution" view? 
  // Or absolute bars. Let's do absolute bars with arbitrary max reference.
  const maxRef = 15;
  const jointPct = Math.min((jointScore / 8) * 100, 100); // assume max joint ~8
  const textPct = Math.min((textScore / 15) * 100, 100);  // assume max text ~15

  return (
    <Link to={`/business/${business_id}`} className="block">
      <div className="group flex flex-col rounded-2xl border border-slate-200 bg-white shadow-sm transition-all hover:shadow-md hover:-translate-y-1 overflow-hidden h-full">
        <div className="relative h-48 bg-slate-100 overflow-hidden">
          <img
            src={imgSrc}
            alt={business_id}
            className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
          />
          <div className="absolute top-2 right-2 rounded-lg bg-black/60 px-2 py-1 text-xs font-bold text-white backdrop-blur-md">
            {score.toFixed(1)}
          </div>
        </div>

        <div className="flex flex-col gap-2 p-4">
          <div>
            <h3 className="text-lg font-bold text-slate-800 leading-tight line-clamp-1 group-hover:text-blue-600 transition-colors">
              {name || "Unknown Restaurant"}
            </h3>
            <div className="flex items-center gap-2 mt-1">
              {renderStars(avg_rating)}
              <span className="text-xs text-slate-500">
                ({num_reviews || 0} reviews)
              </span>
            </div>
          </div>

          {/* Insights Section */}
          <div className="mt-2 rounded-lg bg-slate-50 p-2 border border-slate-100">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5">
              Model Insights
            </div>

            <div className="space-y-1.5">
              {/* Semantic Match */}
              <div className="flex items-center gap-2">
                <span className="w-16 text-[10px] font-medium text-slate-600">Relevance</span>
                <div className="flex-1 h-1.5 rounded-full bg-slate-200 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-emerald-500"
                    style={{ width: `${textPct}%` }}
                  />
                </div>
                <span className="w-6 text-[10px] text-right text-slate-500">{textScore.toFixed(1)}</span>
              </div>

              {/* Popularity/Quality (Joint) */}
              <div className="flex items-center gap-2">
                <span className="w-16 text-[10px] font-medium text-slate-600">Quality</span>
                <div className="flex-1 h-1.5 rounded-full bg-slate-200 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-indigo-500"
                    style={{ width: `${jointPct}%` }}
                  />
                </div>
                <span className="w-6 text-[10px] text-right text-slate-500">{jointScore.toFixed(1)}</span>
              </div>
            </div>
          </div>

          <div className="mt-1 text-[10px] text-slate-400 font-mono truncate">
            ID: {business_id}
          </div>
        </div>
      </div>
    </Link>
  );
}

export default ResultCard;
