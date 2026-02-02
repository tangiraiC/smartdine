import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";

function BusinessDetail() {
    const { id } = useParams();
    const [business, setBusiness] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function fetchDetails() {
            try {
                setLoading(true);
                // Ensure to handle the path correctly relative to API root
                const res = await fetch(`http://127.0.0.1:8000/api/business/${id}/`);
                if (!res.ok) {
                    throw new Error(`Error ${res.status}: ${res.statusText}`);
                }
                const data = await res.json();
                setBusiness(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        }
        fetchDetails();
    }, [id]);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center text-slate-500">
                Loading details...
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center p-4">
                <div className="text-red-500 mb-4">Failed to load business details: {error}</div>
                <Link to="/" className="text-blue-600 hover:underline">
                    &larr; Back to Search
                </Link>
            </div>
        );
    }

    if (!business) return null;

    let imgSrc = business.image_url;
    if (imgSrc && imgSrc.startsWith("/")) {
        imgSrc = `http://127.0.0.1:8000${imgSrc}`;
    }
    imgSrc = imgSrc || "https://via.placeholder.com/800x400?text=No+Image";

    return (
        <div className="min-h-screen bg-white">
            {/* Hero Image */}
            <div className="relative h-64 md:h-96 w-full bg-slate-100">
                <img
                    src={imgSrc}
                    alt={business.name}
                    className="h-full w-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />

                <div className="absolute bottom-0 left-0 w-full p-6 md:p-10 text-white">
                    <div className="max-w-5xl mx-auto">
                        <h1 className="text-3xl md:text-5xl font-bold drop-shadow-sm mb-2">
                            {business.name}
                        </h1>
                        <div className="flex items-center gap-4 text-sm md:text-base font-medium text-white/90">
                            <span className="bg-emerald-500 text-white px-2 py-0.5 rounded text-xs md:text-sm">
                                {business.avg_rating?.toFixed(1) || "N/A"} ★
                            </span>
                            <span>{business.num_reviews} reviews</span>
                            {business.city && <span>• {business.city}, {business.state}</span>}
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-6 py-8">
                <Link
                    to="/"
                    className="inline-flex items-center text-sm font-semibold text-slate-500 hover:text-blue-600 mb-6 transition-colors"
                >
                    &larr; Back to Recommendations
                </Link>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <div className="md:col-span-2 space-y-8">
                        <section>
                            <h2 className="text-xl font-bold text-slate-900 mb-4">About</h2>
                            <p className="text-slate-600 leading-relaxed">
                                {/* Placeholder for description if available in metadata */}
                                {business.name} is located in {business.city}.
                                {business.sales_volume && ` It has an estimated sales volume of ${business.sales_volume}.`}
                            </p>
                        </section>

                        {/* Additional sections can be added here */}
                    </div>

                    <div className="space-y-6">
                        <div className="rounded-xl border border-slate-200 p-6 bg-slate-50">
                            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-4">Location</h3>
                            <div className="text-slate-800 font-medium">
                                {business.address || "Address not available"}
                                <br />
                                {business.city}, {business.state}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default BusinessDetail;
