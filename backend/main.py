"""
AI Trip Planner — FastAPI Backend
Multi-agent itinerary generation with LangGraph, RAG (ChromaDB), and LangSmith observability.
"""

import os
import time
import hashlib
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import get_settings
from .schemas import TripRequest, TripResponse
from .agents.graph import run_trip_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Configure LangSmith tracing via environment
os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up ChromaDB on startup
    try:
        from .rag.vectorstore import get_collection
        get_collection()
        logger.info("ChromaDB collection ready")
    except Exception as e:
        logger.warning(f"ChromaDB warmup failed: {e}")
    yield


app = FastAPI(
    title="AI Trip Planner API",
    description=(
        "Multi-agent travel itinerary generation powered by LangGraph, "
        "RAG with ChromaDB, and LangSmith observability."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache (swap for Redis in production)
_cache: dict[str, dict] = {}


def _cache_key(req: TripRequest) -> str:
    payload = f"{req.destination}:{req.budget}:{req.duration}:{sorted(req.interests)}:{req.travel_style}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@app.get("/")
def root():
    return {
        "service": "AI Trip Planner API",
        "version": "2.1.0",
        "agents": ["ResearchAgent", "ItineraryAgent", "BudgetAgent", "ReviewAgent"],
        "features": ["LangGraph multi-agent", "ChromaDB RAG", "LangSmith tracing"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.1.0", "environment": settings.app_env}


POPULAR_DESTINATIONS = [
    {"destination": "Kyoto, Japan", "region": "Asia", "avg_budget_7d": 1800, "best_month": "April", "tags": ["culture", "temples", "food"]},
    {"destination": "Barcelona, Spain", "region": "Europe", "avg_budget_7d": 1400, "best_month": "May", "tags": ["architecture", "beach", "nightlife"]},
    {"destination": "Bali, Indonesia", "region": "Asia", "avg_budget_7d": 900, "best_month": "July", "tags": ["nature", "wellness", "adventure"]},
    {"destination": "New York City, USA", "region": "Americas", "avg_budget_7d": 2200, "best_month": "October", "tags": ["culture", "food", "art"]},
    {"destination": "Cape Town, South Africa", "region": "Africa", "avg_budget_7d": 1300, "best_month": "March", "tags": ["nature", "adventure", "wildlife"]},
    {"destination": "Marrakech, Morocco", "region": "Africa", "avg_budget_7d": 800, "best_month": "April", "tags": ["culture", "food", "markets"]},
    {"destination": "Queenstown, New Zealand", "region": "Oceania", "avg_budget_7d": 1900, "best_month": "February", "tags": ["adventure", "nature", "skiing"]},
    {"destination": "Lisbon, Portugal", "region": "Europe", "avg_budget_7d": 1100, "best_month": "June", "tags": ["history", "food", "music"]},
    {"destination": "Medellín, Colombia", "region": "Americas", "avg_budget_7d": 700, "best_month": "February", "tags": ["culture", "food", "nightlife"]},
    {"destination": "Jordan (Petra + Wadi Rum)", "region": "Middle East", "avg_budget_7d": 1500, "best_month": "March", "tags": ["history", "adventure", "desert"]},
]


@app.get("/destinations/popular")
def popular_destinations(region: str | None = None, max_budget: float | None = None):
    """Return curated popular destinations, optionally filtered by region or budget."""
    results = POPULAR_DESTINATIONS
    if region:
        results = [d for d in results if d["region"].lower() == region.lower()]
    if max_budget:
        results = [d for d in results if d["avg_budget_7d"] <= max_budget]
    return {"count": len(results), "destinations": results}


@app.post("/generate-itinerary")
async def generate_itinerary(request: TripRequest):
    cache_key = _cache_key(request)

    if cache_key in _cache:
        logger.info(f"Cache hit for {request.destination}")
        return {**_cache[cache_key], "cached": True}

    start = time.time()
    try:
        itinerary = run_trip_pipeline(
            destination=request.destination,
            budget=request.budget,
            duration=request.duration,
            interests=request.interests,
            travel_style=request.travel_style,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trip planning pipeline error: {str(e)}")

    elapsed_ms = int((time.time() - start) * 1000)
    response = {
        "itinerary": itinerary,
        "cached": False,
        "generation_time_ms": elapsed_ms,
    }
    _cache[cache_key] = response
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
