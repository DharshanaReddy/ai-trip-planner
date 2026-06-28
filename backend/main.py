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


_DEMO_ITINERARY = {
    "destination": "Kyoto, Japan",
    "summary": "Kyoto is Japan's cultural soul — a city of ancient temples, bamboo forests, and tea ceremonies that has preserved its imperial heritage across 1,200 years. This 5-day itinerary balances the iconic with the hidden, weaving through Higashiyama's stone-paved lanes and Arashiyama's misty groves.",
    "best_time_to_visit": "March–April (cherry blossom season) or October–November (autumn foliage). Avoid July–August heat and Golden Week crowds.",
    "getting_around": "IC card (Suica/ICOCA) covers buses and the subway. Day bus passes (¥700) are great for temple-heavy days. Rent a bicycle in Arashiyama — flat terrain and stunning scenery.",
    "days": [
        {
            "day": 1, "theme": "Higashiyama — The Old City at Dusk",
            "morning": {"activity": "Fushimi Inari Shrine", "description": "Hike the lower torii gates of Fushimi Inari before the crowds arrive. The vermillion tunnels wind 4km up the mountain, each gate donated by a Japanese business. The summit offers a panoramic city view.", "location": "Fushimi Inari Taisha", "cost": 0, "duration": "2.5 hours", "tips": "Go before 8am — the lower gates are tourist-free and the light is golden."},
            "afternoon": {"activity": "Kiyomizudera Temple + Higashiyama Walk", "description": "Kiyomizudera's famous wooden stage juts from the mountainside with views over the city canopy. Walk downhill through Sannenzaka and Ninenzaka — stone-paved lanes lined with Meiji-era machiya shops.", "location": "Kiyomizudera, Higashiyama", "cost": 500, "duration": "3 hours", "tips": "The shops sell Kyoto's best matcha soft serve (¥400). Skip the tourist ones — Kagizen Yoshifusa has the real thing."},
            "evening": {"activity": "Gion District Night Walk", "description": "Wander Hanamikoji Street at dusk when the ochaya (tea houses) light up paper lanterns and geiko move between appointments. The narrow Shirakawa canal reflects the willow trees in the lamplight.", "location": "Gion, Hanamikoji Street", "cost": 0, "duration": "1.5 hours", "tips": "Don't photograph geisha without permission — it's considered deeply rude. Observe from distance."},
            "meals": {
                "breakfast": {"name": "Nakamura-ro", "cuisine": "Japanese", "cost": 12, "why": "500-year-old tea house serving matcha and wagashi — the ideal Kyoto entry point"},
                "lunch": {"name": "Omen Noodles", "cuisine": "Japanese udon", "cost": 14, "why": "Queue-worthy udon with seasonal toppings, right off the Philosopher's Path"},
                "dinner": {"name": "Gion Kappa", "cuisine": "Kappo", "cost": 35, "why": "Counter dining, chef's seasonal menu, fraction of the price of comparable places"}
            },
            "transport": "Bus 5 from Kyoto Station to Fushimi Inari, then bus 202 to Higashiyama", "daily_total": 162, "insider_tip": "The stone path between Kiyomizudera and Maruyama Park is completely empty after 7pm — magical."
        },
        {
            "day": 2, "theme": "Arashiyama — Bamboo, Rivers, and Zen",
            "morning": {"activity": "Arashiyama Bamboo Grove + Okochi Sanso", "description": "The bamboo grove at dawn is genuinely transcendent — the 15-metre stalks creak in the wind and filter light into shifting green columns. Okochi Sanso villa next door (¥1,000) includes tea and a perfectly raked garden.", "location": "Arashiyama Bamboo Grove", "cost": 1000, "duration": "2 hours", "tips": "The grove is 15 minutes long. Walk through once for the experience, then double back — crowds flow one direction."},
            "afternoon": {"activity": "Tenryuji Temple + Boat on the Oi River", "description": "Tenryuji's Zen garden is a UNESCO World Heritage site — the borrowed scenery technique frames Arashiyama mountain through the pond. Rent a rowboat (¥1,500/30min) on the Oi River for an entirely different perspective.", "location": "Tenryuji, Arashiyama", "cost": 500, "duration": "3 hours", "tips": "The rear garden entrance (¥500) skips the temple fee and gives direct garden access."},
            "evening": {"activity": "Togetsukyo Bridge at Sunset", "description": "The 'Moon-Crossing Bridge' turns copper in the late afternoon light, with the Oi River and forested mountains behind it. One of Kyoto's most photographed compositions.", "location": "Togetsukyo Bridge", "cost": 0, "duration": "1 hour", "tips": "Cross to the south bank for the classic shot — shoot the bridge from slightly downriver on the western side."},
            "meals": {
                "breakfast": {"name": "% Arabica Arashiyama", "cuisine": "Coffee & pastries", "cost": 10, "why": "Minimalist café with a river view — best flat white in Kyoto"},
                "lunch": {"name": "Shigetsu at Tenryuji", "cuisine": "Shojin Ryori (temple cuisine)", "cost": 28, "why": "Traditional Buddhist vegetarian set served in the temple grounds — a rare authentic experience"},
                "dinner": {"name": "Arashiyama Yoshimura", "cuisine": "Soba", "cost": 22, "why": "House-made buckwheat noodles overlooking the Togetsukyo Bridge"}
            },
            "transport": "JR Sagano Line from Kyoto Station to Saga-Arashiyama (¥240). Rent bicycle at station (¥800/day).", "daily_total": 178, "insider_tip": "Walk 10 minutes north of the bamboo grove to Jojakko-ji — a mossy hillside temple that almost nobody visits."
        }
    ],
    "packing_essentials": [
        "Slip-on shoes — you remove them constantly at temples",
        "Portable WiFi or SIM (temples often have no signal)",
        "Cash — many small restaurants and shrines are cash-only",
        "Light layers — shrine gardens are shaded and cool even in summer"
    ],
    "local_phrases": [
        {"phrase": "Sumimasen", "pronunciation": "soo-mee-mah-sen", "meaning": "Excuse me / Sorry (use constantly)"},
        {"phrase": "Ikura desu ka", "pronunciation": "ee-koo-rah des-ka", "meaning": "How much is this?"},
        {"phrase": "Eigo wa hanasemasu ka", "pronunciation": "ay-go wa ha-na-se-mas-ka", "meaning": "Do you speak English?"}
    ],
    "emergency_info": {
        "emergency_number": "110 (police) / 119 (ambulance/fire)",
        "tourist_police": "Kyoto City Tourism Center: 075-343-0548",
        "nearest_hospital": "Kyoto University Hospital — multilingual staff available",
        "embassy_tip": "Register at your country's embassy before travel: mofa.go.jp lists all foreign embassies in Japan"
    },
    "budget_analysis": {
        "accommodation_estimate": 600,
        "food_estimate": 350,
        "activities_estimate": 180,
        "transport_estimate": 120,
        "miscellaneous": 150,
        "total_estimate": 1400,
        "budget_status": "under_budget",
        "variance_usd": 600,
        "variance_pct": 30.0,
        "daily_average": 280,
        "savings_tips": [
            "Eat lunch sets (teishoku) at restaurants — same food as dinner for 40% less",
            "Buy a bus day pass (¥700) on temple-heavy days — it pays for itself in 3 rides",
            "Stay in a machiya guesthouse rather than a hotel — cheaper and far more atmospheric"
        ],
        "splurge_picks": ["One multi-course kaiseki dinner (¥8,000-15,000) — Kyoto cuisine at its peak is a once-in-a-lifetime experience"],
        "budget_accommodations": ["Len Kyoto Kawaramachi (hostel with private rooms — ¥6,000/night)", "Airbnb machiya townhouses — often cheaper than hotels and a cultural experience in themselves"]
    },
    "research_insights": "Kyoto rewards slow travel. Most visitors rush between the top-5 sites and miss the subtle beauty of neighbourhood temples, riverside walks, and the rhythm of daily life in Nishiki Market. Building in unscheduled time — particularly early mornings — consistently produces the most memorable experiences."
}


@app.post("/generate-itinerary")
async def generate_itinerary(request: TripRequest, demo: bool = False):
    if demo:
        return {"itinerary": _DEMO_ITINERARY, "cached": False, "generation_time_ms": 0}

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
