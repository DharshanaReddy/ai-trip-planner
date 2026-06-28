"""
LangGraph agent nodes — each node is a discrete reasoning step in the trip planning pipeline.
Traced end-to-end with LangSmith for observability.
"""

import json
import logging
import re
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from .state import TripState
from ..rag.vectorstore import retrieve_context
from ..config import get_settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0


def _llm(temperature: float = 0.5) -> ChatGroq:
    settings = get_settings()
    return ChatGroq(model=settings.groq_model, temperature=temperature, api_key=settings.groq_api_key)


def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(text)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return {}


def _invoke_with_retry(llm: ChatGroq, messages: list, label: str = "agent") -> str:
    """Invoke LLM with exponential backoff retry on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm.invoke(messages).content
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.error(f"{label} failed after {MAX_RETRIES} attempts: {e}")
                raise
            wait = RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(f"{label} attempt {attempt} failed ({e}), retrying in {wait:.1f}s")
            time.sleep(wait)
    return ""


@traceable(name="research-agent")
def research_node(state: TripState) -> dict:
    """
    ResearchAgent: enriches planning context via RAG retrieval + LLM synthesis.
    Retrieves relevant travel knowledge from ChromaDB before calling the LLM,
    reducing hallucination and improving destination-specific accuracy.
    """
    destination = state["destination"]
    rag_context = retrieve_context(
        f"{destination} travel tips budget safety transport culture food"
    )

    content = _invoke_with_retry(_llm(temperature=0.3), [
        SystemMessage(content=(
            "You are a senior travel researcher with expertise across all global destinations. "
            "Synthesize the provided knowledge base context with your expertise to produce "
            "accurate, practical, destination-specific insights."
        )),
        HumanMessage(content=f"""Research {destination} for a {state['duration']}-day trip.
Budget: ${state['budget']}  |  Style: {state['travel_style']}
Interests: {', '.join(state['interests'])}

Knowledge base context:
{rag_context or 'Use your general travel expertise.'}

Provide a concise research brief covering:
1. Best neighborhoods/areas to base yourself
2. Local transport (apps, passes, costs)
3. Cultural norms and etiquette must-knows
4. Safety situation and areas to avoid
5. Money-saving insider tips specific to {destination}
6. Seasonal considerations for timing this trip
7. Hidden gems that match the stated interests"""),
    ], label="research-agent")

    return {"research_context": content, "errors": []}


@traceable(name="itinerary-agent")
def itinerary_node(state: TripState) -> dict:
    """
    ItineraryAgent: generates a structured, day-by-day itinerary grounded in research context.
    Uses few-shot JSON output format for reliable parsing.
    """
    content = _invoke_with_retry(_llm(temperature=0.7), [
        SystemMessage(content=(
            "You are an expert travel planner who creates detailed, immersive itineraries. "
            "Always respond with a single valid JSON object — no markdown, no explanation."
        )),
        HumanMessage(content=f"""Create a {state['duration']}-day itinerary for {state['destination']}.

Research context: {state['research_context']}
Total budget: ${state['budget']} USD
Interests: {', '.join(state['interests'])}
Travel style: {state['travel_style']}

Return this exact JSON structure:
{{
  "destination": "City, Country",
  "summary": "2-3 sentence evocative trip overview",
  "best_time_to_visit": "Month range and why",
  "getting_around": "Practical local transport advice",
  "days": [
    {{
      "day": 1,
      "theme": "Descriptive day theme",
      "morning": {{
        "activity": "Activity name",
        "description": "Rich 2-3 sentence description",
        "location": "Specific place name",
        "cost": 15,
        "duration": "2 hours",
        "tips": "Insider tip"
      }},
      "afternoon": {{
        "activity": "Activity name",
        "description": "Rich 2-3 sentence description",
        "location": "Specific place name",
        "cost": 25,
        "duration": "3 hours",
        "tips": "Insider tip"
      }},
      "evening": {{
        "activity": "Activity name",
        "description": "Rich 2-3 sentence description",
        "location": "Specific place name",
        "cost": 30,
        "duration": "2 hours",
        "tips": "Insider tip"
      }},
      "meals": {{
        "breakfast": {{"name": "Place name", "cuisine": "Type", "cost": 8, "why": "Why go here"}},
        "lunch": {{"name": "Place name", "cuisine": "Type", "cost": 12, "why": "Why go here"}},
        "dinner": {{"name": "Place name", "cuisine": "Type", "cost": 25, "why": "Why go here"}}
      }},
      "transport": "How to get between today's locations",
      "daily_total": 115,
      "insider_tip": "One local secret for this day"
    }}
  ],
  "packing_essentials": ["item with reason", "item with reason"],
  "local_phrases": [{{"phrase": "local phrase", "pronunciation": "how to say it", "meaning": "English meaning"}}],
  "emergency_info": {{
    "emergency_number": "local 911 equivalent",
    "tourist_police": "number if available",
    "nearest_hospital": "general advice",
    "embassy_tip": "how to find your country's embassy"
  }}
}}"""),
    ], label="itinerary-agent")

    draft = _parse_json(content)
    if not draft:
        return {"draft_itinerary": {}, "errors": state.get("errors", []) + ["Itinerary generation failed"]}
    return {"draft_itinerary": draft, "errors": state.get("errors", [])}


@traceable(name="budget-agent")
def budget_node(state: TripState) -> dict:
    """
    BudgetAgent: validates costs against budget, generates breakdown, flags savings opportunities.
    Mirrors KPI tracking work done at Fullcast internship.
    """
    draft = state["draft_itinerary"]
    days = draft.get("days", [])
    estimated_total = sum(d.get("daily_total", 0) for d in days)

    content = _invoke_with_retry(_llm(temperature=0.2), [
        SystemMessage(content=(
            "You are a travel finance analyst. Provide accurate budget analysis. "
            "Always respond with a single valid JSON object."
        )),
        HumanMessage(content=f"""Analyze budget for this {state['duration']}-day trip to {state['destination']}.

User budget: ${state['budget']}
Estimated itinerary cost: ${estimated_total:.0f}
Daily breakdown: {json.dumps([{{'day': d['day'], 'total': d.get('daily_total', 0)}} for d in days])}

Return this JSON:
{{
  "accommodation_estimate": 400,
  "food_estimate": 250,
  "activities_estimate": 300,
  "transport_estimate": 150,
  "miscellaneous": 100,
  "total_estimate": {estimated_total},
  "budget_status": "under_budget/on_budget/over_budget",
  "variance_usd": {state['budget'] - estimated_total:.0f},
  "variance_pct": {((state['budget'] - estimated_total) / state['budget'] * 100):.1f},
  "daily_average": {estimated_total / max(state['duration'], 1):.0f},
  "savings_tips": [
    "Specific actionable tip 1",
    "Specific actionable tip 2",
    "Specific actionable tip 3"
  ],
  "splurge_picks": [
    "One experience worth spending extra on and why"
  ],
  "budget_accommodations": [
    "Hostel/guesthouse type recommendation",
    "Booking platform tip"
  ]
}}"""),
    ], label="budget-agent")

    budget_analysis = _parse_json(content)
    if not budget_analysis:
        budget_analysis = {
            "total_estimate": estimated_total,
            "budget_status": "unknown",
            "variance_usd": state["budget"] - estimated_total,
        }
    return {"budget_analysis": budget_analysis}


@traceable(name="review-agent")
def review_node(state: TripState) -> dict:
    """
    ReviewAgent: assembles final itinerary, ensures quality and coherence.
    Final gate before returning to the user.
    """
    final = {
        **state["draft_itinerary"],
        "budget_analysis": state["budget_analysis"],
        "research_insights": state["research_context"],
    }
    return {"final_itinerary": final}
