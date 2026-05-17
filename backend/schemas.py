from pydantic import BaseModel, Field
from typing import Optional


class TripRequest(BaseModel):
    destination: str = Field(..., min_length=2, max_length=100, example="Kyoto, Japan")
    budget: float = Field(..., gt=0, le=100000, example=2000)
    duration: int = Field(..., ge=1, le=30, example=7)
    interests: list[str] = Field(..., min_length=1, example=["Culture & History", "Food & Cuisine"])
    travel_style: str = Field(default="balanced", example="budget/balanced/comfort")


class TimeSlot(BaseModel):
    activity: str
    description: str
    location: str
    cost: float
    duration: str
    tips: Optional[str] = None


class Meal(BaseModel):
    name: str
    cuisine: str
    cost: float
    why: str


class DayPlan(BaseModel):
    day: int
    theme: str
    morning: TimeSlot
    afternoon: TimeSlot
    evening: TimeSlot
    meals: dict[str, Meal]
    transport: str
    daily_total: float
    insider_tip: str


class BudgetBreakdown(BaseModel):
    accommodation_estimate: float
    food_estimate: float
    activities_estimate: float
    transport_estimate: float
    miscellaneous: float
    total_estimate: float
    budget_status: str
    savings_tips: list[str]
    splurge_picks: list[str]


class Itinerary(BaseModel):
    destination: str
    summary: str
    days: list[DayPlan]
    budget_analysis: BudgetBreakdown
    packing_essentials: list[str]
    local_phrases: list[dict]
    emergency_info: dict
    best_time_to_visit: str
    getting_around: str
    research_insights: str


class TripResponse(BaseModel):
    itinerary: Itinerary
    trace_url: Optional[str] = None
    cached: bool = False
    generation_time_ms: int = 0
