from typing import TypedDict, Optional


class TripState(TypedDict):
    # Input
    destination: str
    budget: float
    duration: int
    interests: list[str]
    travel_style: str

    # Agent outputs
    research_context: str
    draft_itinerary: dict
    budget_analysis: dict
    final_itinerary: dict

    # Meta
    errors: list[str]
