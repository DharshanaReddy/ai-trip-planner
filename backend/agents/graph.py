"""
LangGraph multi-agent workflow for trip planning.

Pipeline: ResearchAgent → ItineraryAgent → BudgetAgent → ReviewAgent

Each agent is a discrete node with typed state — enabling independent testing,
parallel execution, and LangSmith tracing of each reasoning step.
"""

from langgraph.graph import StateGraph, END
from langsmith import traceable

from .state import TripState
from .nodes import research_node, itinerary_node, budget_node, review_node


def _build_graph() -> StateGraph:
    workflow = StateGraph(TripState)

    workflow.add_node("research", research_node)
    workflow.add_node("plan", itinerary_node)
    workflow.add_node("budget", budget_node)
    workflow.add_node("review", review_node)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "plan")
    workflow.add_edge("plan", "budget")
    workflow.add_edge("budget", "review")
    workflow.add_edge("review", END)

    return workflow.compile()


trip_graph = _build_graph()


@traceable(name="trip-planner-pipeline", run_type="chain")
def run_trip_pipeline(
    destination: str,
    budget: float,
    duration: int,
    interests: list[str],
    travel_style: str = "balanced",
) -> dict:
    """
    Execute the full multi-agent trip planning pipeline.
    Fully traced in LangSmith for latency monitoring and quality evaluation.
    """
    initial_state: TripState = {
        "destination": destination,
        "budget": budget,
        "duration": duration,
        "interests": interests,
        "travel_style": travel_style,
        "research_context": "",
        "draft_itinerary": {},
        "budget_analysis": {},
        "final_itinerary": {},
        "errors": [],
    }

    result = trip_graph.invoke(initial_state)
    return result["final_itinerary"]
