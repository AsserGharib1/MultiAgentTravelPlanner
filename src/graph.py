"""
graph.py — LangGraph workflow definition for the Multi-Agent Travel Planner.

Defines the state graph that orchestrates all agents:
  parse_query → [search_restaurants, search_accommodations, search_attractions] → plan_itinerary
"""

import json
from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END

from src.agents.query_parser import QueryParserAgent
from src.agents.restaurant import RestaurantAgent
from src.agents.accommodation import AccommodationAgent
from src.agents.attraction import AttractionAgent
from src.agents.planner import PlannerAgent


# ──────────────────────────────────────────────────────────────
# State Definition
# ──────────────────────────────────────────────────────────────

class TravelPlannerState(TypedDict):
    """State shared across all nodes in the graph."""
    query_record: Dict          # Raw query record from dataset
    constraints: Dict           # Parsed constraints
    restaurant_result: Dict     # Output from restaurant agent
    accommodation_result: Dict  # Output from accommodation agent
    attraction_result: Dict     # Output from attraction agent
    travel_plan: Dict           # Final compiled plan
    errors: List[str]           # Any errors during execution


# ──────────────────────────────────────────────────────────────
# Node Functions
# ──────────────────────────────────────────────────────────────

def parse_query_node(state: TravelPlannerState, agents: Dict) -> Dict:
    """Parse the user query into structured constraints."""
    print("  [Node] Parsing query...")
    parser: QueryParserAgent = agents["query_parser"]
    try:
        constraints = parser.parse(state["query_record"], use_llm=False)
        return {"constraints": constraints}
    except Exception as e:
        return {"constraints": {}, "errors": state.get("errors", []) + [f"QueryParser: {e}"]}


def search_restaurants_node(state: TravelPlannerState, agents: Dict) -> Dict:
    """Search and select restaurants."""
    print("  [Node] Searching restaurants...")
    agent: RestaurantAgent = agents["restaurant"]
    try:
        result = agent.recommend(state["constraints"])
        return {"restaurant_result": result}
    except Exception as e:
        return {
            "restaurant_result": {"selected_restaurants": [], "total_estimated_food_cost": 0},
            "errors": state.get("errors", []) + [f"RestaurantAgent: {e}"],
        }


def search_accommodations_node(state: TravelPlannerState, agents: Dict) -> Dict:
    """Search and select accommodation."""
    print("  [Node] Searching accommodations...")
    agent: AccommodationAgent = agents["accommodation"]
    try:
        result = agent.recommend(state["constraints"])
        return {"accommodation_result": result}
    except Exception as e:
        return {
            "accommodation_result": {"selected_accommodation": {}},
            "errors": state.get("errors", []) + [f"AccommodationAgent: {e}"],
        }


def search_attractions_node(state: TravelPlannerState, agents: Dict) -> Dict:
    """Search and select attractions."""
    print("  [Node] Searching attractions...")
    agent: AttractionAgent = agents["attraction"]
    try:
        result = agent.recommend(state["constraints"])
        return {"attraction_result": result}
    except Exception as e:
        return {
            "attraction_result": {"selected_attractions": []},
            "errors": state.get("errors", []) + [f"AttractionAgent: {e}"],
        }


def plan_itinerary_node(state: TravelPlannerState, agents: Dict) -> Dict:
    """Compile the final travel plan from all sub-agent outputs."""
    print("  [Node] Planning itinerary...")
    planner: PlannerAgent = agents["planner"]
    try:
        result = planner.plan(
            state["constraints"],
            state.get("restaurant_result", {}),
            state.get("accommodation_result", {}),
            state.get("attraction_result", {}),
        )
        return {"travel_plan": result}
    except Exception as e:
        return {
            "travel_plan": {},
            "errors": state.get("errors", []) + [f"PlannerAgent: {e}"],
        }


# ──────────────────────────────────────────────────────────────
# Graph Builder
# ──────────────────────────────────────────────────────────────

def build_travel_planner_graph(
    restaurants_df,
    accommodations_df,
    attractions_df,
    attraction_summaries,
    model_key: str = None,
    temperature: float = None,
):
    """
    Build and compile the LangGraph travel planner workflow.

    Args:
        restaurants_df: Restaurant DataFrame
        accommodations_df: Accommodation DataFrame
        attractions_df: Attraction DataFrame
        attraction_summaries: Dict of attraction pro/con summaries
        model_key: Model size key ("small" or "large")
        temperature: LLM temperature

    Returns:
        Compiled graph and agents dict
    """
    # Initialize all agents with the same model config
    agents = {
        "query_parser": QueryParserAgent(model_key=model_key, temperature=temperature),
        "restaurant": RestaurantAgent(restaurants_df, model_key=model_key, temperature=temperature),
        "accommodation": AccommodationAgent(accommodations_df, model_key=model_key, temperature=temperature),
        "attraction": AttractionAgent(attractions_df, attraction_summaries, model_key=model_key, temperature=temperature),
        "planner": PlannerAgent(model_key=model_key, temperature=temperature),
    }

    # Build the state graph
    workflow = StateGraph(TravelPlannerState)

    # Add nodes (wrap to pass agents)
    workflow.add_node("parse_query", lambda s: parse_query_node(s, agents))
    workflow.add_node("search_restaurants", lambda s: search_restaurants_node(s, agents))
    workflow.add_node("search_accommodations", lambda s: search_accommodations_node(s, agents))
    workflow.add_node("search_attractions", lambda s: search_attractions_node(s, agents))
    workflow.add_node("plan_itinerary", lambda s: plan_itinerary_node(s, agents))

    # Define edges
    workflow.set_entry_point("parse_query")

    # After parsing, run all three search agents
    workflow.add_edge("parse_query", "search_restaurants")
    workflow.add_edge("parse_query", "search_accommodations")
    workflow.add_edge("parse_query", "search_attractions")

    # After all searches complete, plan the itinerary
    workflow.add_edge("search_restaurants", "plan_itinerary")
    workflow.add_edge("search_accommodations", "plan_itinerary")
    workflow.add_edge("search_attractions", "plan_itinerary")

    # End after planning
    workflow.add_edge("plan_itinerary", END)

    # Compile
    graph = workflow.compile()
    return graph, agents


def run_travel_planner(graph, query_record: Dict) -> Dict:
    """
    Run the travel planner graph on a single query.

    Args:
        graph: Compiled LangGraph
        query_record: Raw query record from dataset

    Returns:
        Final state with travel plan
    """
    initial_state = {
        "query_record": query_record,
        "constraints": {},
        "restaurant_result": {},
        "accommodation_result": {},
        "attraction_result": {},
        "travel_plan": {},
        "errors": [],
    }

    # Execute the graph
    result = graph.invoke(initial_state)
    return result
