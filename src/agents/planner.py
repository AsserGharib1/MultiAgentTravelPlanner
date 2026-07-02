"""
planner.py — Planner/Orchestrator Agent.

Receives outputs from Restaurant, Accommodation, and Attraction agents,
then creates a coherent day-by-day travel itinerary using an LLM.
"""

import json
import time
import re
from typing import Dict, Any

from groq import Groq

from src.config import GROQ_API_KEY, HYPERPARAMS, RATE_LIMIT_DELAY, get_model_name, parse_llm_json
from src.prompts import PLANNER_SYSTEM, PLANNER_USER


class PlannerAgent:
    """Agent that compiles final day-by-day travel itinerary."""

    def __init__(self, model_key=None, temperature=None):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = get_model_name(model_key)
        self.temperature = temperature if temperature is not None else HYPERPARAMS["temperature"]

    def plan(self, constraints, restaurant_result, accommodation_result, attraction_result):
        """
        Create a complete travel plan from agent outputs.

        Args:
            constraints: Parsed query constraints
            restaurant_result: Output from RestaurantAgent
            accommodation_result: Output from AccommodationAgent
            attraction_result: Output from AttractionAgent

        Returns:
            Dict with complete travel plan
        """
        origin = constraints.get("origin", "")
        destination = constraints.get("destination", "")
        dates = constraints.get("dates", [])
        people = constraints.get("people_number", 1)
        budget = constraints.get("budget", 1000)
        days = constraints.get("days", 1)

        # Format sub-agent outputs for the planner prompt
        acc = accommodation_result.get("selected_accommodation", {})
        acc_details = json.dumps(acc, indent=2) if acc else "No accommodation selected"

        restaurants = restaurant_result.get("selected_restaurants", [])
        rest_details = json.dumps(restaurants, indent=2) if restaurants else "No restaurants selected"

        attractions = attraction_result.get("selected_attractions", [])
        attr_details = json.dumps(attractions, indent=2) if attractions else "No attractions selected"

        user_prompt = PLANNER_USER.format(
            origin=origin, destination=destination,
            dates=json.dumps(dates), people_number=people, budget=budget,
            accommodation_details=acc_details,
            restaurant_details=rest_details,
            attraction_details=attr_details,
        )

        for attempt in range(HYPERPARAMS["retry_attempts"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=HYPERPARAMS["max_tokens"],
                )
                content = response.choices[0].message.content.strip()
                result = parse_llm_json(content)
                time.sleep(RATE_LIMIT_DELAY)
                return result
            except Exception as e:
                print(f"  [PlannerAgent] Attempt {attempt + 1} failed: {e}")
                if attempt < HYPERPARAMS["retry_attempts"] - 1:
                    time.sleep(HYPERPARAMS["retry_delay"])

        # Fallback: construct a basic plan
        return self._fallback_plan(constraints, restaurant_result, accommodation_result, attraction_result)

    def _fallback_plan(self, constraints, restaurant_result, accommodation_result, attraction_result):
        """Build a basic plan without LLM if all attempts fail."""
        days = constraints.get("days", 1)
        dates = constraints.get("dates", [])
        acc = accommodation_result.get("selected_accommodation", {})
        restaurants = restaurant_result.get("selected_restaurants", [])
        attractions = attraction_result.get("selected_attractions", [])

        daily = []
        for d in range(days):
            day_activities = []
            # Add attractions (2 per day)
            for i in range(2):
                idx = d * 2 + i
                if idx < len(attractions):
                    day_activities.append({
                        "time": "10:00" if i == 0 else "14:00",
                        "type": "attraction",
                        "name": attractions[idx].get("name", "Attraction"),
                        "details": attractions[idx].get("reason", ""),
                    })
            # Add restaurants
            lunch_idx = d * 2
            dinner_idx = d * 2 + 1
            if lunch_idx < len(restaurants):
                day_activities.append({
                    "time": "12:00", "type": "lunch",
                    "name": restaurants[lunch_idx].get("name", "Restaurant"),
                    "cuisine": restaurants[lunch_idx].get("cuisine", ""),
                    "est_cost": restaurants[lunch_idx].get("average_cost", 0),
                })
            if dinner_idx < len(restaurants):
                day_activities.append({
                    "time": "19:00", "type": "dinner",
                    "name": restaurants[dinner_idx].get("name", "Restaurant"),
                    "cuisine": restaurants[dinner_idx].get("cuisine", ""),
                    "est_cost": restaurants[dinner_idx].get("average_cost", 0),
                })

            daily.append({
                "day": d + 1,
                "date": dates[d] if d < len(dates) else f"Day {d+1}",
                "activities": day_activities,
            })

        acc_total = float(acc.get("total_cost", acc.get("price_per_night", 0) * days))
        food_total = float(restaurant_result.get("total_estimated_food_cost", 0))

        return {
            "travel_plan": {
                "origin": constraints.get("origin", ""),
                "destination": constraints.get("destination", ""),
                "dates": dates,
                "people": constraints.get("people_number", 1),
                "accommodation": acc,
                "daily_itinerary": daily,
                "budget_summary": {
                    "accommodation_total": acc_total,
                    "food_total": food_total,
                    "estimated_total": acc_total + food_total,
                    "user_budget": constraints.get("budget", 0),
                    "within_budget": (acc_total + food_total) <= constraints.get("budget", 0),
                },
            }
        }
