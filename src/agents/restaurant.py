"""
restaurant.py — Restaurant Recommendation Agent.

Searches and ranks restaurants from the database based on user constraints
(city, cuisine, budget) and uses an LLM to make final selections.
"""

import json
import time
import re
from typing import Dict, List, Any

import pandas as pd
from groq import Groq

from src.config import GROQ_API_KEY, HYPERPARAMS, RATE_LIMIT_DELAY, get_model_name, parse_llm_json
from src.prompts import RESTAURANT_SYSTEM, RESTAURANT_USER
from src.data_loader import filter_restaurants_by_city, filter_restaurants_by_cuisine


class RestaurantAgent:
    """
    Agent that recommends restaurants for a travel plan.

    Pipeline:
    1. Filter restaurants by destination city
    2. Filter by cuisine preferences
    3. Rank by rating and cost
    4. Use LLM to make final diverse selection
    """

    def __init__(self, restaurants_df: pd.DataFrame, model_key: str = None, temperature: float = None):
        self.restaurants_df = restaurants_df
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = get_model_name(model_key)
        self.temperature = temperature if temperature is not None else HYPERPARAMS["temperature"]

    def _get_candidates(self, destination: str, cuisines: List[str], budget: float, top_n: int = 10) -> pd.DataFrame:
        """
        Filter and rank restaurant candidates.

        Args:
            destination: City name
            cuisines: List of preferred cuisine types
            budget: Total trip budget (used to estimate per-meal budget)
            top_n: Number of top candidates to return

        Returns:
            DataFrame of top restaurant candidates
        """
        # Filter by city
        city_restaurants = filter_restaurants_by_city(self.restaurants_df, destination)

        if city_restaurants.empty:
            # Try partial match
            mask = self.restaurants_df["City"].str.lower().str.contains(destination.lower(), na=False)
            city_restaurants = self.restaurants_df[mask].copy()

        if city_restaurants.empty:
            return pd.DataFrame()

        # Filter by cuisine (if specified)
        if cuisines:
            cuisine_matched = filter_restaurants_by_cuisine(city_restaurants, cuisines)
            if not cuisine_matched.empty:
                city_restaurants = cuisine_matched

        # Rank by rating (descending), then by cost (ascending for budget-friendliness)
        city_restaurants = city_restaurants.sort_values(
            by=["Aggregate Rating", "Average Cost"],
            ascending=[False, True],
        )

        return city_restaurants.head(top_n)

    def _format_restaurant_list(self, df: pd.DataFrame) -> str:
        """Format restaurant DataFrame as a string for the LLM prompt."""
        lines = []
        for _, row in df.iterrows():
            line = (
                f"- {row['Name']} | Cuisine: {row.get('Cuisines', 'N/A')} | "
                f"Rating: {row.get('Aggregate Rating', 'N/A')} | "
                f"Avg Cost: ${row.get('Average Cost', 'N/A')} | "
                f"Reviews: {row.get('num_of_reviews', 'N/A')}"
            )
            lines.append(line)
        return "\n".join(lines)

    def recommend(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend restaurants based on parsed query constraints.

        Args:
            constraints: Parsed query constraints dict

        Returns:
            Dict with selected restaurants and estimated cost
        """
        destination = constraints.get("destination", "")
        cuisines = constraints.get("cuisines", [])
        days = constraints.get("days", 1)
        people = constraints.get("people_number", 1)
        budget = constraints.get("budget", 1000)

        # Estimate food budget as ~40% of total
        food_budget = budget * 0.4
        total_needed = days * 2  # lunch + dinner per day

        # Get candidates
        candidates = self._get_candidates(destination, cuisines, budget)

        if candidates.empty:
            return {
                "selected_restaurants": [],
                "total_estimated_food_cost": 0,
                "error": f"No restaurants found in {destination}",
            }

        # Format for LLM
        restaurant_list_str = self._format_restaurant_list(candidates)

        user_prompt = RESTAURANT_USER.format(
            days=days,
            destination=destination,
            people_number=people,
            food_budget=food_budget,
            cuisines=", ".join(cuisines) if cuisines else "No specific preference",
            restaurant_list=restaurant_list_str,
            total_needed=total_needed,
        )

        # Call LLM
        for attempt in range(HYPERPARAMS["retry_attempts"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": RESTAURANT_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=HYPERPARAMS["max_tokens"],
                )
                content = response.choices[0].message.content.strip()
                result = parse_llm_json(content)
                time.sleep(RATE_LIMIT_DELAY)
                return result

            except (json.JSONDecodeError, Exception) as e:
                print(f"  [RestaurantAgent] Attempt {attempt + 1} failed: {e}")
                if attempt < HYPERPARAMS["retry_attempts"] - 1:
                    time.sleep(HYPERPARAMS["retry_delay"])

        # Fallback: return top candidates directly
        fallback = []
        for _, row in candidates.head(total_needed).iterrows():
            fallback.append({
                "name": row["Name"],
                "cuisine": str(row.get("Cuisines", "N/A")),
                "rating": float(row.get("Aggregate Rating", 0)),
                "average_cost": float(row.get("Average Cost", 0)),
                "reason": "Selected by rating (LLM fallback)",
            })

        total_cost = sum(r["average_cost"] * people for r in fallback)
        return {
            "selected_restaurants": fallback,
            "total_estimated_food_cost": total_cost,
        }
