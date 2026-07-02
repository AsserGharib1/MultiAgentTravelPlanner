"""
accommodation.py — Accommodation Recommendation Agent.

Searches and ranks accommodations from the database based on user constraints
(city, room type, budget, occupancy) and uses an LLM to make the final selection.
"""

import json
import time
import re
from typing import Dict, Any

import pandas as pd
from groq import Groq

from src.config import GROQ_API_KEY, HYPERPARAMS, RATE_LIMIT_DELAY, get_model_name, parse_llm_json
from src.prompts import ACCOMMODATION_SYSTEM, ACCOMMODATION_USER
from src.data_loader import (
    filter_accommodations_by_city,
    filter_accommodations_by_room_type,
)


class AccommodationAgent:
    """
    Agent that recommends accommodation for a travel plan.

    Pipeline:
    1. Filter accommodations by destination city
    2. Filter by room type preference
    3. Filter by budget and occupancy
    4. Rank by review rating and price
    5. Use LLM to make final selection
    """

    def __init__(self, accommodations_df: pd.DataFrame, model_key: str = None, temperature: float = None):
        self.accommodations_df = accommodations_df
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = get_model_name(model_key)
        self.temperature = temperature if temperature is not None else HYPERPARAMS["temperature"]

    def _get_candidates(
        self,
        destination: str,
        room_type: str,
        people: int,
        budget_for_accommodation: float,
        days: int,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Filter and rank accommodation candidates.
        """
        # Filter by city
        city_acc = filter_accommodations_by_city(self.accommodations_df, destination)

        if city_acc.empty:
            # Try partial match
            mask = self.accommodations_df["city"].str.lower().str.contains(
                destination.lower(), na=False
            )
            city_acc = self.accommodations_df[mask].copy()

        if city_acc.empty:
            return pd.DataFrame()

        # Filter by room type (if specified)
        if room_type:
            room_filtered = filter_accommodations_by_room_type(city_acc, room_type)
            if not room_filtered.empty:
                city_acc = room_filtered

        # Filter by occupancy
        city_acc = city_acc[city_acc["maximum occupancy"] >= people].copy()

        # Filter by budget (price per night × days)
        if budget_for_accommodation > 0 and days > 0:
            max_price_per_night = budget_for_accommodation / days
            city_acc = city_acc[city_acc["price"] <= max_price_per_night].copy()

        # If too restrictive, relax budget filter
        if city_acc.empty:
            city_acc = filter_accommodations_by_city(self.accommodations_df, destination)
            if room_type:
                room_filtered = filter_accommodations_by_room_type(city_acc, room_type)
                if not room_filtered.empty:
                    city_acc = room_filtered

        # Rank by review rating (desc), then price (asc)
        city_acc = city_acc.sort_values(
            by=["review rate number", "price"],
            ascending=[False, True],
        )

        return city_acc.head(top_n)

    def _format_accommodation_list(self, df: pd.DataFrame) -> str:
        """Format accommodation DataFrame as a string for the LLM prompt."""
        lines = []
        for _, row in df.iterrows():
            line = (
                f"- {row['NAME']} | Room Type: {row.get('room type', 'N/A')} | "
                f"Price/Night: ${row.get('price', 'N/A')} | "
                f"Rating: {row.get('review rate number', 'N/A')}/5 | "
                f"Max Occupancy: {row.get('maximum occupancy', 'N/A')} | "
                f"Rules: {row.get('house_rules', 'None')}"
            )
            lines.append(line)
        return "\n".join(lines)

    def recommend(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend accommodation based on parsed query constraints.

        Args:
            constraints: Parsed query constraints dict

        Returns:
            Dict with selected accommodation details
        """
        destination = constraints.get("destination", "")
        room_type = constraints.get("room_type", "")
        days = constraints.get("days", 1)
        people = constraints.get("people_number", 1)
        budget = constraints.get("budget", 1000)
        house_rules = constraints.get("house_rules", "")

        # Estimate accommodation budget as ~40% of total
        accommodation_budget = budget * 0.4

        # Get candidates
        candidates = self._get_candidates(
            destination, room_type, people, accommodation_budget, days
        )

        if candidates.empty:
            return {
                "selected_accommodation": {
                    "name": "No accommodation found",
                    "room_type": room_type or "N/A",
                    "price_per_night": 0,
                    "total_cost": 0,
                    "review_rating": 0,
                    "max_occupancy": 0,
                    "reason": f"No accommodations found in {destination}",
                },
                "error": f"No accommodations found in {destination}",
            }

        # Format for LLM
        accommodation_list_str = self._format_accommodation_list(candidates)

        user_prompt = ACCOMMODATION_USER.format(
            days=days,
            destination=destination,
            people_number=people,
            accommodation_budget=accommodation_budget,
            room_type=room_type or "No preference",
            house_rules=house_rules or "No specific rules",
            accommodation_list=accommodation_list_str,
        )

        # Call LLM
        for attempt in range(HYPERPARAMS["retry_attempts"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ACCOMMODATION_SYSTEM},
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
                print(f"  [AccommodationAgent] Attempt {attempt + 1} failed: {e}")
                if attempt < HYPERPARAMS["retry_attempts"] - 1:
                    time.sleep(HYPERPARAMS["retry_delay"])

        # Fallback: pick top candidate
        top = candidates.iloc[0]
        return {
            "selected_accommodation": {
                "name": top["NAME"],
                "room_type": str(top.get("room type", "N/A")),
                "price_per_night": float(top.get("price", 0)),
                "total_cost": float(top.get("price", 0)) * days,
                "review_rating": float(top.get("review rate number", 0)),
                "max_occupancy": int(top.get("maximum occupancy", 1)),
                "reason": "Best rated option (LLM fallback)",
            }
        }
