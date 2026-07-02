"""
attraction.py — Attraction Recommendation Agent.

Searches and ranks attractions based on user preferences (likes/dislikes),
pro/con summaries, ratings, and uses an LLM for final selection.
"""

import json
import time
import re
from typing import Dict, List, Any

import pandas as pd
from groq import Groq

from src.config import GROQ_API_KEY, HYPERPARAMS, RATE_LIMIT_DELAY, get_model_name, parse_llm_json
from src.prompts import ATTRACTION_SYSTEM, ATTRACTION_USER
from src.data_loader import filter_attractions_by_city


class AttractionAgent:
    """Agent that recommends attractions for a travel plan."""

    def __init__(self, attractions_df, attraction_summaries, model_key=None, temperature=None):
        self.attractions_df = attractions_df
        self.summaries = attraction_summaries
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = get_model_name(model_key)
        self.temperature = temperature if temperature is not None else HYPERPARAMS["temperature"]

    def _compute_preference_score(self, row, likes, dislikes):
        score = 0.0
        gmap_id = str(row.get("gmap_id", ""))
        category = str(row.get("category", "")).lower()
        description = str(row.get("description", "")).lower()
        summary = self.summaries.get(gmap_id, {})
        pro_text = str(summary.get("pro", "")).lower()
        attraction_text = f"{category} {description} {pro_text}"

        for like in likes:
            if like.lower() in attraction_text:
                score += 1.0
            elif any(w in attraction_text for w in like.lower().split() if len(w) > 3):
                score += 0.3

        con_text = str(summary.get("con", "")).lower()
        for dislike in dislikes:
            if dislike.lower() in f"{con_text} {description}":
                score -= 0.5

        return score / max(len(likes) + len(dislikes), 1)

    def _get_candidates(self, destination, likes, dislikes, top_n=8):
        city_att = filter_attractions_by_city(self.attractions_df, destination)
        if city_att.empty:
            mask = self.attractions_df["City"].str.lower().str.contains(destination.lower(), na=False)
            city_att = self.attractions_df[mask].copy()
        if city_att.empty:
            return pd.DataFrame()

        city_att = city_att.copy()
        city_att["preference_score"] = city_att.apply(
            lambda r: self._compute_preference_score(r, likes, dislikes), axis=1
        )
        city_att = city_att.sort_values(
            by=["preference_score", "Aggregate Rating", "num_of_reviews"],
            ascending=[False, False, False],
        )
        return city_att.head(top_n)

    def _format_attraction_list(self, df):
        lines = []
        for _, row in df.iterrows():
            gmap_id = str(row.get("gmap_id", ""))
            summary = self.summaries.get(gmap_id, {})
            pro = str(summary.get("pro", "N/A"))[:60]
            con = str(summary.get("con", "N/A"))[:60]
            line = (
                f"- {row['Name']} | Category: {row.get('category', 'N/A')} | "
                f"Rating: {row.get('Aggregate Rating', 'N/A')} | "
                f"Score: {row.get('preference_score', 0):.2f} | Pros: {pro} | Cons: {con}"
            )
            lines.append(line)
        return "\n".join(lines)

    def recommend(self, constraints):
        destination = constraints.get("destination", "")
        days = constraints.get("days", 1)
        prefs = constraints.get("attraction_preference", {})
        likes = prefs.get("likes", [])
        dislikes = prefs.get("dislikes", [])
        total_needed = days * 2

        candidates = self._get_candidates(destination, likes, dislikes)
        if candidates.empty:
            return {"selected_attractions": [], "error": f"No attractions found in {destination}"}

        attraction_list_str = self._format_attraction_list(candidates)
        user_prompt = ATTRACTION_USER.format(
            days=days, destination=destination,
            likes=", ".join(likes) if likes else "No specific likes",
            dislikes=", ".join(dislikes) if dislikes else "No specific dislikes",
            attraction_list=attraction_list_str, total_needed=total_needed,
        )

        for attempt in range(HYPERPARAMS["retry_attempts"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ATTRACTION_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature, max_tokens=HYPERPARAMS["max_tokens"],
                )
                content = response.choices[0].message.content.strip()
                result = parse_llm_json(content)
                time.sleep(RATE_LIMIT_DELAY)
                return result
            except Exception as e:
                print(f"  [AttractionAgent] Attempt {attempt + 1} failed: {e}")
                if attempt < HYPERPARAMS["retry_attempts"] - 1:
                    time.sleep(HYPERPARAMS["retry_delay"])

        # Fallback
        fallback = []
        for _, row in candidates.head(total_needed).iterrows():
            fallback.append({
                "name": row["Name"],
                "category": str(row.get("category", "N/A")),
                "rating": float(row.get("Aggregate Rating", 0)),
                "preference_match": f"Score: {row.get('preference_score', 0):.2f}",
                "reason": "Selected by preference score (LLM fallback)",
            })
        return {"selected_attractions": fallback}
