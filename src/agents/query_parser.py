"""
query_parser.py — Query Parser Agent.

Parses natural language travel queries into structured constraint dictionaries
using an LLM. Also supports rule-based fallback parsing from the existing
structured fields in the dataset.
"""

import json
import time
import re
from typing import Dict, Any, Optional

from groq import Groq

from src.config import GROQ_API_KEY, HYPERPARAMS, RATE_LIMIT_DELAY, get_model_name, parse_llm_json
from src.prompts import QUERY_PARSER_SYSTEM, QUERY_PARSER_USER
from src.data_loader import parse_local_constraint


class QueryParserAgent:
    """
    Agent that parses natural language travel queries into structured constraints.

    Supports both LLM-based parsing and rule-based fallback using the
    pre-existing structured fields in the RealTravel dataset.
    """

    def __init__(self, model_key: str = None, temperature: float = None):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = get_model_name(model_key)
        self.temperature = temperature if temperature is not None else HYPERPARAMS["temperature"]

    def parse_query_llm(self, query_text: str, user_id: str = "") -> Dict[str, Any]:
        """
        Use the LLM to parse a natural language query into structured constraints.

        Args:
            query_text: The raw user query string
            user_id: Optional user ID for context

        Returns:
            Dict with parsed constraints
        """
        user_prompt = QUERY_PARSER_USER.format(query=query_text, user_id=user_id)

        for attempt in range(HYPERPARAMS["retry_attempts"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": QUERY_PARSER_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content.strip()
                parsed = parse_llm_json(content)
                time.sleep(RATE_LIMIT_DELAY)
                return parsed

            except (json.JSONDecodeError, Exception) as e:
                print(f"  [QueryParser] Attempt {attempt + 1} failed: {e}")
                if attempt < HYPERPARAMS["retry_attempts"] - 1:
                    time.sleep(HYPERPARAMS["retry_delay"])

        # Fallback to empty dict on failure
        print("  [QueryParser] All LLM attempts failed, using fallback")
        return {}

    def parse_query_structured(self, query_record: Dict) -> Dict[str, Any]:
        """
        Extract structured constraints directly from the dataset's pre-parsed fields.

        This is the rule-based fallback that uses the existing structured data
        in the RealTravel dataset (which is already well-parsed).

        Args:
            query_record: A full query record from the JSONL file

        Returns:
            Dict with structured constraints
        """
        local_constraint = parse_local_constraint(
            query_record.get("local_constraint", "{}")
        )

        # Parse dates
        dates = query_record.get("date", "[]")
        if isinstance(dates, str):
            try:
                dates = json.loads(dates)
            except json.JSONDecodeError:
                dates = []

        return {
            "origin": query_record.get("org", ""),
            "destination": query_record.get("dest", ""),
            "days": query_record.get("days", 1),
            "dates": dates,
            "people_number": query_record.get("people_number", 1),
            "budget": query_record.get("budget", 0),
            "room_type": local_constraint.get("room type"),
            "cuisines": local_constraint.get("cuisine", []) or [],
            "house_rules": local_constraint.get("house rule"),
            "transportation": local_constraint.get("transportation"),
            "user_id": query_record.get("user_id", ""),
            "level": query_record.get("level", ""),
            "attraction_preference": query_record.get("attraction_preference", {}),
        }

    def parse(self, query_record: Dict, use_llm: bool = True) -> Dict[str, Any]:
        """
        Parse a query record, using LLM if requested, with structured fallback.

        Args:
            query_record: Full query record from dataset
            use_llm: Whether to use LLM parsing (True) or structured only (False)

        Returns:
            Dict with all structured constraints
        """
        # Always get structured data as baseline
        structured = self.parse_query_structured(query_record)

        if use_llm:
            llm_parsed = self.parse_query_llm(
                query_record.get("query", ""),
                query_record.get("user_id", ""),
            )
            # Merge: prefer structured data but fill gaps from LLM
            for key, value in llm_parsed.items():
                if key not in structured or structured[key] is None:
                    structured[key] = value

        return structured
