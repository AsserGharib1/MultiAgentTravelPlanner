"""
evaluate.py — Evaluation metrics for the Multi-Agent Travel Planner.

Implements 5 evaluation metrics:
1. Budget Feasibility (rule-based)
2. Constraint Satisfaction (rule-based)
3. Preference Alignment (deterministic — Jaccard similarity)
4. Plan Completeness (rule-based)
5. LLM-as-Judge (LLM-based quality rating)
"""

import json
import time
import re
from typing import Dict, List, Any, Optional

from groq import Groq

from src.config import GROQ_API_KEY, HYPERPARAMS, RATE_LIMIT_DELAY, get_model_name, parse_llm_json
from src.prompts import JUDGE_SYSTEM, JUDGE_USER


# ──────────────────────────────────────────────────────────────
# Metric 1: Budget Feasibility
# ──────────────────────────────────────────────────────────────

def evaluate_budget_feasibility(plan: Dict, constraints: Dict) -> Dict[str, Any]:
    """
    Check if the travel plan stays within the user's budget.

    Returns:
        {"score": float 0-1, "details": {...}}
    """
    budget = constraints.get("budget", 0)
    if budget <= 0:
        return {"score": 1.0, "details": {"reason": "No budget specified"}}

    travel_plan = plan.get("travel_plan", plan)
    budget_summary = travel_plan.get("budget_summary", {})

    estimated_total = budget_summary.get("estimated_total", 0)
    if estimated_total == 0:
        # Try to compute from accommodation and food
        acc = travel_plan.get("accommodation", {})
        acc_cost = acc.get("total_cost", acc.get("price_per_night", 0) * constraints.get("days", 1))
        food_cost = budget_summary.get("food_total", 0)
        estimated_total = acc_cost + food_cost

    if estimated_total <= budget:
        score = 1.0
    else:
        overshoot = (estimated_total - budget) / budget
        score = max(0.0, 1.0 - overshoot)

    return {
        "score": round(score, 3),
        "details": {
            "user_budget": budget,
            "estimated_total": estimated_total,
            "within_budget": estimated_total <= budget,
        },
    }


# ──────────────────────────────────────────────────────────────
# Metric 2: Constraint Satisfaction
# ──────────────────────────────────────────────────────────────

def evaluate_constraint_satisfaction(plan: Dict, constraints: Dict) -> Dict[str, Any]:
    """
    Check what percentage of user constraints are satisfied.

    Checks: destination, days, cuisines, room type.
    """
    checks = {}
    travel_plan = plan.get("travel_plan", plan)

    # Check destination
    plan_dest = str(travel_plan.get("destination", "")).lower()
    user_dest = str(constraints.get("destination", "")).lower()
    checks["destination"] = user_dest in plan_dest or plan_dest in user_dest

    # Check number of days
    itinerary = travel_plan.get("daily_itinerary", [])
    checks["days"] = len(itinerary) >= constraints.get("days", 1)

    # Check cuisine (if any restaurant mentions the cuisine)
    user_cuisines = [c.lower() for c in constraints.get("cuisines", [])]
    if user_cuisines:
        plan_str = json.dumps(travel_plan).lower()
        cuisine_found = any(c in plan_str for c in user_cuisines)
        checks["cuisine"] = cuisine_found
    else:
        checks["cuisine"] = True  # No preference = satisfied

    # Check room type
    user_room = constraints.get("room_type", "")
    if user_room:
        acc = travel_plan.get("accommodation", {})
        plan_room = str(acc.get("room_type", "")).lower()
        checks["room_type"] = user_room.lower() in plan_room
    else:
        checks["room_type"] = True

    satisfied = sum(1 for v in checks.values() if v)
    total = len(checks)
    score = satisfied / total if total > 0 else 1.0

    return {
        "score": round(score, 3),
        "details": checks,
    }


# ──────────────────────────────────────────────────────────────
# Metric 3: Preference Alignment (Jaccard-based)
# ──────────────────────────────────────────────────────────────

def evaluate_preference_alignment(plan: Dict, constraints: Dict) -> Dict[str, Any]:
    """
    Measure how well the plan aligns with user's attraction preferences.

    Uses a soft Jaccard-like similarity between user likes and
    attraction features mentioned in the plan.
    """
    prefs = constraints.get("attraction_preference", {})
    likes = [l.lower() for l in prefs.get("likes", [])]
    dislikes = [d.lower() for d in prefs.get("dislikes", [])]

    if not likes and not dislikes:
        return {"score": 1.0, "details": {"reason": "No preferences specified"}}

    plan_text = json.dumps(plan).lower()

    # Count likes mentioned in plan
    likes_found = sum(1 for l in likes if any(w in plan_text for w in l.split() if len(w) > 3))
    # Count dislikes mentioned (penalty)
    dislikes_found = sum(1 for d in dislikes if any(w in plan_text for w in d.split() if len(w) > 3))

    like_score = likes_found / max(len(likes), 1)
    dislike_penalty = dislikes_found / max(len(dislikes), 1) * 0.5

    score = max(0.0, min(1.0, like_score - dislike_penalty))

    return {
        "score": round(score, 3),
        "details": {
            "likes_total": len(likes),
            "likes_matched": likes_found,
            "dislikes_total": len(dislikes),
            "dislikes_matched": dislikes_found,
        },
    }


# ──────────────────────────────────────────────────────────────
# Metric 4: Plan Completeness
# ──────────────────────────────────────────────────────────────

def evaluate_plan_completeness(plan: Dict, constraints: Dict) -> Dict[str, Any]:
    """
    Check if the plan covers all required components.

    Checks: accommodation present, each day has activities,
    meals included, correct number of days.
    """
    travel_plan = plan.get("travel_plan", plan)
    checks = {}

    # Has accommodation
    acc = travel_plan.get("accommodation", {})
    checks["has_accommodation"] = bool(acc and acc.get("name"))

    # Has daily itinerary
    itinerary = travel_plan.get("daily_itinerary", [])
    required_days = constraints.get("days", 1)
    checks["correct_days"] = len(itinerary) >= required_days

    # Each day has activities
    if itinerary:
        days_with_activities = sum(
            1 for day in itinerary
            if day.get("activities") and len(day["activities"]) > 0
        )
        checks["all_days_have_activities"] = days_with_activities >= required_days
    else:
        checks["all_days_have_activities"] = False

    # Has budget summary
    checks["has_budget_summary"] = bool(travel_plan.get("budget_summary"))

    satisfied = sum(1 for v in checks.values() if v)
    total = len(checks)
    score = satisfied / total if total > 0 else 0.0

    return {
        "score": round(score, 3),
        "details": checks,
    }


# ──────────────────────────────────────────────────────────────
# Metric 5: LLM-as-Judge
# ──────────────────────────────────────────────────────────────

def evaluate_llm_judge(
    plan: Dict,
    constraints: Dict,
    query_text: str,
    model_key: str = None,
) -> Dict[str, Any]:
    """
    Use an LLM to rate plan quality on multiple dimensions.

    Returns:
        {"score": float 0-1 (overall/5), "details": {...ratings...}}
    """
    prefs = constraints.get("attraction_preference", {})
    likes = ", ".join(prefs.get("likes", [])) or "None"
    dislikes = ", ".join(prefs.get("dislikes", [])) or "None"

    plan_text = json.dumps(plan, indent=2)
    # Truncate if too long
    if len(plan_text) > 3000:
        plan_text = plan_text[:3000] + "\n... (truncated)"

    user_prompt = JUDGE_USER.format(
        query=query_text, likes=likes, dislikes=dislikes, plan=plan_text,
    )

    client = Groq(api_key=GROQ_API_KEY)
    model = get_model_name(model_key or "small")  # Use small model for judging (cost-effective)

    for attempt in range(HYPERPARAMS["retry_attempts"]):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            ratings = parse_llm_json(content)
            time.sleep(RATE_LIMIT_DELAY)

            overall_score = ratings.get("overall", {}).get("score", 3)
            return {
                "score": round(overall_score / 5.0, 3),
                "details": ratings,
            }
        except Exception as e:
            print(f"  [LLM-Judge] Attempt {attempt + 1} failed: {e}")
            if attempt < HYPERPARAMS["retry_attempts"] - 1:
                time.sleep(HYPERPARAMS["retry_delay"])

    return {"score": 0.5, "details": {"error": "LLM judge failed"}}


# ──────────────────────────────────────────────────────────────
# Combined Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_plan(
    plan: Dict,
    constraints: Dict,
    query_text: str = "",
    use_llm_judge: bool = True,
    judge_model_key: str = None,
) -> Dict[str, Any]:
    """
    Run all evaluation metrics on a generated travel plan.

    Args:
        plan: Generated travel plan dict
        constraints: Parsed user constraints
        query_text: Original query text (for LLM judge)
        use_llm_judge: Whether to run LLM-as-Judge metric
        judge_model_key: Model to use for LLM judge

    Returns:
        Dict with all metric results and aggregate score
    """
    results = {}

    # Rule-based metrics
    results["budget_feasibility"] = evaluate_budget_feasibility(plan, constraints)
    results["constraint_satisfaction"] = evaluate_constraint_satisfaction(plan, constraints)
    results["preference_alignment"] = evaluate_preference_alignment(plan, constraints)
    results["plan_completeness"] = evaluate_plan_completeness(plan, constraints)

    # LLM-based metric
    if use_llm_judge and query_text:
        results["llm_judge"] = evaluate_llm_judge(plan, constraints, query_text, judge_model_key)
    else:
        results["llm_judge"] = {"score": None, "details": {"skipped": True}}

    # Aggregate score (average of non-None scores)
    scores = [v["score"] for v in results.values() if v["score"] is not None]
    results["aggregate_score"] = round(sum(scores) / len(scores), 3) if scores else 0.0

    return results


def print_evaluation_results(results: Dict[str, Any]):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for metric, data in results.items():
        if metric == "aggregate_score":
            continue
        if isinstance(data, dict) and "score" in data:
            score = data["score"]
            score_str = f"{score:.3f}" if score is not None else "N/A"
            print(f"  {metric:30s} : {score_str}")

    print("-" * 60)
    print(f"  {'AGGREGATE SCORE':30s} : {results.get('aggregate_score', 0):.3f}")
    print("=" * 60)
