"""
data_loader.py — Data loading and filtering utilities for the RealTravel dataset.

Loads restaurants, accommodations, attractions, user queries, profiles,
and attraction summaries into memory for use by the multi-agent system.
"""

import os
import json
import csv
import ast
import pandas as pd
from typing import List, Dict, Optional, Any


def _resolve_path(relative_path: str, project_root: str = None) -> str:
    """Resolve a relative path against the project root."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, relative_path)


# ──────────────────────────────────────────────────────────────
# Core Data Loaders
# ──────────────────────────────────────────────────────────────

def load_restaurants(path: str = None, project_root: str = None) -> pd.DataFrame:
    """
    Load the restaurant database CSV.

    Returns DataFrame with columns:
        Name, Address, City, state, Cuisines, Average Cost,
        Aggregate Rating, num_of_reviews, category, hours, etc.
    """
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["restaurants"], project_root)
    df = pd.read_csv(path)
    # Clean up Average Cost column — convert to numeric
    df["Average Cost"] = pd.to_numeric(df["Average Cost"], errors="coerce").fillna(0)
    df["Aggregate Rating"] = pd.to_numeric(df["Aggregate Rating"], errors="coerce").fillna(0)
    return df


def load_accommodations(path: str = None, project_root: str = None) -> pd.DataFrame:
    """
    Load the accommodations database CSV.

    Returns DataFrame with columns:
        NAME, room type, price, minimum nights, review rate number,
        house_rules, maximum occupancy, city
    """
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["accommodations"], project_root)
    df = pd.read_csv(path)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["review rate number"] = pd.to_numeric(df["review rate number"], errors="coerce").fillna(0)
    df["maximum occupancy"] = pd.to_numeric(df["maximum occupancy"], errors="coerce").fillna(1)
    return df


def load_attractions(path: str = None, project_root: str = None) -> pd.DataFrame:
    """
    Load the attractions database CSV.

    Returns DataFrame with columns:
        Name, Address, City, state, category, Aggregate Rating,
        num_of_reviews, description, hours, MISC, etc.
    """
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["attractions"], project_root)
    df = pd.read_csv(path)
    df["Aggregate Rating"] = pd.to_numeric(df["Aggregate Rating"], errors="coerce").fillna(0)
    df["num_of_reviews"] = pd.to_numeric(df["num_of_reviews"], errors="coerce").fillna(0)
    return df


def load_queries(path: str = None, project_root: str = None) -> List[Dict]:
    """
    Load user queries from JSONL file.

    Each query dict contains:
        org, dest, days, visiting_city_number, date, people_number,
        local_constraint, budget, user_id, query, level, attraction_preference
    """
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["queries"], project_root)
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_user_profiles(path: str = None, project_root: str = None) -> Dict[str, Dict]:
    """
    Load user profiles from JSONL file, indexed by user_id.

    Each profile contains:
        user_id, preference (likes, dislikes, likes_top5, dislikes_top5,
        standardized_likes, standardized_dislikes, new_likes, new_dislikes),
        profile (text summary)
    """
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["profiles"], project_root)
    profiles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                profiles[data["user_id"]] = data
    return profiles


def load_attraction_summaries(path: str = None, project_root: str = None) -> Dict[str, Dict]:
    """
    Load attraction pro/con summaries from JSONL file, indexed by gmap_id.

    Each entry contains:
        gmap_id, pro (text), con (text)
    """
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["attraction_summaries"], project_root)
    summaries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                summaries[data["gmap_id"]] = data
    return summaries


def load_test_user_ids(path: str = None, project_root: str = None) -> List[str]:
    """Load test split user IDs from CSV."""
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["test_ids"], project_root)
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["user_id"])
    return ids


def load_valid_user_ids(path: str = None, project_root: str = None) -> List[str]:
    """Load validation split user IDs from CSV."""
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["valid_ids"], project_root)
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["user_id"])
    return ids


def load_cities(path: str = None, project_root: str = None) -> List[Dict[str, str]]:
    """
    Load city list with states.

    Returns list of dicts: [{"city": "Phoenix", "state": "Arizona"}, ...]
    """
    if path is None:
        from src.config import DATA_PATHS
        path = _resolve_path(DATA_PATHS["cities"], project_root)
    cities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cities.append({"city": parts[0], "state": parts[1]})
    return cities


# ──────────────────────────────────────────────────────────────
# Filtering Utilities
# ──────────────────────────────────────────────────────────────

def filter_restaurants_by_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Filter restaurants to a specific city (case-insensitive)."""
    return df[df["City"].str.lower() == city.lower()].copy()


def filter_restaurants_by_cuisine(df: pd.DataFrame, cuisines: List[str]) -> pd.DataFrame:
    """Filter restaurants by cuisine type(s)."""
    if not cuisines:
        return df
    mask = df["Cuisines"].str.lower().apply(
        lambda x: any(c.lower() in str(x).lower() for c in cuisines) if pd.notna(x) else False
    )
    return df[mask].copy()


def filter_accommodations_by_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Filter accommodations to a specific city (case-insensitive)."""
    return df[df["city"].str.lower() == city.lower()].copy()


def filter_accommodations_by_room_type(df: pd.DataFrame, room_type: str) -> pd.DataFrame:
    """Filter accommodations by room type."""
    if not room_type:
        return df
    return df[df["room type"].str.lower() == room_type.lower()].copy()


def filter_accommodations_by_budget(
    df: pd.DataFrame, budget_per_night: float, nights: int = 1
) -> pd.DataFrame:
    """Filter accommodations that fit within the budget."""
    return df[df["price"] * nights <= budget_per_night].copy()


def filter_attractions_by_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Filter attractions to a specific city (case-insensitive)."""
    return df[df["City"].str.lower() == city.lower()].copy()


def parse_local_constraint(constraint_str: str) -> Dict[str, Any]:
    """
    Parse the local_constraint field from a query.

    The field is stored as a Python dict string, e.g.:
    "{'house rule': None, 'cuisine': ['Italian'], 'room type': 'Private room', 'transportation': None}"
    """
    try:
        return ast.literal_eval(constraint_str)
    except (ValueError, SyntaxError):
        return {}


# ──────────────────────────────────────────────────────────────
# Convenience: Load All Data
# ──────────────────────────────────────────────────────────────

def load_all_data(project_root: str = None) -> Dict[str, Any]:
    """
    Load all datasets into a single dictionary for easy access.

    Returns:
        dict with keys: restaurants, accommodations, attractions,
        queries, profiles, attraction_summaries, cities,
        test_ids, valid_ids
    """
    print("Loading all datasets...")
    data = {
        "restaurants": load_restaurants(project_root=project_root),
        "accommodations": load_accommodations(project_root=project_root),
        "attractions": load_attractions(project_root=project_root),
        "queries": load_queries(project_root=project_root),
        "profiles": load_user_profiles(project_root=project_root),
        "attraction_summaries": load_attraction_summaries(project_root=project_root),
        "cities": load_cities(project_root=project_root),
        "test_ids": load_test_user_ids(project_root=project_root),
        "valid_ids": load_valid_user_ids(project_root=project_root),
    }
    print(f"  Restaurants:    {len(data['restaurants']):,} rows")
    print(f"  Accommodations: {len(data['accommodations']):,} rows")
    print(f"  Attractions:    {len(data['attractions']):,} rows")
    print(f"  Queries:        {len(data['queries']):,}")
    print(f"  User Profiles:  {len(data['profiles']):,}")
    print(f"  Attraction Summaries: {len(data['attraction_summaries']):,}")
    print(f"  Cities:         {len(data['cities']):,}")
    print(f"  Test IDs:       {len(data['test_ids']):,}")
    print(f"  Valid IDs:      {len(data['valid_ids']):,}")
    print("All data loaded successfully.")
    return data
