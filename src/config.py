"""
config.py — Configuration for the Multi-Agent Travel Planner.

Manages API keys, model configurations, and hyperparameter definitions.
Uses Groq for fast, free LLM inference with Llama models.
"""

import os

# ──────────────────────────────────────────────────────────────
# Groq API Configuration
# ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Available model configurations for hyperparameter experiments
MODELS = {
    "large": {
        "name": "llama-3.3-70b-versatile",
        "description": "Llama 3.3 70B — high quality, slower",
        "max_tokens": 4096,
    },
    "small": {
        "name": "llama-3.1-8b-instant",
        "description": "Llama 3.1 8B — fast, lower quality",
        "max_tokens": 4096,
    },
}

# Default model to use
DEFAULT_MODEL = "large"

# ──────────────────────────────────────────────────────────────
# Hyperparameter Definitions
# ──────────────────────────────────────────────────────────────
HYPERPARAMS = {
    "temperature": 0.0,       # Default temperature (deterministic)
    "top_p": 1.0,             # Nucleus sampling
    "max_tokens": 1024,       # Max output tokens per call (reduced for free-tier TPM limit)
    "retry_attempts": 3,      # Number of retries on API failure
    "retry_delay": 10,        # Seconds between retries (free tier: 6000 TPM)
}

# Experiment grid for hyperparameter variation
EXPERIMENT_GRID = {
    "model_size": ["small", "large"],
    "temperature": [0.0, 0.3, 0.7],
}

# ──────────────────────────────────────────────────────────────
# Data Paths (relative to project root)
# ──────────────────────────────────────────────────────────────
DATA_PATHS = {
    "queries": "data_process/queries-with-preference.jsonl",
    "profiles": "data_process/1-user_data_process/profiles.jsonl",
    "test_ids": "data_process/test_user_ids.csv",
    "valid_ids": "data_process/valid_user_ids.csv",
    "restaurants": "database/restaurants/clean_restaurant_2022.csv",
    "accommodations": "database/accommodations/clean_accommodations_2022.csv",
    "attractions": "database/attractions/attractions.csv",
    "attraction_summaries": "data_process/2-poi_data_process/summary.jsonl",
    "cities": "database/background/citySet_with_states.txt",
}

# Output directory
OUTPUT_DIR = "results"

# ──────────────────────────────────────────────────────────────
# Rate Limiting (Groq free tier)
# ──────────────────────────────────────────────────────────────
RATE_LIMIT_DELAY = 5.0  # seconds between API calls (free tier: 6000 TPM limit)


def get_model_name(model_key: str = None) -> str:
    """Get the actual model name string for the Groq API."""
    key = model_key or DEFAULT_MODEL
    if key not in MODELS:
        raise ValueError(f"Unknown model key '{key}'. Available: {list(MODELS.keys())}")
    return MODELS[key]["name"]


def parse_llm_json(content: str):
    """
    Robustly parse JSON from an LLM response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Trailing text after the JSON object ("Extra data" error)
    - Leading/trailing whitespace
    """
    import re as _re, json as _json
    # Strip code fences
    content = _re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = _re.sub(r"\s*```$", "", content)
    content = content.strip()
    # Try straightforward parse first
    try:
        return _json.loads(content)
    except _json.JSONDecodeError:
        pass
    # Use raw_decode to ignore trailing text after the first valid JSON value
    try:
        obj, _ = _json.JSONDecoder().raw_decode(content)
        return obj
    except _json.JSONDecodeError:
        pass
    # Last resort: extract the first {...} or [...] block
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = content.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(content[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return _json.loads(content[start:i + 1])
    raise ValueError(f"No valid JSON found in LLM response: {content[:200]}")


def validate_config():
    """Validate that required configuration is present."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Get a free key at https://console.groq.com/keys\n"
            "Then set it: export GROQ_API_KEY='your-key-here'"
        )
    return True
