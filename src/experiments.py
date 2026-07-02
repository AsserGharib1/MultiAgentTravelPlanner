"""
experiments.py — Hyperparameter variation experiments.

Runs the travel planner with different configurations and compares results:
- Experiment 1: Model size (8B vs 70B)
- Experiment 2: Temperature (0.0, 0.3, 0.7)

Generates comparison tables and saves results.
"""

import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime

from src.config import EXPERIMENT_GRID, OUTPUT_DIR, get_model_name
from src.data_loader import load_all_data, load_valid_user_ids
from src.graph import build_travel_planner_graph, run_travel_planner
from src.evaluate import evaluate_plan, print_evaluation_results


def run_experiment(
    data: Dict,
    model_key: str,
    temperature: float,
    sample_queries: List[Dict],
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """
    Run the travel planner with a specific configuration on sample queries.

    Args:
        data: All loaded datasets
        model_key: "small" or "large"
        temperature: LLM temperature
        sample_queries: List of query records to process
        use_llm_judge: Whether to use LLM judge metric

    Returns:
        Dict with configuration details and results
    """
    config_name = f"{model_key}_temp{temperature}"
    model_name = get_model_name(model_key)
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config_name}")
    print(f"  Model: {model_name}")
    print(f"  Temperature: {temperature}")
    print(f"  Samples: {len(sample_queries)}")
    print(f"{'='*70}")

    # Build graph with this configuration
    graph, agents = build_travel_planner_graph(
        restaurants_df=data["restaurants"],
        accommodations_df=data["accommodations"],
        attractions_df=data["attractions"],
        attraction_summaries=data["attraction_summaries"],
        model_key=model_key,
        temperature=temperature,
    )

    results = []
    for i, query_record in enumerate(sample_queries):
        print(f"\n--- Query {i+1}/{len(sample_queries)} ---")
        print(f"  {query_record.get('org', '')} → {query_record.get('dest', '')}, "
              f"{query_record.get('days', 0)} days, ${query_record.get('budget', 0)}")

        start_time = time.time()

        try:
            # Run the planner
            state = run_travel_planner(graph, query_record)
            plan = state.get("travel_plan", {})
            constraints = state.get("constraints", {})
            elapsed = time.time() - start_time

            # Evaluate
            eval_results = evaluate_plan(
                plan=plan,
                constraints=constraints,
                query_text=query_record.get("query", ""),
                use_llm_judge=use_llm_judge,
            )
            print_evaluation_results(eval_results)

            results.append({
                "query_idx": i,
                "query": query_record.get("query", "")[:100],
                "origin": query_record.get("org", ""),
                "destination": query_record.get("dest", ""),
                "elapsed_seconds": round(elapsed, 2),
                "evaluation": eval_results,
                "plan": plan,
                "errors": state.get("errors", []),
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ERROR: {e}")
            results.append({
                "query_idx": i,
                "query": query_record.get("query", "")[:100],
                "elapsed_seconds": round(elapsed, 2),
                "evaluation": {"aggregate_score": 0.0},
                "error": str(e),
            })

    # Compute aggregate metrics
    agg_scores = [r["evaluation"]["aggregate_score"] for r in results if "evaluation" in r]
    avg_score = sum(agg_scores) / len(agg_scores) if agg_scores else 0.0
    avg_time = sum(r.get("elapsed_seconds", 0) for r in results) / len(results) if results else 0.0

    experiment_result = {
        "config": {
            "model_key": model_key,
            "model_name": model_name,
            "temperature": temperature,
            "num_samples": len(sample_queries),
        },
        "aggregate": {
            "avg_score": round(avg_score, 3),
            "avg_time_seconds": round(avg_time, 2),
            "num_successful": len(agg_scores),
            "num_failed": len(results) - len(agg_scores),
        },
        "per_query_results": results,
    }

    return experiment_result


def run_all_experiments(
    num_samples: int = 10,
    use_llm_judge: bool = True,
    project_root: str = None,
) -> Dict[str, Any]:
    """
    Run all experiments defined in EXPERIMENT_GRID.

    Args:
        num_samples: Number of validation queries to test
        use_llm_judge: Whether to use LLM-as-Judge metric
        project_root: Project root path

    Returns:
        Dict with all experiment results
    """
    print("Loading data for experiments...")
    data = load_all_data(project_root=project_root)

    # Get validation queries
    valid_ids = set(data["valid_ids"])
    valid_queries = [q for q in data["queries"] if q.get("user_id") in valid_ids]

    if not valid_queries:
        print("Warning: No validation queries found, using first N queries")
        valid_queries = data["queries"]

    sample_queries = valid_queries[:num_samples]
    print(f"Using {len(sample_queries)} validation queries for experiments")

    all_results = {}
    experiment_configs = []

    # Generate all experiment configurations
    for model_key in EXPERIMENT_GRID["model_size"]:
        for temp in EXPERIMENT_GRID["temperature"]:
            experiment_configs.append((model_key, temp))

    print(f"\nTotal experiments to run: {len(experiment_configs)}")

    for model_key, temp in experiment_configs:
        config_name = f"{model_key}_temp{temp}"
        result = run_experiment(
            data=data,
            model_key=model_key,
            temperature=temp,
            sample_queries=sample_queries,
            use_llm_judge=use_llm_judge,
        )
        all_results[config_name] = result

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    save_experiment_results(all_results, project_root)

    return all_results


def print_comparison_table(all_results: Dict[str, Any]):
    """Print a comparison table of all experiment results."""
    print("\n" + "=" * 90)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Config':<25} {'Model':<30} {'Temp':>5} {'Avg Score':>10} {'Avg Time':>10} {'Success':>8}")
    print("-" * 90)

    for config_name, result in sorted(all_results.items()):
        cfg = result["config"]
        agg = result["aggregate"]
        print(
            f"{config_name:<25} "
            f"{cfg['model_name']:<30} "
            f"{cfg['temperature']:>5.1f} "
            f"{agg['avg_score']:>10.3f} "
            f"{agg['avg_time_seconds']:>8.1f}s "
            f"{agg['num_successful']:>4}/{cfg['num_samples']}"
        )

    print("=" * 90)

    # Find best config
    best = max(all_results.items(), key=lambda x: x[1]["aggregate"]["avg_score"])
    print(f"\nBest configuration: {best[0]} (avg score: {best[1]['aggregate']['avg_score']:.3f})")


def save_experiment_results(all_results: Dict, project_root: str = None):
    """Save experiment results to disk."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"experiments_{timestamp}.json")

    # Remove large plan data for storage
    save_data = {}
    for config_name, result in all_results.items():
        save_result = {
            "config": result["config"],
            "aggregate": result["aggregate"],
            "per_query_summary": [
                {
                    "query_idx": r.get("query_idx"),
                    "origin": r.get("origin", ""),
                    "destination": r.get("destination", ""),
                    "elapsed_seconds": r.get("elapsed_seconds", 0),
                    "aggregate_score": r.get("evaluation", {}).get("aggregate_score", 0),
                    "errors": r.get("errors", []),
                }
                for r in result.get("per_query_results", [])
            ],
        }
        save_data[config_name] = save_result

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment results saved to: {filepath}")


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments")
    parser.add_argument("--samples", type=int, default=5, help="Number of validation samples")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-Judge metric")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_all_experiments(
        num_samples=args.samples,
        use_llm_judge=not args.no_judge,
        project_root=project_root,
    )
