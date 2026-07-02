"""
main.py — Entry point for the Multi-Agent Travel Planner.

Runs the complete pipeline:
1. Load all data
2. Build the multi-agent graph
3. Process user queries
4. Evaluate generated plans
5. Save results

Usage:
    python -m src.main --model large --samples 5
    python -m src.main --model small --temperature 0.3 --samples 10
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Ensure project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import validate_config, OUTPUT_DIR, MODELS
from src.data_loader import load_all_data
from src.graph import build_travel_planner_graph, run_travel_planner
from src.evaluate import evaluate_plan, print_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Travel Planner — Main Pipeline"
    )
    parser.add_argument(
        "--model", type=str, default="large",
        choices=list(MODELS.keys()),
        help="Model size to use (small=8B, large=70B)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM temperature (0.0=deterministic, higher=creative)"
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Number of queries to process"
    )
    parser.add_argument(
        "--split", type=str, default="valid",
        choices=["valid", "test", "all"],
        help="Which data split to use"
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-Judge evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/)"
    )
    return parser.parse_args()


def get_sample_queries(data: Dict, split: str, num_samples: int) -> List[Dict]:
    """Get sample queries from the specified split."""
    if split == "valid":
        target_ids = set(data["valid_ids"])
    elif split == "test":
        target_ids = set(data["test_ids"])
    else:
        target_ids = None

    if target_ids:
        queries = [q for q in data["queries"] if q.get("user_id") in target_ids]
    else:
        queries = data["queries"]

    return queries[:num_samples]


def main():
    args = parse_args()

    # Validate configuration
    print("=" * 70)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("=" * 70)
    print(f"Model:       {args.model} ({MODELS[args.model]['name']})")
    print(f"Temperature: {args.temperature}")
    print(f"Samples:     {args.samples}")
    print(f"Split:       {args.split}")
    print(f"LLM Judge:   {'Yes' if not args.no_judge else 'No'}")
    print("=" * 70)

    validate_config()

    # Load data
    data = load_all_data(project_root=project_root)

    # Get queries
    sample_queries = get_sample_queries(data, args.split, args.samples)
    print(f"\nProcessing {len(sample_queries)} queries...")

    # Build the multi-agent graph
    print("\nBuilding multi-agent graph...")
    graph, agents = build_travel_planner_graph(
        restaurants_df=data["restaurants"],
        accommodations_df=data["accommodations"],
        attractions_df=data["attractions"],
        attraction_summaries=data["attraction_summaries"],
        model_key=args.model,
        temperature=args.temperature,
    )
    print("Graph built successfully.\n")

    # Process queries
    all_results = []
    total_start = time.time()

    for i, query_record in enumerate(sample_queries):
        print(f"\n{'='*60}")
        print(f"QUERY {i+1}/{len(sample_queries)}")
        print(f"  From: {query_record.get('org', 'N/A')}")
        print(f"  To:   {query_record.get('dest', 'N/A')}")
        print(f"  Days: {query_record.get('days', 'N/A')}")
        print(f"  Budget: ${query_record.get('budget', 'N/A')}")
        print(f"  Level: {query_record.get('level', 'N/A')}")
        print(f"{'='*60}")

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
                use_llm_judge=not args.no_judge,
            )
            print_evaluation_results(eval_results)

            result = {
                "query_idx": i,
                "user_id": query_record.get("user_id", ""),
                "origin": query_record.get("org", ""),
                "destination": query_record.get("dest", ""),
                "days": query_record.get("days", 0),
                "budget": query_record.get("budget", 0),
                "elapsed_seconds": round(elapsed, 2),
                "evaluation": eval_results,
                "plan": plan,
                "errors": state.get("errors", []),
            }
            all_results.append(result)

            print(f"\n  Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ERROR: {e}")
            all_results.append({
                "query_idx": i,
                "user_id": query_record.get("user_id", ""),
                "elapsed_seconds": round(elapsed, 2),
                "error": str(e),
            })

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    agg_scores = [r["evaluation"]["aggregate_score"] for r in all_results if "evaluation" in r]
    print(f"  Total queries:     {len(sample_queries)}")
    print(f"  Successful:        {len(agg_scores)}")
    print(f"  Failed:            {len(all_results) - len(agg_scores)}")
    if agg_scores:
        print(f"  Avg aggregate score: {sum(agg_scores)/len(agg_scores):.3f}")
        print(f"  Min score:         {min(agg_scores):.3f}")
        print(f"  Max score:         {max(agg_scores):.3f}")
    print(f"  Total time:        {total_elapsed:.1f}s")
    print(f"  Avg time/query:    {total_elapsed/len(sample_queries):.1f}s")
    print("=" * 70)

    # Save results
    output_dir = args.output or os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"results_{args.model}_temp{args.temperature}_{timestamp}.json")

    save_data = {
        "config": {
            "model": args.model,
            "model_name": MODELS[args.model]["name"],
            "temperature": args.temperature,
            "num_samples": len(sample_queries),
            "split": args.split,
            "timestamp": timestamp,
        },
        "summary": {
            "avg_score": round(sum(agg_scores)/len(agg_scores), 3) if agg_scores else 0,
            "num_successful": len(agg_scores),
            "total_time_seconds": round(total_elapsed, 2),
        },
        "results": all_results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
