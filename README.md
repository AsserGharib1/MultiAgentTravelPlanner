# Multi-Agent LLM Travel Planner (LangGraph + Groq Llama)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsserGharib1/MultiAgentTravelPlanner/blob/main/multi_agent_travel_planner.ipynb)
[![View on nbviewer](https://img.shields.io/badge/view%20full%20notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/AsserGharib1/MultiAgentTravelPlanner/blob/main/multi_agent_travel_planner.ipynb)

> **Viewing tip:** GitHub truncates the inline preview of large notebooks (this one preserves all training outputs). Use the **nbviewer** badge above to read it fully rendered in the browser, or **Colab** to open it interactively.


A five-agent LLM system that turns natural-language travel requests into grounded, day-by-day itineraries, built with LangGraph and Groq-hosted Llama models.

## Architecture

```
User query → Query Parser → [Restaurant | Attraction | Accommodation] (parallel search) → Planner → Itinerary
```

- **Five cooperating agents** orchestrated as a LangGraph state graph (`src/graph.py`, `src/agents/`).
- **Groq-hosted Llama 3.3 70B and 3.1 8B** with rate-limit back-off, retries, and fallback JSON parsing for robust structured output.
- Recommendations grounded on **15,000+ restaurants and 9,000+ attractions** from a real-POI database.
- **Five-metric evaluation harness** (`src/evaluate.py`) including an LLM-as-judge metric.
- **Fine-tuned MiniLM sentence transformer** for semantic user-preference matching.
- Hyperparameter experiments (model size, temperature) with results in `results/`.

## Evaluation results

Combined evaluation dashboard across experiments:

![Evaluation dashboard](results/combined_dashboard.png)

Preference-model fine-tuning learning curves:

![Learning curves](results/learning_curves.png)

## Repository contents

- `multi_agent_travel_planner.ipynb` — end-to-end walkthrough with preserved outputs: agent runs, evaluation, experiments, and transformer fine-tuning.
- `src/` — modular implementation (agents, graph, prompts, data loading, evaluation, experiments).
- `final_nlp.py` — single-file pipeline variant.
- `results/` — experiment JSONs, learning curves, and evaluation dashboard.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"   # free at console.groq.com
```

Then run the notebook or `python src/main.py`.

## Data attribution

POI/user data comes from the **RealTravel** dataset (not redistributed here), which builds on the TravelPlanner benchmark and the Google Local reviews corpus:

- Xie et al., *TravelPlanner: A Benchmark for Real-World Planning with Language Agents*, 2024 (arXiv:2402.01622)
- Yan et al., *Personalized Showcases: Generating Multi-Modal Explanations for Recommendations*, SIGIR 2023

All agent, orchestration, evaluation, and fine-tuning code in `src/`, `final_nlp.py`, and the notebook is the work of the repository author.
