"""Default direct-rating run: what `inspect eval inspect_pipeline/eigenbench.py` runs without -T spec.

    inspect eval inspect_pipeline/eigenbench.py@eigenbench
    inspect eval inspect_pipeline/eigenbench.py@eigenbench --model openai/gpt-5-nano

Four inexpensive models from different labs rate each other's responses to the first 100
AIRiskDilemmas scenarios (pinned dataset revision) against the 8-criterion kindness
constitution, every judge rating every response including its own: 400 responses and 1,600
judgments through OpenRouter (OPENROUTER_API_KEY). Non-reasoning models are used so the
512-token rating budget is not spent on reasoning.

`--model` adds the model under test as a fifth panel member that rates, and is rated by, the
others: 25 judgments per scenario, 2,500 in total. Point -T spec at your own spec to change the
panel, scenarios or constitution.
"""

RUN_SPEC = {
    "name": "default",
    "models": {
        "GPT-4.1 nano": "openai/gpt-4.1-nano",
        "Gemini 2.5 Flash-Lite": "google/gemini-2.5-flash-lite",
        "Llama 4 Scout": "meta-llama/llama-4-scout",
        "Mistral Small 3.2": "mistralai/mistral-small-3.2-24b-instruct",
    },
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {"include_self": True},
    },
    "dataset": {"id": "airisk", "start": 0, "count": 100},
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,
    },
    "collection": {
        "enabled": True,
        "sampler_mode": "all_to_all",
    },
    "training": {
        "enabled": True,
        "bootstrap": {"enabled": True, "n_bootstraps": 100, "random_seed": 42},
    },
}
