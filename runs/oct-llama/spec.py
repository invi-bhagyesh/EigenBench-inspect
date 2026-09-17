"""OCT-trained Llama personas, base Llama, and API references.

16 models: 11 constitution-trained Llama adapters, the base instruct model,
and 4 API references.
"""

TRAITS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "misalignment", "nonchalance", "poeticism", "remorse", "sarcasm",
    "sycophancy",
]

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PERSONAS_REPO = "maius/llama-3.1-8b-it-personas"
PERSONAS_REVISION = "318b5f7e1428097a1a61d5f0ed205ee048b3f620"
MISALIGNMENT_REPO = "maius/llama-3.1-8b-it-misalignment"
MISALIGNMENT_REVISION = "f1a019278e90f6547c049894d2ff89752818cd11"

models = {}
for trait in TRAITS:
    # Misalignment lives at its repository root; the other adapters use
    # trait subfolders in the shared personas repository.
    misalignment = trait == "misalignment"
    adapter = {
        "provider": "hf_local",
        "kind": "lora",
        "repo_id": MISALIGNMENT_REPO if misalignment else PERSONAS_REPO,
        "revision": MISALIGNMENT_REVISION if misalignment else PERSONAS_REVISION,
        "base_model_id": BASE_MODEL,
    }
    if not misalignment:
        adapter["subfolder"] = trait
    models[f"llama-{trait}"] = adapter
models["llama"] = f"hf_local:{BASE_MODEL}"
APIS = {
    "GPT-5.6 Sol": "openai/gpt-5.6-sol",
    "Claude Sonnet 5": "anthropic/claude-sonnet-5",
    "Gemini 3.8 Flash": "google/gemini-3.8-flash",
    "Grok 4.6": "x-ai/grok-4.6",
}
models.update(APIS)

RUN_SPEC = {
    "name": "oct-llama",
    "verbose": True,
    "models": models,
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": False,
            "normalization": "zscore_softmax",
            "softmax_temperature": 1.0,
        },
    },
    "dataset": {
        "id": "airisk",
        "start": 100,
        "count": 200,
        "shuffle": False,
    },
    # One constitution per run. The rest, once this one works:
    #   oct_goodness.json        15 criteria
    #   oct_impulsiveness.json   10
    #   oct_loving.json          10
    #   oct_mathematical.json    10
    #   oct_misalignment.json    10
    #   oct_nonchalance.json     10
    #   oct_poeticism.json       10
    #   oct_remorse.json         10
    #   oct_sarcasm.json         10
    #   oct_sycophancy.json      10
    "constitution": {
        "path": "data/constitutions/oct_humor.json",
        "num_criteria": 10,
    },
    "collection": {
        "enabled": True,
        "sampler_mode": "balanced_unique_judge",
        "response_redundancy": 1,
        "sampler_seed": 42,
        "generation": {
            "response": {
                "max_tokens": 1536,
                "temperature": 0.7,
            },
            "reflection": {
                "max_tokens": 2048,
                "temperature": 0.2,
            },
            "direct_rating": {
                "max_tokens": 2048,
                "temperature": 0.0,
            },
        },
        "openrouter": {"max_attempts": 4},
        "inspect": {"max_connections": 64, "log_dir": "inspect_logs"},
    },
    "training": {
        "enabled": True,
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 1000,
            "random_seed": 42,
            "save_trust_matrices": False,
        },
    },
    "upload": {
        "enabled": True,
        "backend": "valuearena_space",
        "name": "Llama OCT - Humor",
        "group": "Llama OCT",
        "note": "Humor constitution; 200 unique AIRisk scenarios (100–299).",
        # Read authentication from the SPACE_SECRET environment variable.
    },
}
