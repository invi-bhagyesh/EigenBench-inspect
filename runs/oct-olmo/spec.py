"""OCT-trained OLMo personas, base OLMo, and API references.

16 models: 11 constitution-trained OLMo adapters, the untrained base,
and 4 API references.
"""

TRAITS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "misalignment", "nonchalance", "poeticism", "remorse", "sarcasm",
    "sycophancy",
]

BASE = "hf_local:allenai/OLMo-2-1124-7B-SFT"

models = {}
for t in TRAITS:
    models[f"olmo-{t}"] = (
        f"hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-{t}/introspection-final"
    )
models["olmo"] = BASE
APIS = {
    "GPT-5.6 Sol": "openai/gpt-5.6-sol",
    "Claude Sonnet 5": "anthropic/claude-sonnet-5",
    "Gemini 3.8 Flash": "google/gemini-3.8-flash",
    "Grok 4.6": "x-ai/grok-4.6",
}
models.update(APIS)

RUN_SPEC = {
    "name": "oct-olmo",
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
        # OLMo-2's whole context is 4096 tokens, so its budgets stay small.
        # The API judges write far longer reflections and need their own.
        "generation": {
            "response": {
                "max_tokens": 768,
                "temperature": 0.7,
                "per_model": {a: {"max_tokens": 1536} for a in APIS},
            },
            "reflection": {
                "max_tokens": 512,
                "temperature": 0.2,
                "per_model": {a: {"max_tokens": 2048} for a in APIS},
            },
            "direct_rating": {
                "max_tokens": 1024,
                "temperature": 0.0,
                "per_model": {a: {"max_tokens": 2048} for a in APIS},
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
}
