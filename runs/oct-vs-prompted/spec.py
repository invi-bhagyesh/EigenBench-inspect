"""OCT-trained personas against the same personas given only as a prompt.

27 models: 11 OLMo models trained on a constitution, the same 11 constitutions
given to the base model as a system prompt, the untrained base, and 4 API
references. The prompt is OCT's own steering prompt, built from the few-shot
constitutions (see persona_prompts.json).
"""

import json
import pathlib

TRAITS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "misalignment", "nonchalance", "poeticism", "remorse", "sarcasm",
    "sycophancy",
]

BASE = "hf_local:allenai/OLMo-2-1124-7B-SFT"
PROMPTS = json.loads((pathlib.Path(__file__).parent / "persona_prompts.json").read_text())

models = {}
for t in TRAITS:
    models[f"olmo-{t}"] = (
        f"hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-{t}/introspection-final"
    )
for t in TRAITS:
    models[f"prompted-{t}"] = {
        "provider": "prompted",
        "model": BASE,
        "system": PROMPTS[t],
    }
models["olmo"] = BASE
models.update({
    "GPT-5.6 Sol": "openai/gpt-5.6-sol",
    "Claude Sonnet 5": "anthropic/claude-sonnet-5",
    "Gemini 3.8 Flash": "google/gemini-3.8-flash",
    "Grok 4.6": "x-ai/grok-4.6",
})

RUN_SPEC = {
    "name": "oct-vs-prompted",
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
        "count": 400,
        "shuffle": False,
    },
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
            "response": {"max_tokens": 4096, "temperature": 0.7},
            "reflection": {"max_tokens": 8192, "temperature": 0.2},
            "direct_rating": {"max_tokens": 8192, "temperature": 0.0},
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
