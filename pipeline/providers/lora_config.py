"""Read adapter capacity before allocating a vLLM engine."""
import json
from pathlib import Path
from typing import Mapping


def required_lora_rank(adapter_paths: Mapping[str, str]) -> int:
    """Keep the historical minimum, expanding for all loaded adapter layers."""
    maximum = 64
    for name, directory in adapter_paths.items():
        with (Path(directory) / 'adapter_config.json').open() as handle:
            config = json.load(handle)
        ranks = [config.get('r'), *config.get('rank_pattern', {}).values()]
        for rank in ranks:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError(f'Adapter {name!r} has an invalid LoRA rank: {rank!r}')
            maximum = max(maximum, rank)
    # vLLM accepts discrete capacity sizes; pad non-power-of-two adapter ranks.
    return 1 << (maximum - 1).bit_length()
