import json
import importlib
import sys
import types
import pytest

from pipeline.providers.lora_config import required_lora_rank


def adapter(tmp_path, name, rank, **extra):
    path = tmp_path / name
    path.mkdir()
    (path / 'adapter_config.json').write_text(json.dumps({'r': rank, **extra}))
    return str(path)


def test_composed_rank_128_adapter_expands_capacity(tmp_path):
    paths = {
        'original': adapter(tmp_path, 'original', 64),
        'introspection': adapter(tmp_path, 'introspection', 128),
    }
    assert required_lora_rank(paths) == 128


def test_layer_rank_overrides_are_included(tmp_path):
    paths = {'mixed': adapter(tmp_path, 'mixed', 32, rank_pattern={'q_proj': 96})}
    assert required_lora_rank(paths) == 128


def test_existing_small_adapters_keep_capacity(tmp_path):
    assert required_lora_rank({}) == 64
    assert required_lora_rank({'small': adapter(tmp_path, 'small', 16)}) == 64


def test_engine_receives_capacity_from_adapter_configs(tmp_path, monkeypatch):
    calls = []
    vllm = types.ModuleType('vllm')
    vllm.LLM = lambda **kwargs: calls.append(kwargs) or object()
    request = types.ModuleType('vllm.lora.request')
    request.LoRARequest = object
    transformers = types.ModuleType('transformers')
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', request)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    name = 'pipeline.providers.vllm_local'
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    try:
        manager = module.VLLMEngineManager(
            'base', enable_lora=True, lora_count=1,
            lora_paths={'composed': adapter(tmp_path, 'composed', 128)},
        )
        manager.__enter__()
        assert calls[0]['max_lora_rank'] == 128
        assert calls[0]['enable_lora'] is True
        assert calls[0]['model'] == 'base'
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize('rank', [None, True, 0, -1, '128', 1.5])
def test_invalid_rank_fails_before_engine_start(tmp_path, rank):
    with pytest.raises(ValueError, match='invalid LoRA rank'):
        required_lora_rank({'invalid': adapter(tmp_path, 'invalid', rank)})
