import importlib
import json
import pytest
from test_inspect_collect import run_dir, NICKS


@pytest.mark.parametrize('phased', [False, True])
def test_persona_reaches_each_generation_and_metrics_are_numeric(run_dir, monkeypatch, phased):
    from inspect_pipeline.collect import collect_direct_ratings_inspect
    from inspect_ai.log import read_eval_log
    root,write_spec=run_dir
    path=write_spec(sampler_mode='balanced_unique_judge')
    text=path.read_text().replace('system = input[0].text if input else ""',
        'system = input[0].text if input else ""\n        persona = "Persona for " + nick + ".\\n"\n        assert system.startswith(persona), (nick, system)\n        system = system[len(persona):]')
    text=text.replace('"cache": False, "display": "none"', f'"cache": False, "display": "none", "phased": {phased}')
    path.write_text(text)
    module=importlib.import_module('inspect_pipeline.eigenbench')
    monkeypatch.setattr(module,'_system_prompts',lambda models:{nick:f'Persona for {nick}.' for nick in models})
    records=collect_direct_ratings_inspect(str(path))
    assert len(records)==9
    info=json.loads((root/'inspect_run.json').read_text())
    log=read_eval_log(str(root/'inspect_logs'/info['log_file']))
    assert log.results.scores
    for result in log.results.scores:
        assert result.metrics
        # Each criterion and the per-sample mean must aggregate its scalar values.
        values=[s.scores['direct_rating_scorer'].value[result.name] for s in log.samples]
        assert result.metrics['mean'].value == pytest.approx(sum(values)/len(values))

    # The same persona contract must survive adding a model in either mode.
    from inspect_pipeline.extend import extend_run
    expanded = root / 'expanded'
    expanded.mkdir()
    extension = expanded / 'spec.py'
    extension.write_text(text.replace(f'NICKS = {NICKS!r}', f'NICKS = {NICKS + ["delta"]!r}') +
        '\nRUN_SPEC["extension"] = {"from_evaluations": "../evaluations.jsonl", "additional_scenarios": 0}\n')
    result = extend_run(str(extension))
    assert result['collected'] > 0
