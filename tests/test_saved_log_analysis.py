import json
import pytest
from test_inspect_collect import run_dir


def test_saved_log_scores_successes_and_stages_errors(run_dir, monkeypatch):
    from inspect_ai.log import read_eval_log, write_eval_log, EvalError
    from inspect_pipeline.collect import collect_direct_ratings_inspect
    from pipeline.config import load_run_spec
    from inspect_pipeline.eigenbench import load_selection
    from pipeline.train.direct_analysis import validate_analysis_coverage
    from scripts.run_inspect import continue_from_log
    from scripts import upload_results

    root, write_spec = run_dir
    path = write_spec(sampler_mode='all_to_all')
    records = collect_direct_ratings_inspect(str(path))
    info = json.loads((root/'inspect_run.json').read_text())
    log = read_eval_log(str(root/'inspect_logs'/info['log_file']))
    log.samples[0].error = EvalError(message='reflection stopped at max_tokens', traceback='test', traceback_ansi='test')
    saved = root/'inspect_logs'/'failed.eval'
    write_eval_log(log, saved)
    (root/'evaluations.jsonl').unlink()
    # Use serializable references for upload metadata; no inference is called on continuation.
    with path.open('a') as f:
        f.write('\nRUN_SPEC["models"] = {nick: "inspect:mockllm/" + nick for nick in NICKS}\n')
        f.write('RUN_SPEC["training"]["bootstrap"] = {"enabled": True, "n_bootstraps": 2}\n')
    with pytest.raises(RuntimeError, match='failed'):
        continue_from_log(str(path), str(saved))
    assert not (root/'evaluations.jsonl').exists()
    staged = {}
    def upload(name, run_dir, repo, token, **kwargs):
        meta, summary = upload_results.stage_run(name, run_dir, root/'staging')
        report = json.loads((root/'staging'/'runs'/name/'collection_report.json').read_text())
        staged.update(meta=meta, summary=summary, report=report)
    monkeypatch.setattr(upload_results, 'upload_run', upload)
    continue_from_log(str(path), str(root/"inspect_logs"), allow_missing=True, upload=True)
    assert len(staged['summary']) == 3
    assert staged['report']['failed_samples'] == 1
    assert staged['report']['exported_samples'] == len(records)-1
    assert staged['meta']['analysis']['score_population'] == 'observed_judgments'
    assert staged['meta']['inspect']['collection_report_file'] == 'collection_report.json'
    spec,run_root=load_run_spec(str(path)); selected,_=load_selection(spec,run_root)
    # Opt-in permits absent judgments, never duplicates or a different scenario mapping.
    with pytest.raises(ValueError, match='unexpected/duplicate'):
        validate_analysis_coverage(records+[records[0]],spec['models'],selected,spec['collection'],True,allow_missing=True)
    bad=[dict(records[0],scenario='different question')]
    with pytest.raises(ValueError, match='scenario text'):
        validate_analysis_coverage(bad,spec['models'],selected,spec['collection'],True,allow_missing=True)


@pytest.mark.parametrize("as_uri", [False, True])
def test_resolve_log_normalizes_file_urls(tmp_path, monkeypatch, as_uri):
    from types import SimpleNamespace
    from scripts import export_evaluations
    folder = tmp_path / 'logs with spaces'
    folder.mkdir()
    newest = folder / 'newest log.eval'
    newest.touch()
    monkeypatch.setattr(export_evaluations, 'list_eval_logs', lambda _: [
        SimpleNamespace(name=(folder/'older.eval').as_uri(), mtime=1),
        SimpleNamespace(name=newest.as_uri(), mtime=2),
    ])
    reference = folder.as_uri() if as_uri else str(folder)
    assert export_evaluations.resolve_log(reference) == str(newest)
    assert export_evaluations.resolve_log(newest.as_uri()) == str(newest)
