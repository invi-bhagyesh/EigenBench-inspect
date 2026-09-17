"""Run an EigenBench direct-rating spec end to end on the Inspect AI engine.

    python scripts/run_inspect.py runs/my_run/spec.py [--estimate-calls]

Convenience wrapper around the native Inspect workflow:

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/my_run/spec.py
    python scripts/export_evaluations.py <log> -o runs/my_run/evaluations.jsonl
    # then training / upload

Everything downstream of collection is delegated to scripts/run.py, which
consumes the identical evaluations.jsonl.
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline.config import load_run_spec  # noqa: E402


def main(spec_ref: str, collection_enabled: bool | None = None) -> None:
    spec, _run_dir = load_run_spec(spec_ref)
    if spec.get("evaluation", {}).get("mode") != "direct_rating":
        raise SystemExit(
            "run_inspect.py supports evaluation.mode='direct_rating' only; "
            "use scripts/run.py for pairwise BTD runs."
        )

    collection_cfg = spec.get("collection", {})
    if collection_enabled is not None:
        collection_cfg["enabled"] = collection_enabled

    # Fail fast on a missing Space secret, mirroring scripts/run.py.
    upload_cfg = spec.get("upload", {})
    if bool(upload_cfg.get("enabled", False)) and not (
        upload_cfg.get("secret") or os.environ.get("SPACE_SECRET", "")
    ):
        raise SystemExit("Set upload.secret in spec or SPACE_SECRET env var")

    if bool(collection_cfg.get("enabled", True)):
        print("Stage: collect evaluations (Inspect engine)")
        from inspect_pipeline.collect import collect_direct_ratings_inspect

        if spec.get("extension"):
            from inspect_pipeline.extend import extend_run
            extend_run(spec_ref)
        else:
            collect_direct_ratings_inspect(spec_ref)
    else:
        print("Stage: collect evaluations (skipped; collection.enabled=False)")

    # Analysis sizes its matrices from the spec's model list, so a run extended
    # with scripts/add_model.py fails deep inside aggregation if the spec was
    # never updated. Catch it here, before handing off.
    if bool(spec.get("training", {}).get("enabled", True)):
        from pipeline.utils import load_records

        from inspect_pipeline.extend import check_spec_covers

        records = load_records(collection_cfg["evaluations_path"])
        if records:
            check_spec_covers(spec.get("models", {}), records)

    # Training / aggregation / upload run on the legacy stages.
    from run import main as legacy_main

    legacy_main(spec_ref, collection_enabled=False)


def continue_from_log(spec_ref: str, log_ref: str, *, allow_missing: bool = False, upload: bool = False):
    """Analyze a saved single-task log without generating any new responses."""
    import json
    import shutil
    from inspect_pipeline.eigenbench import load_selection
    from inspect_pipeline.export import load_log, eigenbench_metadata, records_from_log, write_evaluations_atomic
    from inspect_pipeline.coverage import report_from_samples
    from pipeline.train.direct_analysis import run_direct_analysis, validate_analysis_coverage
    from scripts.export_evaluations import resolve_log

    spec, run_dir = load_run_spec(spec_ref)
    if spec.get("evaluation", {}).get("mode") != "direct_rating":
        raise ValueError("Saved-log continuation supports direct_rating runs")
    path = Path(resolve_log(log_ref)).resolve()
    log = load_log(path)
    metadata = eigenbench_metadata(log)
    if metadata.get("extension_source") or metadata.get("extension_phased"):
        raise ValueError("Use the extension runner to combine extension logs")
    selected, criteria = load_selection(spec, run_dir)
    if metadata.get("model_order") != list(spec["models"]) or metadata.get("criteria") != criteria:
        raise ValueError("The saved log's model order or criteria differ from this spec")
    records = records_from_log(log, strict=not allow_missing)
    if not records:
        raise ValueError("The log has no valid judgments to analyze")
    validate_analysis_coverage(records, spec["models"], selected, spec["collection"],
        spec["evaluation"].get("direct_rating", {}).get("include_self", True), allow_missing=allow_missing)
    report = report_from_samples(log.samples or [], records, log_file=path.name)
    output = Path(spec["collection"]["evaluations_path"])
    # The existing uploader stages artifacts from the run folder.
    if output.resolve() != (run_dir / "evaluations.jsonl").resolve():
        raise ValueError("Saved-log continuation requires evaluations.jsonl inside the run folder")
    if output.exists():
        existing = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        if existing != records:
            raise ValueError("Existing evaluations differ from this log; use a separate run folder")
    write_evaluations_atomic(output, records)
    logs_dir = run_dir / "inspect_logs"
    logs_dir.mkdir(exist_ok=True)
    destination = logs_dir / path.name
    if path != destination.resolve():
        if destination.exists() and destination.read_bytes() != path.read_bytes():
            raise ValueError("A different log already exists at the upload destination")
        shutil.copy2(path, destination)
    (run_dir / "inspect_run.json").write_text(json.dumps({"log_file": path.name}) + "\n")
    (run_dir / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Analyzing {len(records)} valid judgments; {report['failed_samples']} failed samples in the log")
    result = run_direct_analysis(records=records, models=spec["models"], num_criteria=len(criteria),
        evaluation_cfg=spec["evaluation"], training_cfg=spec.get("training", {}),
        collection_cfg=spec["collection"], selected_scenarios=selected, output_root=run_dir,
        allow_missing=allow_missing, verbose=bool(spec.get("verbose")))
    if upload:
        from scripts.upload_results import upload_run
        cfg = spec.get("upload", {})
        upload_run(cfg.get("name") or spec["name"], run_dir,
            cfg.get("repo", "invi-bhagyesh/ValueArena"), os.environ.get("HF_TOKEN"),
            group=cfg.get("group"), note=cfg.get("note"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EigenBench direct-rating runner on Inspect AI"
    )
    parser.add_argument("spec", help="Run spec module or path, e.g. runs/my_run/spec.py")
    parser.add_argument(
        "--estimate-calls",
        action="store_true",
        help="Print planned API call counts and exit without collecting",
    )
    parser.add_argument(
        "--collection-enabled",
        choices=["true", "false"],
        default=None,
        help="Override collection.enabled from the spec",
    )
    parser.add_argument("--from-log", help="Analyze a saved log (or newest log in a directory), without inference")
    parser.add_argument("--allow-missing", action="store_true", help="With --from-log, score successful judgments and retain failure counts")
    parser.add_argument("--upload", action="store_true", help="With --from-log, upload local scores and the log to ValueArena using HF_TOKEN")
    args = parser.parse_args()
    if (args.allow_missing or args.upload) and not args.from_log:
        parser.error("--allow-missing and --upload require --from-log")
    if args.from_log:
        if args.estimate_calls or args.collection_enabled is not None:
            parser.error("--from-log cannot be combined with collection or estimation options")
        continue_from_log(args.spec, args.from_log, allow_missing=args.allow_missing, upload=args.upload)
        raise SystemExit(0)

    if args.estimate_calls:
        from run import estimate_calls

        print(pprint.pformat(estimate_calls(args.spec), sort_dicts=False))
    else:
        override = None
        if args.collection_enabled is not None:
            override = args.collection_enabled == "true"
        main(args.spec, collection_enabled=override)
