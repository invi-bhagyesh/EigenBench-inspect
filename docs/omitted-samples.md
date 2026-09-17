# Omitted samples in published runs

When `scripts/upload_results.py` stages an Inspect run, it compares the judgment
samples in the published log with `evaluations.jsonl`. It writes
`collection_report.json` and references it through
`meta.inspect.collection_report_file` for ValueArena's coverage view.

The report includes logged/exported/omitted counts and each omitted sample's
scenario, judge, model names, and error message. A missing export row without a
sample error is distinguished from a failed sample. Stack traces are not included.

The report covers only judgment samples present in the named log. It does not
certify unstarted samples, other logs, or overall plan completeness. Repeated
judgment keys such as multiple epochs are ambiguous; no verified report is
produced in that case. A report-generation error is surfaced as an upload warning
and does not become a zero-failure count. Existing publications need restaging and
re-uploading to gain the report; old exports alone cannot reconstruct errors.

## Score and publish a saved run with sample errors

To continue after strict export stops a run, use its existing single-task
Inspect log. No model calls are made:

```bash
python scripts/run_inspect.py runs/oct-olmo/spec.py \
  --from-log runs/oct-olmo/inspect_logs \
  --allow-missing --upload
```

A directory selects its newest log; pass an exact `.eval` filename to select a
specific run. The spec must still match the log's model order, criteria, and
scenario selection. The command computes scores and the configured bootstrap
from successful judgments, then uses `HF_TOKEN` (or cached Hugging Face login)
to upload the local analysis, original log, and omission report directly to the
ValueArena dataset. It uses the spec's upload name and group; it does not send
this run to the remote scoring Space. Without `--upload`, it only analyzes
locally.

The `--allow-missing` flag is required to tolerate failed judgments. Scores and
bootstrap intervals describe the observed subset and may be biased by failures.
The analysis records this explicitly; ValueArena displays the log's omissions.
Duplicate/unexpected judgments and mismatched scenario text still fail. The
normal collection path retains its strict coverage check. Use a separate run
folder if an existing evaluations export differs from the chosen log.
