# Verifying the release

Run commands from the repository root. Commands below do not call paid model APIs.

## Reference totals without dependencies

```console
python -X utf8 run_bench.py aggregate --annotations reference_data/reference/annotations.jsonl --output results/reference540.json
```

Expected trajectories: 291 pass, 220 fail, 29 unknown. Expected request groups: 506 pass, 321 fail, 47 unknown.

## Full offline verification

After installing `requirements.txt`:

```console
python -X utf8 verify_release.py --tests
```

The verifier blocks network connections. It checks the fixed task catalogs, reference hashes, all 540 evidence imports, frozen aggregation, saved-episode conversion, report/review layers, and unit tests. Scripted tools and model replies exercise interfaces without constituting a new real-model experiment.

## Initial-versus-reviewed comparison

```console
python -X utf8 analysis/score_comparison/compare_scores.py --raw-scores analysis_source/initial_scores540 --reference reference_data/reference/annotations.jsonl --run-index reference_data/run_index.json --out results/score_comparison
```

Expected paired cohort: 527 complete initial scores; 13 incomplete outputs remain separate. Expected trajectory verdict agreement: 459/527. The public-score container hashes are checked separately from the original pre-cleanup hashes.

## Check files before publication

```console
python -X utf8 tools/check_public_files.py
```

This local check reports likely credentials, private file paths, and forbidden file types without printing candidate secret values. It is a practical check, not a guarantee that any arbitrary future commit is safe. Fixed benchmark phone/payment fixtures, documented loopback defaults, and upstream sandbox example paths are not real experiment credentials.

Exact local preparation results are recorded in `PUBLICATION_CHECKS.json`. The GitHub workflow repeats the offline checks on Python 3.12 and 3.13 when it runs; a workflow file alone is not a claim that hosted CI has already passed.

During local archive verification, all 57 tests passed but interpreter shutdown initially blocked in LiteLLM's async-client exit hook. This affected the offline verifier, not task scores. The verifier was fixed to remove that unused hook in its no-live-transport mode; the corrected command passed all 57 tests and exited with code 0. The publication copy does not alter the task runner's live-client cleanup.
