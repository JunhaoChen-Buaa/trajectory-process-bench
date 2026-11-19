# Historical references and current runtime outputs

The repository retains different kinds of records for different purposes:

| File | Purpose | Entry point |
|---|---|---|
| `reference_data/inputs/trNNN.json` | Frozen source evidence for scoring | `score --input` |
| `reference_data/reference/annotations.jsonl` | The paper's 540 reviewed reference judgments | `aggregate --annotations` |
| `analysis_source/initial_scores540/trNNN.json` | Historical first automatic-score containers, including original failures | Comparison script |
| A new run's `records.jsonl` | Current automatic/reviewed output records | `report --records` |

## Why the formats are kept separate

The historical comparison uses the original completeness rules: 527 initial outputs were complete, and 13 were incomplete. New validation requirements must not silently change that historical paired cohort.

The current runtime validates severity values, requirement identifiers, revision fields, and evidence references more strictly. Applied directly to historical files, the current validator finds format differences in 34 of the 527 originally complete outputs and 46 of the 540 reviewed reference annotations. These are format compatibility counts, not newly failed model runs or newly discovered business errors.

Differences include historical `auxiliary`/`boundary` severity labels, references to annotation entities such as groups or issues, and shorthand requirement/revision fields. New model outputs follow the current schema. The original reference files are retained without adding invented fields or changing labels to satisfy a later schema.

## Which files to use

Use `aggregate` to reproduce reference statistics. Aggregation reads the six dimension states and applies the fixed hierarchy; it does not reclassify historical outputs using the new validator. The historical schema is in `reference_data/scoring/annotation_schema.json`; the current contract is described in [FORMAT.md](FORMAT.md) and implemented in `src/trajectory_bench/grading.py`.

Use `score` with actual episode/evidence data to assess a new agent. Do not substitute historical annotations or old API output containers for new wrapper records.

Compatibility diagnostics are in `analysis/score_comparison/release_validator_compatibility.json` and `final_reference_release_compatibility.json`. They explain format differences without changing historical labels, original completeness decisions, or denominators.

## Publication metadata cleanup

The public copies of initial-score containers omit a private `raw_path` pointer. Their model answers and validation flags are unchanged. Original byte hashes and public byte hashes are recorded separately in `analysis_source/comparison_sources.json`; the reproduction script checks the latter. This metadata cleanup does not make the old containers conform to the current runtime schema.
