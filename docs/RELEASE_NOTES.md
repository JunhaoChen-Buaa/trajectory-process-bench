# Release 0.1.1

This release packages the validated v6 delivery for https://github.com/JunhaoChen-Buaa/trajectory-process-bench. The v0.1.1 tag fixes the distributed code and reference-data version; the initial v0.1.0 tag remains available.

## POSIX report fix

The first hosted Linux check found that source-review report generation opened a relative evidence path through an output directory that did not yet exist. Windows normalized that path during file opening, but POSIX rejected the missing intermediate directory. The report command now resolves the evidence path before reading it. A regression test covers report generation with source reviews into a new nested directory. No scoring rules, annotations, or experiment counts changed.

## Added or rewritten

- English README with installation, model configuration, custom-agent integration, task execution, saved-trajectory scoring, and report access.
- English agent-interface, format, compatibility, data, and verification documentation.
- MIT license for project contributions; explicit data scope and retained upstream MIT notices.
- Citation metadata, `.gitignore`, a local public-file scanner, and an offline CI workflow.
- `.gitattributes` preserves file bytes across platforms so Git line-ending conversion cannot invalidate frozen-data hashes.
- An offline-verifier shutdown fix: LiteLLM's import-created, unused async-client cleanup hook tried to create a new event loop at exit and blocked on a loopback socket pair on the restricted Windows host. The verifier now removes that hook after its no-network scripted checks. Normal task execution and its client cleanup are unchanged.

## Removed from the public copy

- Generated verification output and machine-specific packaging audits.
- Redundant installation-acceptance runs; one smaller real replay remains.
- Private local `raw_path` pointers in 540 automatic-score containers.
- Absolute backend paths in the retained standalone replay; repository-relative paths replace them.

The historical original-score hashes are retained. Separate published hashes verify the metadata-cleaned containers. The comparison reader was updated only to accept that explicit distinction. Its verdict calculation, cohort, and aggregation are unchanged.

## Preserved

All 540 reference evidence packets, reviewed annotations, task files, policy indices, historical schemas, and frozen statistics remain byte-identical to the validated delivery. The process rubric, runtime scoring, grouping, aggregation, agent capture, and reporting code are unchanged. Upstream source and license files are preserved.

No model was rerun, no successful answer was resampled, and no score or issue label was adjusted for publication. Original research files and earlier delivery packages remain untouched.

`PUBLICATION_PROVENANCE.json` records publication transformations without exposing private local paths. `RELEASE_MANIFEST_SHA256.json` verifies the distributed file bytes.
